import json, random, os, re, requests, chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))

URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/chat/completions'
LLM = os.getenv('CHAT_MODEL', 'qwen3.5-9b')
OUT = str(RAG_DIR / 'evalset_9b.jsonl')
N = 60
random.seed(42)

WEIRD = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ²³¶µ§'

SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'q',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {'question': {'type': 'string'}},
            'required': ['question'],
        },
    },
}

BAD_START = ('maybe', 'but ', 'need ', 'or ', 'ensure', 'does ', 'that ',
             'which ', 'so ', 'perhaps', 'let ', 'we ', 'i ', 'excerpt',
             'should', 'the excerpt', 'is this', 'wait', 'attempt')


def is_mojibake(t):
    s = t[:3000]
    arm = sum(1 for c in s if '\u0530' <= c <= '\u058f')
    cyr = sum(1 for c in s if '\u0400' <= c <= '\u04ff')
    weird = sum(1 for c in s if c in WEIRD)
    return weird > 40 and weird > (arm + cyr)


def bad(s):
    if len(s) < 30:
        return 'too short'
    if not (s.endswith('?') or s.endswith('\u055e')):
        return 'no question mark'
    if s.lower().startswith(BAD_START):
        return 'deliberation'
    if '*' in s or s.count('"') > 2:
        return 'markup junk'
    return None


def detect_lang(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 'Unknown'
    lat = sum(1 for c in letters if 'a' <= c.lower() <= 'z') / len(letters)
    arm = sum(1 for c in letters if '\u0530' <= c <= '\u058f') / len(letters)
    if lat > 0.5:
        return 'English'
    if arm > 0.3:
        return 'Armenian'
    return 'Russian'


def extract(msg):
    raw = (msg.get('content') or '').strip()
    if raw:
        try:
            q = json.loads(raw).get('question', '').strip()
            if len(q) > 10:
                return q
        except:
            pass
    blob = raw + ' ' + (msg.get('reasoning_content') or '')
    m = re.findall(r'"([^"]{15,200}[?\u055e])"', blob)
    if m:
        return m[-1].strip()
    m = re.findall(r'([^.!?\n"]{15,200}[?\u055e])', blob)
    return m[-1].strip() if m else ''


col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection('water')

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding='utf-8'):
        try:
            done.add(json.loads(line)['chunk_id'])
        except:
            pass
print('already done:', len(done), flush=True)

all_ids = col.get(include=[])['ids']
random.shuffle(all_ids)

PROMPT = ('CRITICAL: write the question in {lang}. '
          'Read the excerpt from a water-sector document. '
          'Produce ONE question in {lang} that this excerpt answers. '
          'Paraphrase, do not copy the wording. Make it specific. '
          'The question must end with a question mark. '
          'Reply with JSON only: {{"question": "..."}}')

made = skipped = failed = 0
for cid in all_ids:
    if made >= N:
        break
    if cid in done:
        continue
    rec = col.get(ids=[cid], include=['documents', 'metadatas'])
    txt = rec['documents'][0]
    meta = rec['metadatas'][0]
    if len(txt.strip()) < 400:
        continue
    if sum(1 for c in txt if c.isalpha()) < len(txt) * 0.5:
        continue
    if is_mojibake(txt):
        skipped += 1
        continue
    lang = 'Russian' if made % 3 == 0 else 'Armenian'
    body = {
        'model': LLM,
        'messages': [
            {'role': 'system', 'content': PROMPT.format(lang=lang)},
            {'role': 'user', 'content': txt[:2500]},
        ],
        'temperature': 0.7,
        'max_tokens': 2500,
        'response_format': SCHEMA,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    try:
        r = requests.post(URL, json=body, timeout=1800)
        q = extract(r.json()['choices'][0]['message'])
    except Exception as e:
        print('error:', str(e)[:80], flush=True)
        continue

    why = bad(q)
    if why:
        failed += 1
        print('rejected (' + why + '):', q[:60], flush=True)
        continue

    actual = detect_lang(q)
    row = {'question': q, 'lang': lang, 'actual_lang': actual,
           'chunk_id': cid, 'doc': meta['name'], 'path': meta['path'],
           'chunk': txt[:1200]}
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    made += 1
    print(f'{made}/{N} [{actual}] {q[:70]}', flush=True)

print(f'done. mojibake skipped: {skipped}, rejected: {failed}', flush=True)