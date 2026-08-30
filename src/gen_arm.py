import json
import random
import os
import re
import statistics
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))

URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/chat/completions'
LLM = os.getenv('ARM_CHAT_MODEL', 'armeniangpt-1.0-3b')
OUT = str(RAG_DIR / 'evalset_arm.jsonl')
COLL = 'water2'
N = 50
RETRIES = 3
random.seed(777)

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


def arm_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if '\u0530' <= c <= '\u058f') / len(letters)


def good_chunk(txt):
    """Отсекаем таблицы, анкеты, спецификации — оставляем связный текст."""
    if len(txt) < 700:
        return False
    if arm_ratio(txt) < 0.75:
        return False
    words = [w for w in txt.split() if len(w) > 1]
    if not words:
        return False
    if statistics.mean(len(w) for w in words) < 4.5:
        return False
    if sum(1 for c in txt if c.isdigit()) / len(txt) > 0.07:
        return False
    return True


def bad(s):
    if len(s) < 30:
        return 'too short'
    if not (s.endswith('?') or s.endswith('\u055e') or '\u055e' in s):
        return 'no question mark'
    if arm_ratio(s) < 0.7:
        return 'not armenian'
    if '\\boxed' in s or '$$' in s or '**' in s:
        return 'markup junk'
    if s.lower().startswith(('\u0574\u0565\u056f \u0570\u0561\u0580\u0581', 'here', 'question')):
        return 'preamble'
    return None


def extract(msg):
    raw = (msg.get('content') or '').strip()
    if raw:
        try:
            q = json.loads(raw).get('question', '').strip()
            if len(q) > 10:
                return q
        except Exception:
            pass
    blob = raw + ' ' + (msg.get('reasoning_content') or '')
    m = re.findall(r'"([^"]{15,250}[?\u055e])"', blob)
    if m:
        return m[-1].strip()
    m = re.findall(r'([^.!?\n"]{20,250}[?\u055e])', blob)
    return m[-1].strip() if m else ''


col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLL)

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding='utf-8'):
        try:
            done.add(json.loads(line)['chunk_id'])
        except Exception:
            pass
print('already done:', len(done), flush=True)

all_ids = col.get(include=[])['ids']
random.shuffle(all_ids)

PROMPT = ('Կարդա տեքստը և գրիր ՄԵԿ հարց հայերենով, որին այս տեքստը պատասխանում է։ '
          'Մի՛ պատճենիր տեքստի բառերը, ձևակերպիր քո խոսքերով։ '
          'Հարցը պետք է լինի կոնկրետ։ '
          'Պատասխանիր միայն JSON-ով՝ {"question": "..."}')

made = skipped = failed = 0
checked = 0

for cid in all_ids:
    if made >= N:
        break
    if cid in done:
        continue

    rec = col.get(ids=[cid], include=['documents', 'metadatas'])
    txt = rec['documents'][0]
    meta = rec['metadatas'][0]

    checked += 1
    if not good_chunk(txt):
        skipped += 1
        continue

    q = ''
    for attempt in range(RETRIES):
        body = {
            'model': LLM,
            'messages': [
                {'role': 'system', 'content': PROMPT},
                {'role': 'user', 'content': txt[:1800]},
            ],
            'temperature': 0.6 + attempt * 0.2,
            'max_tokens': 600,
            'response_format': SCHEMA,
        }
        try:
            r = requests.post(URL, json=body, timeout=900)
            cand = extract(r.json()['choices'][0]['message'])
        except Exception as e:
            print('  error:', str(e)[:70], flush=True)
            continue
        why = bad(cand)
        if why is None:
            q = cand
            break
        print(f'  попытка {attempt + 1}: {why} | {cand[:50]}', flush=True)

    if not q:
        failed += 1
        continue

    row = {'question': q, 'lang': 'Armenian', 'actual_lang': 'Armenian',
           'chunk_id': cid, 'doc': meta['name'], 'path': meta['path'],
           'chunk': txt[:1200]}
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    made += 1
    print(f'{made}/{N} {q[:75]}', flush=True)

print(f'\nГотово. Сделано: {made}, чанков проверено: {checked}, '
      f'отсеяно как непригодные: {skipped}, провалов генерации: {failed}', flush=True)