import os
import sys
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
try:
    from .retrieval import bm25_matches, load_bm25, retrieve
except ImportError:
    from retrieval import bm25_matches, load_bm25, retrieve
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
EMB = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
LLM = os.getenv('CHAT_MODEL', 'qwen3.5-9b')
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
TOP_N = 3
def open_bm25(col):
    """BM25 is optional: without it retrieval is dense-only, never a crash."""
    data = load_bm25(BM25_FILE)
    if data is None:
        print('warning: no BM25 index at ' + str(BM25_FILE) + ' - dense-only retrieval', file=sys.stderr)
        return None
    ok, why = bm25_matches(data, col)
    if not ok:
        print('warning: ' + why + ' - dense-only retrieval', file=sys.stderr)
        return None
    return data
q = ' '.join(sys.argv[1:]) or input('Vopros: ')
e = requests.post(URL+'/embeddings', json={'model': EMB, 'input': [q]}, timeout=60).json()['data'][0]['embedding']
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
ids, paths, docs, metas, used_bm25 = retrieve(col, q, e, open_bm25(col), 'adaptive')
rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, device='cpu')
sc = rr.predict([(q, d) for d in docs])
top = sorted(range(len(docs)), key=lambda i: -sc[i])[:TOP_N]
ctx = ''
for n, i in enumerate(top, 1):
    ctx = ctx + '=== Istochnik ' + str(n) + ': ' + metas[i]['name'] + ' ===' + chr(10) + docs[i] + chr(10)+chr(10)
sysmsg = 'You are an assistant answering questions about a document archive. RULES: 1) Answer ONLY from the excerpts below. 2) Answer in the same language as the question. 3) Cite source numbers in brackets, like [1]. 4) If the answer is not in the excerpts, say so plainly in one sentence. 5) Be brief - at most 5 sentences. Answer directly; do not think step by step.'
usr = 'VYPISKI:' + chr(10)+chr(10) + ctx + chr(10) + 'VOPROS: ' + q
r = requests.post(URL+'/chat/completions', json={'model': LLM, 'messages': [{'role':'system','content':sysmsg},{'role':'user','content':usr}], 'temperature': 0.2, 'max_tokens': 3000, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=600)
print(chr(10) + '=== OTVET (' + ('dense+BM25' if used_bm25 else 'dense') + ') ===' + chr(10))
mm = r.json()['choices'][0]['message']
print(mm.get('content') or mm.get('reasoning_content') or '[pusto]')
print(chr(10) + '=== ISTOCHNIKI ===')
for n, i in enumerate(top, 1):
    print(str(n) + '. ' + metas[i]['name'] + '  [' + format(float(sc[i]), '.2f') + ']')
    print('   ' + metas[i]['path'])
