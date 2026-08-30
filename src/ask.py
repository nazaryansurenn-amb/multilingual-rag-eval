import os
import sys
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

try:
    from .retrieval import open_bm25, search
except ImportError:
    from retrieval import open_bm25, search
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
LLM = os.getenv('CHAT_MODEL', 'qwen3.5-9b')
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
TOP_N = 3
q = ' '.join(sys.argv[1:]) or input('Vopros: ')
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
hits, used_bm25, _n = search(col, q, open_bm25(BM25_FILE, col), top_n=TOP_N)
ctx = ''
for n, h in enumerate(hits, 1):
    ctx = ctx + '=== Istochnik ' + str(n) + ': ' + h['name'] + ' ===' + chr(10) + h['text'] + chr(10)+chr(10)
sysmsg = 'You are an assistant answering questions about a document archive. RULES: 1) Answer ONLY from the excerpts below. 2) Answer in the same language as the question. 3) Cite source numbers in brackets, like [1]. 4) If the answer is not in the excerpts, say so plainly in one sentence. 5) Be brief - at most 5 sentences. Answer directly; do not think step by step.'
usr = 'VYPISKI:' + chr(10)+chr(10) + ctx + chr(10) + 'VOPROS: ' + q
r = requests.post(URL+'/chat/completions', json={'model': LLM, 'messages': [{'role':'system','content':sysmsg},{'role':'user','content':usr}], 'temperature': 0.2, 'max_tokens': 3000, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=600)
print(chr(10) + '=== OTVET (' + ('dense+BM25' if used_bm25 else 'dense') + ') ===' + chr(10))
mm = r.json()['choices'][0]['message']
print(mm.get('content') or mm.get('reasoning_content') or '[pusto]')
print(chr(10) + '=== ISTOCHNIKI ===')
for n, h in enumerate(hits, 1):
    print(str(n) + '. ' + h['name'] + '  [' + format(h['score'], '.2f') + ']')
    print('   ' + h['path'])
