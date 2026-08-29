import os, sys, requests, chromadb
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
try:
    from .retrieval import bm25_matches, load_bm25, retrieve
except ImportError:
    from retrieval import bm25_matches, load_bm25, retrieve
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
MODEL = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
TOP_N = 5
def embed(t):
    r = requests.post(URL, json={'model': MODEL, 'input': [t]}, timeout=60)
    return r.json()['data'][0]['embedding']
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
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
ids, paths, docs, metas, used_bm25 = retrieve(col, q, embed(q), open_bm25(col), 'adaptive')
print('kandidatov: ' + str(len(docs)) + ' (' + ('dense+BM25' if used_bm25 else 'dense') + '), ranjiruem...')
rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, device='cpu')
scores = rr.predict([(q, d) for d in docs])
order = sorted(range(len(docs)), key=lambda i: -scores[i])[:TOP_N]
for rank, i in enumerate(order, 1):
    print('--- ' + str(rank) + ' | score: ' + format(float(scores[i]), '.3f') + ' | ' + metas[i]['name'])
    print('    papka: ' + metas[i]['folder'])
    print(docs[i][:400].replace(chr(10), ' '))
    print('')
