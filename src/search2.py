import os, sys, requests, chromadb
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
MODEL = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
COLLECTION = 'water2'
TOP_K, TOP_N = 25, 5
def embed(t):
    r = requests.post(URL, json={'model': MODEL, 'input': [t]}, timeout=60)
    return r.json()['data'][0]['embedding']
q = ' '.join(sys.argv[1:]) or input('Vopros: ')
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
res = col.query(query_embeddings=[embed(q)], n_results=TOP_K)
docs = res['documents'][0]
metas = res['metadatas'][0]
print('kandidatov: ' + str(len(docs)) + ', ranjiruem...')
rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, device='cpu')
scores = rr.predict([(q, d) for d in docs])
order = sorted(range(len(docs)), key=lambda i: -scores[i])[:TOP_N]
for rank, i in enumerate(order, 1):
    print('--- ' + str(rank) + ' | score: ' + format(float(scores[i]), '.3f') + ' | ' + metas[i]['name'])
    print('    papka: ' + metas[i]['folder'])
    print(docs[i][:400].replace(chr(10), ' '))
    print('')
