import os, sys, requests, chromadb
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
MODEL = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
COLLECTION = 'water2'
def embed(t):
    r = requests.post(URL, json={'model': MODEL, 'input': [t]}, timeout=60)
    return r.json()['data'][0]['embedding']
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
q = ' '.join(sys.argv[1:]) or input('Vopros: ')
res = col.query(query_embeddings=[embed(q)], n_results=5)
for i in range(len(res['documents'][0])):
    d = res['documents'][0][i]
    m = res['metadatas'][0][i]
    s = 1 - res['distances'][0][i]
    print('--- ' + str(i+1) + ' | blizost: ' + format(s, '.3f') + ' | ' + m['name'])
    print('    papka: ' + m['folder'])
    print(d[:400].replace(chr(10), ' '))
    print('')
