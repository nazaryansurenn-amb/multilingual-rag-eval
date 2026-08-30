import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

try:
    from .retrieval import open_bm25, search
except ImportError:
    from retrieval import open_bm25, search
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
TOP_N = 5
q = ' '.join(sys.argv[1:]) or input('Vopros: ')
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
hits, used_bm25, n_cand = search(col, q, open_bm25(BM25_FILE, col), top_n=TOP_N)
print('kandidatov: ' + str(n_cand) + ' (' + ('dense+BM25' if used_bm25 else 'dense') + '), ranjiruem...')
for rank, h in enumerate(hits, 1):
    print('--- ' + str(rank) + ' | score: ' + format(h['score'], '.3f') + ' | ' + h['name'])
    print('    papka: ' + h['folder'])
    print(h['text'][:400].replace(chr(10), ' '))
    print('')
