import os
import sys
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from fastmcp import FastMCP
try:
    from .retrieval import bm25_matches, load_bm25, retrieve
except ImportError:
    from retrieval import bm25_matches, load_bm25, retrieve
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
EMB = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
mcp = FastMCP('water-docs')
_rr = None
def rr():
    global _rr
    if _rr is None:
        _rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, device='cpu')
    return _rr
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
# Warnings go to stderr: this server speaks MCP over stdout, and anything
# printed there corrupts the protocol.
if not BM25_FILE.exists():
    print('warning: no BM25 index at ' + str(BM25_FILE) + ' - dense-only retrieval', file=sys.stderr)
_bm25 = None
_bm25_loaded = False
def bm25():
    """Loaded on first query so the server always starts, with or without it."""
    global _bm25, _bm25_loaded
    if not _bm25_loaded:
        _bm25_loaded = True
        data = load_bm25(BM25_FILE)
        if data is not None:
            ok, why = bm25_matches(data, col)
            if ok:
                _bm25 = data
            else:
                print('warning: ' + why + ' - dense-only retrieval', file=sys.stderr)
    return _bm25
@mcp.tool()
def search_documents(query: str, top_n: int = 5) -> str:
    """Search a water-sector document archive (2,209 documents, 14,252 chunks; Armenian, Russian and English). Hybrid dense + BM25 retrieval with reranking. Returns relevant excerpts with source filenames."""
    e = requests.post(URL, json={'model': EMB, 'input': [query]}, timeout=60).json()['data'][0]['embedding']
    _ids, _paths, docs, metas, used_bm25 = retrieve(col, query, e, bm25(), 'adaptive')
    sc = rr().predict([(query, d) for d in docs])
    top = sorted(range(len(docs)), key=lambda i: -sc[i])[:top_n]
    out = []
    for n, i in enumerate(top, 1):
        out.append('=== Source ' + str(n) + ': ' + metas[i]['name'] + ' (score ' + format(float(sc[i]), '.2f') + ') ===')
        out.append('Path: ' + metas[i]['path'])
        out.append(docs[i][:2000])
        out.append('')
    if not out:
        return 'Nothing found.'
    out.append('[retrieval: ' + ('dense+BM25' if used_bm25 else 'dense only') + ']')
    return chr(10).join(out)
if __name__ == '__main__':
    mcp.run()
