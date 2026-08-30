import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastmcp import FastMCP

try:
    from .retrieval import open_bm25, search
except ImportError:
    from retrieval import open_bm25, search
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
COLLECTION = 'water2'
BM25_FILE = RAG_DIR / 'bm25_water2.pkl'
mcp = FastMCP('water-docs')
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
        _bm25 = open_bm25(BM25_FILE, col, warn=BM25_FILE.exists())
    return _bm25
@mcp.tool()
def search_documents(query: str, top_n: int = 5) -> str:
    """Search a water-sector document archive (2,209 documents, 14,252 chunks; Armenian, Russian and English). Hybrid dense + BM25 retrieval with reranking. Returns relevant excerpts with source filenames."""
    hits, used_bm25, _n = search(col, query, bm25(), top_n=top_n)
    out = []
    for n, h in enumerate(hits, 1):
        out.append('=== Source ' + str(n) + ': ' + h['name'] + ' (score ' + format(h['score'], '.2f') + ') ===')
        out.append('Path: ' + h['path'])
        out.append(h['text'][:2000])
        out.append('')
    if not out:
        return 'Nothing found.'
    out.append('[retrieval: ' + ('dense+BM25' if used_bm25 else 'dense only') + ']')
    return chr(10).join(out)
if __name__ == '__main__':
    mcp.run()
