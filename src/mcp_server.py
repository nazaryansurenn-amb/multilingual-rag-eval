import os, requests, chromadb
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from fastmcp import FastMCP
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
EMB = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')
COLLECTION = 'water2'
mcp = FastMCP('water-docs')
_rr = None
def rr():
    global _rr
    if _rr is None:
        _rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, device='cpu')
    return _rr
col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(COLLECTION)
@mcp.tool()
def search_documents(query: str, top_n: int = 5) -> str:
    """Search a water-sector document archive (2,209 documents, 14,252 chunks; Armenian, Russian and English). Returns relevant excerpts with source filenames."""
    e = requests.post(URL, json={'model': EMB, 'input': [query]}, timeout=60).json()['data'][0]['embedding']
    res = col.query(query_embeddings=[e], n_results=25)
    docs, metas = res['documents'][0], res['metadatas'][0]
    sc = rr().predict([(query, d) for d in docs])
    top = sorted(range(len(docs)), key=lambda i: -sc[i])[:top_n]
    out = []
    for n, i in enumerate(top, 1):
        out.append('=== Source ' + str(n) + ': ' + metas[i]['name'] + ' (score ' + format(float(sc[i]), '.2f') + ') ===')
        out.append('Path: ' + metas[i]['path'])
        out.append(docs[i][:2000])
        out.append('')
    return chr(10).join(out) if out else 'Nothing found.'
if __name__ == '__main__':
    mcp.run()
