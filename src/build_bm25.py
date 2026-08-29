"""Build a BM25 keyword index over a Chroma collection.

    python src/build_bm25.py --collection water2 --out bm25_water2.pkl

The pickle holds the BM25 model plus the chunk ids and document paths, in the
same order as the tokenised corpus, so a query can map BM25 row numbers back
to chunks. Tokenisation comes from src/retrieval.py — the query side imports
the same function, and the two must never drift apart.

The output is derived from document content and is git-ignored.
"""
import argparse
import os
import pickle
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

try:
    from .retrieval import tokenize
except ImportError:
    from retrieval import tokenize

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))

BATCH = 500


def build(col):
    ids = col.get(include=[])['ids']
    print('chunks    :', len(ids), flush=True)

    chunk_ids, paths, corpus = [], [], []
    for i in range(0, len(ids), BATCH):
        batch = col.get(ids=ids[i:i + BATCH], include=['documents', 'metadatas'])
        # Pair on the ids Chroma returns, not the ids requested — it does not
        # preserve the requested order.
        for cid, doc, meta in zip(batch['ids'], batch['documents'], batch['metadatas']):
            chunk_ids.append(cid)
            paths.append(meta['path'])
            corpus.append(tokenize(doc))
        print('  tokenised', min(i + BATCH, len(ids)), '/', len(ids), flush=True)

    print('tokens    :', sum(len(c) for c in corpus))
    print('vocabulary:', len(set(t for c in corpus for t in c)))
    print('building BM25...', flush=True)
    return {'bm25': BM25Okapi(corpus), 'chunk_ids': chunk_ids, 'paths': paths}


def main():
    ap = argparse.ArgumentParser(description='Build a BM25 index over a Chroma collection.')
    ap.add_argument('--collection', required=True,
                    help='source Chroma collection (no default, on purpose)')
    ap.add_argument('--out', required=True,
                    help='output pickle, relative to RAG_DIR')
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = RAG_DIR / out
    if out.exists():
        raise SystemExit(f'{out} already exists — delete it first, or choose another --out')

    db = RAG_DIR / 'chroma'
    print(f'store     : {db}')
    print(f'collection: {args.collection}')
    print(f'out       : {out}')
    print(flush=True)

    col = chromadb.PersistentClient(path=str(db)).get_collection(args.collection)
    data = build(col)

    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print('saved', out, round(out.stat().st_size / 1024 / 1024, 1), 'MB')


if __name__ == '__main__':
    main()
