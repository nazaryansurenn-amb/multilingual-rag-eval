"""Chunk a corpus, embed it, and write the vectors to a Chroma collection.

    python src/index.py --corpus corpus2.jsonl --collection water2
    python src/index.py --corpus corpus2.jsonl --collection water3 --with-header

Merges the former index.py (collection `water`, header prepended to the
embedded text) and index2.py (collection `water2`, no header). --with-header
selects between them.

Chunk size and overlap are hardcoded: they are experimental parameters, not
configuration. --collection has no default, and an existing collection is
never replaced without --overwrite.
"""
import argparse
import json
import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/') + '/embeddings'
MODEL = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')

SIZE, OVERLAP, BATCH = 1500, 200, 16


def chunks(text):
    step = SIZE - OVERLAP
    return [text[i:i+SIZE] for i in range(0, len(text), step)
            if len(text[i:i+SIZE].strip()) > 100]


def embed(texts):
    r = requests.post(URL, json={'model': MODEL, 'input': texts}, timeout=300)
    r.raise_for_status()
    return [d['embedding'] for d in r.json()['data']]


def open_collection(client, name, overwrite):
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name in existing:
        if not overwrite:
            raise SystemExit(
                f"collection '{name}' already exists. Pass --overwrite to replace it, "
                f"or choose another --collection.")
        print(f"replacing existing collection '{name}'", flush=True)
        client.delete_collection(name)
    return client.create_collection(name, metadata={'hnsw:space': 'cosine'})


def build(corpus_path, col, with_header):
    buf_txt, buf_meta, buf_ids = [], [], []
    total = 0

    def flush():
        nonlocal buf_txt, buf_meta, buf_ids, total
        if not buf_txt:
            return
        col.add(ids=buf_ids, documents=buf_txt,
                embeddings=embed(buf_txt), metadatas=buf_meta)
        total += len(buf_txt)
        print('проиндексировано:', total, flush=True)
        buf_txt, buf_meta, buf_ids = [], [], []

    with open(corpus_path, encoding='utf-8') as f:
        for doc_i, line in enumerate(f):
            d = json.loads(line)
            for ci, ch in enumerate(chunks(d['text'])):
                text = ch
                if with_header:
                    text = f"[Dokument: {d['name']} | Papka: {d['folder']}]\n" + ch
                buf_txt.append(text)
                buf_meta.append({'name': d['name'], 'folder': d['folder'],
                                 'path': d['path']})
                buf_ids.append(f'{doc_i}_{ci}')
                if len(buf_txt) >= BATCH:
                    flush()
    flush()
    return total


def main():
    ap = argparse.ArgumentParser(description='Chunk, embed and index a corpus.')
    ap.add_argument('--corpus', required=True,
                    help='input jsonl, relative to RAG_DIR')
    ap.add_argument('--collection', required=True,
                    help='target Chroma collection (no default, on purpose)')
    ap.add_argument('--with-header', dest='with_header', action='store_true',
                    help='prepend [Dokument: name | Papka: folder] to the embedded text')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace the collection if it already exists')
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = RAG_DIR / corpus_path
    db = RAG_DIR / 'chroma'

    print(f'corpus     : {corpus_path}')
    print(f'store      : {db}')
    print(f'collection : {args.collection}')
    print(f'header     : {"on" if args.with_header else "off"}')
    print(f'chunk      : {SIZE} chars, overlap {OVERLAP}, batch {BATCH}')
    print(f'embedder   : {MODEL} @ {URL}')
    print(flush=True)

    if not corpus_path.exists():
        raise SystemExit(f'corpus not found: {corpus_path}')

    client = chromadb.PersistentClient(path=str(db))
    col = open_collection(client, args.collection, args.overwrite)
    total = build(corpus_path, col, args.with_header)
    print(f'\nГотово. Всего кусков: {total}', flush=True)


if __name__ == '__main__':
    main()
