"""Retrieval evaluation: embeddings-only, then after reranking.

    python src/eval.py --collection water2 --evalset evalset_arm.jsonl --out results_arm.json
    python src/eval.py --collection water2 --evalset evalset_9b.jsonl  --out results_new.json \
                       --compare results_baseline_doc.json

Ground truth is the document path (metadata['path']), not the chunk id. Chunk
ids shift whenever chunking or extraction changes, which would invalidate every
before/after comparison.

Replaces eval_run.py, eval_doc.py, eval_new.py and eval_arm.py.
Note: eval_run.py matched on chunk_id, so results_baseline.json is NOT
reproducible with this script. The other three are.

This script evaluates every question it is given. To drop vague questions,
filter the set first with filter_eval.py.
"""
import argparse
import json
import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
LM_URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
EMB = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')

TOP_K = 25
RERANKER = 'BAAI/bge-reranker-v2-m3'
METRICS = ('recall@25_emb', 'recall@5_emb', 'mrr_emb', 'recall@5_rr', 'mrr_rr')
LANGS = ('English', 'Russian', 'Armenian')


def resolve(p):
    """Relative paths are read as living in RAG_DIR, whatever the cwd."""
    p = Path(p)
    return p if p.is_absolute() else RAG_DIR / p


def embed(text):
    r = requests.post(f'{LM_URL}/embeddings',
                      json={'model': EMB, 'input': [text]}, timeout=120)
    r.raise_for_status()
    return r.json()['data'][0]['embedding']


def first_rank(paths, truth):
    """1-based rank of the ground-truth document; 0 if it is not there at all."""
    for i, p in enumerate(paths, 1):
        if p == truth:
            return i
    return 0


def recall_at(recs, key, n):
    return sum(1 for r in recs if 0 < r[key] <= n) / len(recs) if recs else 0.0


def mrr(recs, key):
    return sum(1 / r[key] for r in recs if r[key] > 0) / len(recs) if recs else 0.0


def summarise(recs):
    return {
        'recall@25_emb': recall_at(recs, 'rank_emb', 25),
        'recall@5_emb': recall_at(recs, 'rank_emb', 5),
        'mrr_emb': mrr(recs, 'rank_emb'),
        'recall@5_rr': recall_at(recs, 'rank_rr', 5),
        'mrr_rr': mrr(recs, 'rank_rr'),
    }


def report(recs, label):
    if not recs:
        return
    s = summarise(recs)
    print(f'\n=== {label} (n={len(recs)}) ===')
    print('  stage 1 (embeddings only)')
    print(f'    recall@25 : {s["recall@25_emb"]:.3f}   <- ceiling for the reranker')
    print(f'    recall@5  : {s["recall@5_emb"]:.3f}')
    print(f'    MRR       : {s["mrr_emb"]:.3f}')
    print('  stage 2 (after reranking)')
    print(f'    recall@5  : {s["recall@5_rr"]:.3f}')
    print(f'    MRR       : {s["mrr_rr"]:.3f}')


def compare(before_path, after):
    with open(before_path, encoding='utf-8') as f:
        before = json.load(f)
    print('\n' + '=' * 46)
    print(f'{"metric":<16}{"before":>9}{"after":>9}{"delta":>10}')
    print('=' * 46)
    for k in METRICS:
        print(f'{k:<16}{before[k]:>9.3f}{after[k]:>9.3f}{after[k] - before[k]:>+10.3f}')
    print(f'\nbefore: {before_path} (n={before.get("n", "?")})')


def evaluate(collection, rows, reranker):
    col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(collection)
    records = []
    for i, d in enumerate(rows, 1):
        q, truth = d['question'], d['path']

        res = col.query(query_embeddings=[embed(q)], n_results=TOP_K)
        docs = res['documents'][0]
        paths = [m['path'] for m in res['metadatas'][0]]

        rank_emb = first_rank(paths, truth)

        scores = reranker.predict([(q, t) for t in docs])
        order = sorted(range(len(docs)), key=lambda j: -scores[j])
        rank_rr = first_rank([paths[j] for j in order], truth)

        records.append({
            'question': q,
            'lang': d.get('actual_lang') or d.get('lang', '?'),
            'truth': truth,
            'doc': d.get('doc', ''),
            'rank_emb': rank_emb,
            'rank_rr': rank_rr,
        })
        mark = 'OK ' if 0 < rank_rr <= 5 else 'MISS'
        print(f'{i}/{len(rows)} {mark} emb#{rank_emb} rr#{rank_rr} | {q[:50]}', flush=True)
    return records


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate retrieval against an eval set.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--collection', required=True,
                    help='chroma collection name, e.g. water2')
    ap.add_argument('--evalset', required=True,
                    help='jsonl eval set; relative paths resolve against RAG_DIR')
    ap.add_argument('--out', required=True,
                    help='results json to write')
    ap.add_argument('--compare', metavar='RESULTS_JSON',
                    help='earlier results json; prints a before/after delta table')
    args = ap.parse_args()

    evalset, out = resolve(args.evalset), resolve(args.out)

    with open(evalset, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f'questions: {len(rows)}', flush=True)

    reranker = CrossEncoder(RERANKER, max_length=1024, device='cpu')
    records = evaluate(args.collection, rows, reranker)

    report(records, 'ALL')
    for lang in LANGS:
        report([r for r in records if r['lang'] == lang], lang)

    summary = {'n': len(records), **summarise(records), 'records': records}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    if args.compare:
        compare(resolve(args.compare), summary)

    print(f'\nsaved to {out}')


if __name__ == '__main__':
    main()
