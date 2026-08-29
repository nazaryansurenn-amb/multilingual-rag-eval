"""Retrieval evaluation: retrieval stage, then after reranking.

    python src/eval.py --collection water2 --evalset evalset_arm.jsonl --out results_arm.json
    python src/eval.py --collection water2 --evalset evalset_arm.jsonl \
                       --out results_arm_adaptive.json --bm25 bm25_water2.pkl \
                       --compare results_arm.json

Ground truth is the document path (metadata['path']), not the chunk id. Chunk
ids shift whenever chunking or extraction changes, which would invalidate every
before/after comparison.

Without --bm25 this is dense-only retrieval. With it, dense and BM25 candidates
are fused with RRF, and --bm25-mode decides when BM25 is consulted:

  adaptive  only when the query language matches the corpus language (default)
  always    unconditionally
  never     dense-only, identical to omitting --bm25

Replaces eval_run.py, eval_doc.py, eval_new.py, eval_arm.py, eval_hybrid.py
and eval_adaptive.py. Note: eval_run.py matched on chunk_id, so
results_baseline.json is NOT reproducible with this script. The rest are.

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

try:
    from .retrieval import BM25_MODES, bm25_matches, load_bm25, retrieve
except ImportError:
    from retrieval import BM25_MODES, bm25_matches, load_bm25, retrieve

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
LM_URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
EMB = os.getenv('EMBED_MODEL', 'text-embedding-bge-m3')

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
    used = sum(1 for r in recs if r.get('used_bm25'))
    print(f'\n=== {label} (n={len(recs)}) ===')
    print('  stage 1 (retrieval)')
    print(f'    recall@25 : {s["recall@25_emb"]:.3f}   <- ceiling for the reranker')
    print(f'    recall@5  : {s["recall@5_emb"]:.3f}')
    print(f'    MRR       : {s["mrr_emb"]:.3f}')
    print('  stage 2 (after reranking)')
    print(f'    recall@5  : {s["recall@5_rr"]:.3f}')
    print(f'    MRR       : {s["mrr_rr"]:.3f}')
    print(f'  BM25 used on {used} of {len(recs)} queries')


def compare(before_path, after):
    with open(before_path, encoding='utf-8') as f:
        before = json.load(f)
    if before.get('n') != after['n']:
        print(f'\nWARNING: {after["n"]} questions vs {before.get("n")} in the '
              f'baseline - not comparable')
    print('\n' + '=' * 46)
    print(f'{"metric":<16}{"before":>9}{"after":>9}{"delta":>10}')
    print('=' * 46)
    for k in METRICS:
        print(f'{k:<16}{before[k]:>9.3f}{after[k]:>9.3f}{after[k] - before[k]:>+10.3f}')
    print(f'\nbefore: {before_path} (n={before.get("n", "?")})')


def evaluate(col, rows, reranker, bm25_data, mode):
    records = []
    for i, d in enumerate(rows, 1):
        q, truth = d['question'], d['path']

        _ids, paths, texts, _metas, used = retrieve(col, q, embed(q), bm25_data, mode)
        rank_emb = first_rank(paths, truth)

        scores = reranker.predict([(q, t) for t in texts])
        order = sorted(range(len(texts)), key=lambda j: -scores[j])
        rank_rr = first_rank([paths[j] for j in order], truth)

        records.append({
            'question': q,
            'lang': d.get('actual_lang') or d.get('lang', '?'),
            'used_bm25': used,
            'truth': truth,
            'doc': d.get('doc', ''),
            'rank_emb': rank_emb,
            'rank_rr': rank_rr,
        })
        tag = 'HYB' if used else 'VEC'
        mark = 'OK ' if 0 < rank_rr <= 5 else 'MISS'
        print(f'{i}/{len(rows)} {tag} {mark} emb#{rank_emb} rr#{rank_rr} | {q[:46]}', flush=True)
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
    ap.add_argument('--bm25', metavar='PICKLE',
                    help='BM25 index from build_bm25.py; enables hybrid retrieval')
    ap.add_argument('--bm25-mode', dest='bm25_mode', choices=BM25_MODES,
                    default='adaptive',
                    help='when to consult BM25 (default: adaptive)')
    args = ap.parse_args()

    evalset, out = resolve(args.evalset), resolve(args.out)

    with open(evalset, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]

    col = chromadb.PersistentClient(path=str(RAG_DIR / 'chroma')).get_collection(args.collection)

    bm25_data, mode = None, 'never'
    if args.bm25:
        bm25_path = resolve(args.bm25)
        bm25_data = load_bm25(bm25_path)
        if bm25_data is None:
            raise SystemExit(f'BM25 index not found: {bm25_path}')
        ok, why = bm25_matches(bm25_data, col)
        if not ok:
            raise SystemExit(why)
        mode = args.bm25_mode

    print(f'collection: {args.collection}')
    print(f'eval set  : {evalset} ({len(rows)} questions)')
    retrieval_line = 'dense only' if bm25_data is None else f'dense + BM25, mode {mode}'
    print(f'retrieval : {retrieval_line}')
    print(flush=True)

    reranker = CrossEncoder(RERANKER, max_length=1024, device='cpu')
    records = evaluate(col, rows, reranker, bm25_data, mode)

    n_bm25 = sum(1 for r in records if r['used_bm25'])
    print(f'\nBM25 used on {n_bm25} of {len(records)} queries')

    report(records, 'ALL')
    for lang in LANGS:
        report([r for r in records if r['lang'] == lang], lang)

    summary = {'n': len(records), 'mode': mode, 'bm25_used': n_bm25,
               **summarise(records), 'records': records}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    if args.compare:
        compare(resolve(args.compare), summary)

    print(f'\nsaved to {out}')


if __name__ == '__main__':
    main()
