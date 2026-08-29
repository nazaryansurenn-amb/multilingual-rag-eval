"""Shared retrieval: tokenisation, language detection, and dense + BM25 fusion.

Imported by the BM25 builder, the evaluation harness and every serving path,
so that all of them tokenise identically. A tokenisation mismatch between the
index and the query degrades retrieval silently — there is no error to notice.

Fusion is Reciprocal Rank Fusion: each retriever contributes 1/(RRF_K + rank)
per candidate. It combines rankings rather than scores, which matters because
cosine similarity and BM25 scores are on unrelated scales and cannot be added
without a weight — and a weight is a parameter to overfit on 50 questions.
"""
import pickle
import re
import unicodedata
from pathlib import Path

# Measured constants. Changing any of them invalidates the recorded results.
POOL = 25       # candidates drawn from each retriever before fusion
TOP_K = 25      # candidates kept after fusion, handed to the reranker
RRF_K = 60      # RRF damping constant

# BM25 matches literal tokens, so it only helps when the query language matches
# the document language. This corpus is predominantly Armenian.
# Known limitation: this is a single hardcoded language, not a per-document
# comparison. It stops being correct once the Russian legacy files are indexed.
CORPUS_LANG = 'Armenian'
ARM_THRESHOLD = 0.5

BM25_MODES = ('adaptive', 'always', 'never')


def tokenize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return [t for t in re.split(r'[^\w]+', text, flags=re.UNICODE) if len(t) > 1]


def detect_lang(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 'Unknown'
    arm = sum(1 for c in letters if '԰' <= c <= '֏') / len(letters)
    return 'Armenian' if arm >= ARM_THRESHOLD else 'Other'


def rrf(dense, kw, k=RRF_K):
    s = {}
    for rank, cid in enumerate(dense, 1):
        s[cid] = s.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(kw, 1):
        s[cid] = s.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(s, key=lambda c: -s[c])


def load_bm25(path):
    """Load a BM25 pickle, or return None if it is not there."""
    if not path or not Path(path).exists():
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return {'bm25': d['bm25'], 'ids': d['chunk_ids'], 'paths': d['paths']}


def bm25_matches(bm25_data, col):
    """(ok, message) — whether a BM25 index was built from this collection."""
    n_bm25 = len(bm25_data['ids'])
    n_col = len(col.get(include=[])['ids'])
    if n_bm25 != n_col:
        return False, (f'BM25 index has {n_bm25} chunks, collection has {n_col} '
                       f'— built from different data')
    return True, ''


def wants_bm25(query, mode):
    if mode == 'never':
        return False
    if mode == 'always':
        return True
    return detect_lang(query) == CORPUS_LANG


def retrieve(col, query, embedding, bm25_data=None, mode='adaptive'):
    """Top-TOP_K candidates for a query.

    Returns (ids, paths, texts, metas, used_bm25). With mode 'never', or no
    BM25 index, this is plain dense retrieval of POOL candidates.
    """
    use_bm25 = bm25_data is not None and wants_bm25(query, mode)

    dense = col.query(query_embeddings=[embedding], n_results=POOL)
    dense_ids = dense['ids'][0]
    text_by_id = dict(zip(dense_ids, dense['documents'][0]))
    meta_by_id = dict(zip(dense_ids, dense['metadatas'][0]))
    path_by_id = {c: m['path'] for c, m in meta_by_id.items()}

    if use_bm25:
        scores = bm25_data['bm25'].get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda j: -scores[j])[:POOL]
        kw_ids = [bm25_data['ids'][j] for j in top]
        for j in top:
            path_by_id.setdefault(bm25_data['ids'][j], bm25_data['paths'][j])

        cand = rrf(dense_ids, kw_ids)[:TOP_K]
        missing = [c for c in cand if c not in text_by_id]
        if missing:
            got = col.get(ids=missing, include=['documents', 'metadatas'])
            # Chroma does NOT return rows in the order requested. Pair on the
            # ids it hands back, never on the list that was asked for.
            text_by_id.update(zip(got['ids'], got['documents']))
            meta_by_id.update(zip(got['ids'], got['metadatas']))
    else:
        cand = dense_ids

    paths = [path_by_id[c] for c in cand]
    texts = [text_by_id[c] for c in cand]
    metas = [meta_by_id.get(c) or {'path': path_by_id[c], 'name': '', 'folder': ''}
             for c in cand]
    return cand, paths, texts, metas, use_bm25
