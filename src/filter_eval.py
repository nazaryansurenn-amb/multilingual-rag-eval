"""Drop questions too vague to have one correct document, and write a copy.

    python src/filter_eval.py --evalset evalset_9b.jsonl --out evalset_9b_filtered.jsonl

This lives outside eval.py on purpose. Evaluation should score every question
it is handed; dropping questions is an editorial decision about the eval set,
so it happens once, visibly, and the result is a file you can inspect.

The VAGUE list is manual. Each entry is a question whose answer is spread over
many documents, so any single ground-truth document would score as a miss.
Read the "dropped" output before trusting a run.
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))

# Questions too vague to have one correct answer -> would give false failures.
VAGUE = (
    'what does this table show',
    'what does the water committee want',
    'what is the specification for this project',
    'regarding what specific action',
    'what are his skills',
    'what changes are made during restoration',
    'is there a discrepancy',
    'what are the numerical values associated',
    'how many doses or units',
    "let's try",
)


def resolve(p):
    """Relative paths are read as living in RAG_DIR, whatever the cwd."""
    p = Path(p)
    return p if p.is_absolute() else RAG_DIR / p


def is_vague(question):
    ql = question.lower()
    return any(v in ql for v in VAGUE)


def main():
    ap = argparse.ArgumentParser(description='Filter vague questions out of an eval set.')
    ap.add_argument('--evalset', required=True,
                    help='jsonl eval set; relative paths resolve against RAG_DIR')
    ap.add_argument('--out', required=True, help='filtered jsonl to write')
    args = ap.parse_args()

    src, dst = resolve(args.evalset), resolve(args.out)
    if src == dst:
        ap.error('--out must differ from --evalset; this writes a copy')

    kept, dropped = [], []
    with open(src, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            (dropped if is_vague(d['question']) else kept).append(d)

    with open(dst, 'w', encoding='utf-8') as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f'read:    {len(kept) + len(dropped)}  {src}')
    print(f'kept:    {len(kept)}  -> {dst}')
    print(f'dropped: {len(dropped)}')
    for d in dropped:
        print('  -', d['question'][:80])


if __name__ == '__main__':
    main()
