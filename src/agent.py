"""Agent loop over the archive.

Unlike ask.py, which does one retrieval and one answer, the model here decides
when to search, with what query, and whether the results are good enough -
looping until satisfied or the step limit hits.

    python src/agent.py "who maintains the Shirak canal?"
    python src/agent.py --question "..." --max-steps 8 --model qwen3.8-27b

v2 changes, after watching v1 loop on a query that had already succeeded:
  - the tool refuses to run a query it has already run, and says so
  - the system prompt states an explicit stopping threshold
  - each run prints a trace of the queries tried and their best scores

The repeat guard lives in the tool rather than the prompt on purpose. An
instruction can be outweighed; a tool that refuses cannot.
"""

import argparse
import json
import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

try:
    from .retrieval import open_bm25, search
except ImportError:
    from retrieval import open_bm25, search

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))
LM_URL = os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
CHAT_URL = LM_URL + '/chat/completions'

GOOD_SCORE = 0.6
TOP_N = 4

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'search_documents',
        'description': (
            'Search the Water Committee archive: 2209 documents, mostly Armenian, '
            'some Russian and English. Covers irrigation, water supply, pump '
            'stations, reservoirs, contracts and correspondence from 2015-2025. '
            'Returns excerpts with source filenames and relevance scores. '
            'Each distinct query can only be run once per conversation.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': ('search query; the archive is mostly Armenian, '
                                    'so Armenian queries work best for names and '
                                    'places')
                }
            },
            'required': ['query'],
        },
    },
}]

SYSTEM = (
    'You answer questions about a water-sector document archive using the '
    'search_documents tool.\n\n'
    'Rules:\n'
    '1. Search before answering. Do not answer from your own knowledge.\n'
    '2. Relevance scores tell you whether a search worked:\n'
    '   - above {good}: good result. STOP SEARCHING and write your answer.\n'
    '   - 0.3 to {good}: weak. One more attempt with different wording is fine.\n'
    '   - below 0.3: the search failed. Reformulate.\n'
    '3. The archive is mostly Armenian. Proper nouns and place names often fail '
    'in Russian or English - retry them in Armenian. Note that "canal" is '
    'jrancq, not jraghac, which means mill.\n'
    '4. Never repeat a query you have already run. The tool will refuse it.\n'
    '5. Once you have results above {good}, answer from them. Do not keep '
    'searching for something marginally better.\n'
    '6. Cite the source filename for each claim.\n'
    '7. If the archive does not contain the answer, say so plainly. Do not '
    'invent.'
).format(good=GOOD_SCORE)


def call_model(messages, model):
    body = {'model': model, 'messages': messages, 'tools': TOOLS,
            'temperature': 0.3, 'max_tokens': 2000}
    r = requests.post(CHAT_URL, json=body, timeout=1800)
    r.raise_for_status()
    return r.json()['choices'][0]['message']


def run_tool(name, args, ctx, seen, trace):
    """Execute a tool call.

    `seen` maps normalised queries to their best score, so a repeated query
    gets a refusal instead of identical results. Without this the model can
    loop on a query that already worked, never deciding it is done - which is
    exactly what v1 did.
    """
    if name != 'search_documents':
        return 'unknown tool: ' + name, None

    query = (args.get('query') or '').strip()
    if not query:
        return 'Empty query. Provide a search query.', None

    key = query.lower().strip('"\' ')
    if key in seen:
        msg = ('You already ran this query and got a best score of {:.2f}. '
               'The results are earlier in this conversation. Either answer '
               'from them, or search for something different.').format(seen[key])
        return msg, None

    hits, used_bm25, _n = search(ctx['col'], query, ctx['bm25'], top_n=TOP_N)
    if not hits:
        seen[key] = 0.0
        trace.append((query, 0.0))
        return 'No results.', 0.0

    best = hits[0]['score']
    seen[key] = best
    trace.append((query, best))

    mode = 'dense+BM25' if used_bm25 else 'dense only'
    out = ['[retrieval: {}] best score {:.2f}'.format(mode, best)]
    if best >= GOOD_SCORE:
        out.append('This is a good result. Answer from it rather than '
                   'searching again.')
    for n, h in enumerate(hits, 1):
        out.append('--- result {} | score {:.2f} | {}'.format(
            n, h['score'], h['name']))
        out.append(h['text'][:900])
    return '\n'.join(out), best


def agent(question, ctx, model, max_steps, verbose=True):
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': question}]
    seen = {}
    trace = []

    for step in range(1, max_steps + 1):
        msg = call_model(messages, model)
        calls = msg.get('tool_calls')

        if not calls:
            answer = msg.get('content') or msg.get('reasoning_content') or '(empty)'
            if verbose:
                print('\n[step {}] final answer after {} searches\n'.format(
                    step, len(trace)), flush=True)
            return answer, trace

        messages.append({'role': 'assistant',
                         'content': msg.get('content') or '',
                         'tool_calls': calls})

        for c in calls:
            name = c['function']['name']
            try:
                args = json.loads(c['function']['arguments'])
            except Exception:
                args = {}
            result, best = run_tool(name, args, ctx, seen, trace)
            if verbose:
                q = args.get('query', '')
                if best is None:
                    print('[step {}] {!r} -> refused'.format(step, q), flush=True)
                else:
                    print('[step {}] {!r} -> {:.2f}'.format(step, q, best),
                          flush=True)
            messages.append({'role': 'tool',
                             'tool_call_id': c['id'],
                             'content': result})

    return ('Stopped after {} steps without a final answer.'.format(max_steps),
            trace)


def main():
    ap = argparse.ArgumentParser(description='Agentic search over the archive.')
    ap.add_argument('question', nargs='*', help='the question to answer')
    ap.add_argument('--question', dest='q_flag', help='the question to answer')
    ap.add_argument('--model', default='qwen3.8-27b',
                    help='chat model with tool-calling support (default: qwen3.8-27b)')
    ap.add_argument('--collection', default='water2',
                    help='chroma collection (default: water2)')
    ap.add_argument('--max-steps', dest='max_steps', type=int, default=8,
                    help='hard stop on loop length (default: 8)')
    ap.add_argument('--bm25', default='bm25_water2.pkl',
                    help='BM25 index, relative to RAG_DIR (default: bm25_water2.pkl)')
    args = ap.parse_args()

    question = args.q_flag or ' '.join(args.question) or input('Question: ')

    bm25_path = Path(args.bm25)
    if not bm25_path.is_absolute():
        bm25_path = RAG_DIR / bm25_path

    col = chromadb.PersistentClient(
        path=str(RAG_DIR / 'chroma')).get_collection(args.collection)
    ctx = {'col': col, 'bm25': open_bm25(bm25_path, col)}

    answer, trace = agent(question, ctx, args.model, args.max_steps)
    print(answer)
    if trace:
        print('\n--- search trace ---')
        for q, s in trace:
            print('  {:.2f}  {}'.format(s, q))


if __name__ == '__main__':
    main()
