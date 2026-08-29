# multilingual-rag-eval

A local retrieval-augmented generation pipeline over ~2,200 real
water-sector documents in Armenian, Russian and English — built to be
**measured**, not just built.

Everything runs on a single consumer laptop (RTX 5070 Ti, 12 GB VRAM).
No document content leaves the machine.

The interesting part of this repository is not the RAG. It is the
evaluation harness, and the fact that it disproved my main hypothesis.

---

## Results

### 1. Text extraction was the bottleneck — not ranking, not embeddings

Three variables changed between the first and second index: the PDF
parser (`pypdf` → `PyMuPDF`), removal of a filename header from the
embedded chunk text, and a filter dropping 19 documents in a legacy
Armenian encoding. The combined gain was +0.078 recall@25 — but a
three-variable change attributes nothing.

So I built a third index to separate them:

| index | corpus | header | mojibake filter |
|---|---|---|---|
| `water` | pypdf | on | no |
| `water2` | PyMuPDF | off | yes |
| `water3` | PyMuPDF | **on** | yes |

`water3` differs from `water2` only in the header, and from `water` only
in the parser and filter. Same eval set, 51 questions, all three.

| metric | water | water2 | water3 |
|---|---|---|---|
| recall@25 (embeddings) | 0.588 | 0.667 | 0.706 |
| recall@5 (embeddings) | 0.431 | 0.510 | 0.471 |
| MRR (embeddings) | 0.361 | 0.411 | 0.376 |
| recall@5 (reranked) | 0.490 | 0.529 | 0.588 |
| MRR (reranked) | 0.386 | 0.453 | 0.492 |

**Parser + encoding filter (water3 − water):**
recall@25 **+0.118**, recall@5 reranked **+0.098**, MRR reranked
**+0.106**. Consistent direction across every metric.

**Header (water2 − water3):** −0.039, +0.039, +0.035, −0.059, −0.039.
The sign is not stable. At n=51 a delta of 0.039 is two questions. **This
is within noise, and my hypothesis about it was wrong.**

I had argued that the header — `[Dokument: New Microsoft Word
Document.docx | Papka: New folder (56)]`, prepended to every chunk —
was polluting every vector with identical boilerplate and pulling them
toward a common point. It sounded right and explained an observation
(every candidate clustering at 0.58). It did not survive measurement.

**Why the parser mattered.** A PDF stores glyphs at coordinates, not
sentences; the parser reconstructs word boundaries from geometry.
`pypdf` concatenated adjacent labels in Armenian engineering drawings —
`ՆՇԱՆՆԵՐՆերանցումային` where there should be two separate words. The
tokenizer then split that into fragments matching nothing, so the
embedding was noise. Retrieval succeeded on 66% of `.docx`-sourced
documents against 29% of PDF-sourced ones, which is what pointed here in
the first place.

`recall@25` is the ceiling of the whole system: if the correct document
is not among the 25 candidates the embedder returns, no reranker can
recover it. **Garbage at extraction time cannot be repaired downstream.**

### 2. Connected prose retrieves far better than tabular content

| metric | mixed set (n=51) | prose-only set (n=50) | delta |
|---|---|---|---|
| recall@25 | 0.667 | 0.780 | +0.113 |
| recall@5 (embeddings) | 0.510 | 0.700 | **+0.190** |
| recall@5 (reranked) | 0.529 | 0.760 | **+0.231** |
| MRR (reranked) | 0.453 | 0.583 | +0.130 |

I first read this as a *language* effect — the prose-only set is
Armenian, the mixed set English and Russian, and the corpus is
predominantly Armenian. That reading was wrong, and the flaw was in how
the sets were built.

The Armenian set was generated with a chunk filter requiring ≥700
characters, ≥75% Armenian letters, average word length ≥4.5 and <7%
digits. The English/Russian set had no such filter. So one set is
connected prose and the other is full of tables, specifications and form
fragments. **The gap measures content type, not language.**

That filter rejected **84% of the Armenian chunks it examined** — 257 of
307. A large share of this archive is tabular, and tabular chunks embed
badly: a row stripped of its column headers is a sequence of numbers with
no recoverable meaning.

This also explains the one part of finding #1 that is not noise. On the
prose-only Armenian set the header consistently *hurt* (four of five
metrics). On the tabular-heavy mixed set it slightly *helped* after
reranking. That is consistent with the header being a substitute for
meaning where the chunk has none of its own — a filename is the only
interpretable text in a chunk that is otherwise a row of numbers.

Separately, one language-specific failure is worth recording: the query
"Ширакский канал" (Russian) scored 0.00, while "Շիրակի ջրանցք" — the same
canal in Armenian — scored 0.93. Cross-lingual matching holds for general
concepts that co-occur in similar contexts across languages, and breaks
on proper nouns, which have no such statistical bridge. That is the
textbook case for hybrid BM25 + dense retrieval, which is next.

### 3. Model size does not substitute for language specialisation

Generating the Armenian eval set:

| model | outcome |
|---|---|
| Qwen 3.5 9B | wrote English despite explicit instructions; ~4 usable Armenian questions out of 61 |
| Qwen 3.8 27B | same failure mode |
| ArmenianGPT 1.0 **3B** | 50/50 valid Armenian questions, zero failures |

A 3B model fine-tuned on the target language beat a 27B general model
decisively — while the 27B ran at 4–5 tok/s under partial GPU offload
and the 3B fit entirely in VRAM.

Both Qwen models also ignored `chat_template_kwargs: {enable_thinking:
false}` over the API and spent their whole token budget on reasoning,
leaving `content` empty. Enforcing a JSON schema through
`response_format` was what actually fixed the output.

---

## Architecture

```
documents ──► extract ──► chunk ──► embed ──────► Chroma
 (docx/xlsx/pdf)                    (bge-m3)      (14,252 chunks)
                                                       │
query ──► embed ──► top-25 ──► rerank ──► top-5 ──► LLM / MCP tool
                              (bge-reranker-v2-m3)
```

- **Embedder:** BAAI/bge-m3 (Q8_0) served by LM Studio
- **Vector store:** ChromaDB, cosine similarity
- **Reranker:** BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder
- **Generation:** local models via LM Studio, or Claude through an MCP server
- **Chunking:** 1,500 characters, 200-character overlap

The retrieval layer is exposed as an MCP server, so an agent can call it
as a tool, judge what comes back and reformulate — rather than getting
one fixed retrieval per question.

---

## How the eval sets were built

Manual ground-truth labelling is the usual bottleneck. This avoids it:

1. Sample a random chunk from the index.
2. Ask a model to write one question that this chunk answers.
3. The chunk's document is, by construction, the correct answer —
   **ground truth for free**.

Two decisions shaped everything downstream:

**Ground truth is the source document path, not the chunk ID.** Chunk IDs
shift whenever chunking or extraction changes, which would have
invalidated every before/after comparison in finding #1 — the whole
three-index experiment depends on this. The cost is that near-duplicate
documents across folders count as misses even when retrieval was
sensible.

**Generated questions need aggressive filtering.** The first run had a
42% junk rate: reasoning models leak deliberation into the output. Fixed
with a JSON schema in `response_format`, a raised token budget, and
validation rejecting Latin-script output where another language was
requested, missing question marks, markdown artefacts and hedge-word
openers.

---

## Metrics

**recall@k** — fraction of questions where the correct document appears
in the top *k*.

**MRR** — mean reciprocal rank: 1st place scores 1.0, 2nd 0.5, 3rd 0.33,
absent 0. Weights the gap between first and second more heavily than
between fourth and fifth, matching how results are consumed.

Both are computed **separately for each retrieval stage**. Stage-1
recall@25 is the ceiling; stage-2 metrics show how well the reranker uses
what it was given. Reporting only the final number hides which half is
broken — here the ceiling was the problem, and the reranker was already
doing its job.

**Noise floor:** at n≈50, one question is worth 0.02. Deltas below ~0.05
are not interpreted as signal. This is why finding #1's header effect is
reported as inconclusive rather than as a small negative result.

---

## Repository layout

```
src/
  extract.py       docx/xlsx/pdf → corpus.jsonl   --parser {pymupdf,pypdf}
  index.py         chunk + embed + write to Chroma  --collection --with-header
  search.py        vector search only
  search2.py       two-stage: vectors + reranker
  ask.py           full RAG with grounded generation
  mcp_server.py    MCP server exposing search_documents
  gen_eval.py      synthetic eval question generation
  gen_arm.py       Armenian variant (ArmenianGPT + strict chunk filter)
  eval.py          measurement: recall@k, MRR, per-language breakdown
  filter_eval.py   drops questions too vague to have one correct answer
```

Both the extractor and the indexer are parameterised rather than
duplicated, so every index in finding #1 is reproducible from one script:

```bash
python src/index.py --corpus corpus2.jsonl --collection water3 --with-header
python src/eval.py --collection water3 --evalset evalset_9b_filtered.jsonl \
                   --out results_header.json --compare results_new.json
```

`index.py` refuses to overwrite an existing collection without
`--overwrite`, because a mistyped name would otherwise destroy an
experimental record.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit paths
```

Start LM Studio with **bge-m3** loaded and the local server on port 1234.

```bash
python src/extract.py --parser pymupdf --out corpus2.jsonl
python src/index.py --corpus corpus2.jsonl --collection water2
python src/eval.py --collection water2 --evalset evalset_arm.jsonl --out results_arm.json
```

---

## Limitations

Stated plainly, because the numbers above only mean something with them.

- **The corpus is not public.** The archive is real working material;
  only code and aggregate metrics are published.
- **Eval sets are small** (51 and 50 questions). See the noise floor
  above.
- **Finding #2 measures content structure, not language**, because the
  two eval sets were built with different chunk filters. Isolating
  language requires regenerating the EN/RU set under the same filter.
- **Questions are model-generated** and inherit some bias toward source
  phrasing, despite instructions to paraphrase and filtering afterwards.
- **The Armenian set was never run against `water`**, so finding #1 has
  no Armenian baseline column.
- **A third of retrievals still fail.** Much of that is tabular content
  that embeds poorly; some is near-duplicate documents scored as misses.
- **Out of scope so far:** 254 scanned PDFs (need OCR), 351 legacy
  `.doc`/`.xls` files, table-aware chunking, hybrid retrieval.

---

## Next

1. **Hybrid BM25 + dense retrieval.** Proper nouns are where dense
   vectors fail, and this archive is full of them. Largest available
   gain.
2. **Table-aware chunking** — carry column headers into each row chunk.
   Finding #2 says this is where a third of the failures live.
3. Regenerate the EN/RU eval set under the same chunk filter as the
   Armenian one, isolating language from content structure.
4. OCR for the 254 scans.

---

## What this was for

Learning the discipline, not the library. Anyone can assemble a RAG from
a tutorial in an afternoon. The part worth practising is being able to
answer *"how do you know your change helped?"* — which requires a
baseline, a metric that stays stable under the change, and the
willingness to run the experiment that shows your explanation was wrong.

The header hypothesis was mine, it was plausible, and it was incorrect.
Finding that out cost one re-index and one evaluation run. Not finding it
out would have meant carrying a wrong belief into the next set of
decisions.
