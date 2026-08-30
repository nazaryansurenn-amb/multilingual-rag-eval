# Agent run logs

Raw traces from `src/agent.py`. The README summarises what these show; this
file keeps the evidence. Model: `qwen3.8-27b` via LM Studio, tool-calling
against `search_documents`, step limit 8.

Question in both runs concerned the Shirak canal.

## v1 — looped until the step limit

```
[step 1] search_documents({'query': 'Shirak canal'})          -> 0.48
[step 1] search_documents({'query': 'Շիրակ ջրաղաց'})          -> 0.58
[step 2] search_documents({'query': 'Շիրակի ջրանցք'})         -> 0.92
[step 2] search_documents({'query': 'canal Shirak irrigation'}) -> 0.70
[step 3] search_documents({'query': '"Շիրակի ջրանցք"'})       -> 0.83
[step 3] search_documents({'query': 'Shirak canal water supply Ajapnyat Tavshut'}) -> 0.16
[step 4] search_documents({'query': 'Շիրակի ջրանցք'})         -> 0.92
[step 5] search_documents({'query': 'Շիրակի ջրանցք'})         -> 0.92
[step 6] search_documents({'query': 'Շիրակի ջրանցք ոռոգում'}) -> 0.95
Stopped after 6 steps without a final answer.
```

Three things are visible in that trace.

**The language switch was emergent.** The first query went out in English and
scored 0.48. Without being told to, the model retried in Armenian. Nothing in
the code branches on language — it inferred the move from the tool description
and the score it got back.

**It mistranslated on the first attempt.** `ջրաղաց` is *mill*; the word for
canal is `ջրանցք`. It corrected itself on the next try, which took it from
0.58 to 0.92.

**It could not stop.** Having scored 0.92 at step 2, it re-ran the identical
query at steps 4 and 5, getting 0.92 both times, then found 0.95 at step 6 and
was cut off by the step limit one step later. The loop had the answer four
steps before it ran out and never recognised it.

## v2 — one search, then an answer

```
[step 1] 'Շիրակի ջրանցք' -> 0.92
[step 2] final answer after 1 searches
```

Correct Armenian on the first attempt, one search instead of six, and a
grounded answer: it cited two filenames, distinguished the Shirak canal from
the neighbouring Ajapnya canal, and stated plainly that the archive holds only
administrative references to the canal and no technical description of it.

## What changed between them

Three changes, in descending order of how much they mattered.

**1. The tool refuses a repeated query.** `run_tool` keeps a `seen` map from
normalised query to best score. A query already run comes back as *"you already
ran this and got 0.92 — answer from those results or search for something
different"* rather than the same hits again. This is the change that actually
fixed the loop, and it is the transferable one: the failure was not that the
model lacked the instruction, it was that the instruction competed with the
model's own judgement and lost. A tool response does not compete. It is the
environment, and the model has to react to it.

**2. The tool volunteers a verdict.** When the best score clears 0.6 the
response appends *"this is a good result, answer from it rather than searching
again"* — the stopping signal arrives attached to the evidence rather than
sitting in a system prompt many turns back.

**3. The system prompt gained an explicit threshold** and a note that canal is
`ջրանցք`, not `ջրաղաց`. This is the weakest of the three. It plausibly explains
the correct first-attempt Armenian in v2, but it is also the part that v1 shows
a model will happily override.
