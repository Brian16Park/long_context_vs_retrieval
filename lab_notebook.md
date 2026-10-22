# Lab notebook: Long Context vs. Retrieval, at Matched Cost

## Part 1: logging and cost model

### Decisions

- **Generator model: Claude Haiku 5.5** (`claude-haiku-5-5`) for both systems. 1M-token context window, cheapest Claude model, and the API reports cache-write and cache-read tokens on every call.
- **Thinking disabled** for both systems (`thinking={"type": "disabled"}`). Haiku 5.5 thinks by default; thinking tokens are billed as output and vary per question, so leaving it on would add cost unrelated to retrieval vs. long context.
- **5-minute cache only.** Every cache write is priced at the 5-minute rate.
- **Compute:** local Windows PC, CPU-only. The model runs on Anthropic's servers; local work (embedding, reranking, vector search) is small, and the cost model prices it as CPU-seconds.
- **Log raw token counts, never dollars.** Dollars = tokens x a frozen price sheet, computed afterwards. This lets us fix formula mistakes and re-price under other providers' rules without re-running anything.
- **Every call is labeled** (run, system, corpus, question_id, ...) and the labels are required, so no call can be logged without saying what it was for.
- **Retries are ours, not the SDK's** (`max_retries=0`), so every attempt, including failures, is logged. Exponential backoff 2, 4, 8, ... s, capped at 60 s per wait; 8 attempts (about 3 minutes in total).
- **Runs are resumable:** a call is "done" when its exact label combination has a successful (`ok: true`) log line. Rerunning a crashed script skips finished calls.

### Price sheet (frozen 2026-10-21)

Source: https://platform.claude.com/docs/en/about-claude/pricing. US dollars per million tokens.

| Prompt length | Input | Cache write (5 min) | Cache read | Output |
|---|---|---|---|---|
| up to 100,000 tokens | 0.10 | 0.125 | 0.01 | 0.50 |
| over 100,000 tokens | 0.50 | 0.625 | 0.05 | 2.50 |

- The tier is decided per request by the total prompt (input + cache write + cache read), and applies to every token in that request, output included.
- Exactly 100,000 tokens is still the cheaper tier.
- Batch API: 50% off (not used yet).

### Findings

**1. Caching works, and its value depends on query volume Q.** Measured on a 6,101-token synthetic document:

| Call | Cache write | Cache read | Cost |
|---|---|---|---|
| 1st question | 6,101 | 0 | $0.00077 |
| 2nd question | 0 | 6,101 | $0.00007 |

Without caching, each call would cost about $0.00062. Measuring document cost in "one uncached read" units (write = 1.25, read = 0.1), the average per question is (1.25 + 0.1 x (Q - 1)) / Q:

| Q | 1 | 2 | 10 | 100 | 1,000 |
|---|---|---|---|---|---|
| Avg cost per question | 1.25 | 0.675 | 0.215 | 0.112 | 0.101 |

At Q = 1 caching is worse than no cache; from Q = 2 it is cheaper, approaching 0.1x. This is the first term of the long-context cost formula in the proposal (§4) and the reason Q is a core axis.

**2. Q means "questions while the cache is alive", not "questions per script run".** The cache lives on Anthropic's servers, keyed to the exact prefix text, and each read resets its 5-minute timer. A second script run 9 seconds after the first got cache reads on every call. **Trap for validation runs:** to measure a truly cold call, wait more than 5 minutes after the last use of that exact corpus, and confirm `cache_write_tokens > 0` in the log.

**3. The price cliff at 100K tokens.** A prompt 10 tokens over the threshold costs about 5x more than one just under it ($0.0050 vs. $0.0010 in our example). Our 64K corpus sits below the cliff and 256K/512K above it, so expect long-context cost to jump between them, a plausible place for the crossover.

**4. Measure tokens, never estimate them.** The synthetic document was estimated at about 4,000 tokens; the API billed 6,101.

**5. The cache needs an exact prefix match.** It stores the model's computed keys and values for every token up to the cache marker. Because tokens only attend backward, the question (placed after the marker) can change freely, but any edit to the document invalidates everything after the edit. Corpus layout must be: [instructions] [corpus + cache marker] [question].

**6. Retry and crash tests behaved as designed.** A permanent error (404, wrong model name) was logged once and not retried. Wi-Fi turned off mid-run produced failed attempts at 2, 4, 8 and 16 s gaps, then success on the 5th attempt. A run killed with Ctrl+C resumed without repeating finished calls. A half-written log line from a crash no longer corrupts the next record.

### Code so far

| File | Purpose |
|---|---|
| `lcvr/llm.py` | `call_claude()`: the single entry point for model calls; caching, labels, retries, logging to `logs/calls.jsonl`; `load_done_keys()` for resuming |
| `lcvr/prices.py` | The frozen price sheet |
| `lcvr/cost.py` | `call_cost()`: dollar cost of one logged call |
| `tests/test_cost.py` | 7 pytest checks against hand-calculated costs, including both sides of the 100K boundary |
| `check_cost.py` | Prices every call in the log and totals it |
| `tutorial/` | Learning and demo scripts (`demo_*.py` make real API calls) |

### Next

Step 4: download MuSiQue-Ans and 2WikiMultihopQA, inspect real questions, and build the shared corpora at 16K / 64K / 256K / 512K tokens.