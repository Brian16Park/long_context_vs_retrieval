# Lab notebook: Long Context vs. Retrieval, at Matched Cost

## Part 1 complete (2026-10-21): logging and cost model

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
| `scripts/check_cost.py` | Prices every call in the log and totals it |
| `tutorial/` | Learning and demo scripts (`demo_*.py` make real API calls) |

## Datasets chosen (2026-10-22): Weeks 1-2 complete

### Decision

- **Primary: MuSiQue-Ans, dev split**, all 2,417 questions eligible (all answerable).
- **Secondary: 2WikiMultihopQA, dev split, only `compositional` and `inference` questions** (5,236 + 1,549 = 6,785 eligible).
  - Excluded `comparison` and `bridge_comparison`: these name both candidate answers in the question ("X or Y?"), so guessing gets about 50% without reading anything. That would inflate both systems' accuracy and blur the difference between them.
  - The two kept types are fact chains, like MuSiQue, so both datasets measure the same skill.
- Train splits are not used (we don't train models).

### What the data looks like

| | MuSiQue-Ans dev | 2Wiki dev |
|---|---|---|
| Questions | 2,417 | 12,576 (6,785 kept) |
| Paragraphs per question | 17-20 | 10 |
| Gold paragraphs per question | = number of hops | 2 (4 for bridge_comparison, excluded) |
| Hops | 2: 1,252 / 3: 760 / 4: 405 | n/a |
| Avg words per paragraph | 83 (max 299) | 60 (max 1,687) |
| Unique paragraphs | 21,100 of 48,315 (56% repeats) | 54,957 of 125,760 |
| Words in unique paragraphs | 1.65M (about 2.1M tokens, rough) | 3.95M |
| Gold titles missing from context | n/a | 0 |

- **Formats differ.** MuSiQue: JSON Lines; each paragraph has `paragraph_text` and an `is_supporting` flag; `question_decomposition` lists the hops, with `paragraph_support_idx` naming the gold paragraph for each hop (enables per-hop retrieval-miss analysis). 2Wiki: one JSON list; each paragraph is `[title, [sentences]]`; gold paragraphs are found by matching titles in `supporting_facts`.
- **Enough text:** MuSiQue alone has about 4x the unique text a 512K-token corpus needs.
- **Duplicates are common**, because MuSiQue chains share Wikipedia paragraphs and distractors. Corpus building must remove them.
- **One outlier paragraph** in 2Wiki has 1,687 words; keep it in mind when chunking for retrieval.

### Data sources (data/ is not committed; download these)

| File to keep in `data/raw/` | Source | License |
|---|---|---|
| `musique_ans_v1.0_dev.jsonl` | Google Drive zip linked from https://github.com/StonyBrookNLP/musique (file `data/musique_ans_v1.0_dev.jsonl` inside) | CC BY 4.0 |
| `2wiki_dev.json` (renamed from `dev.json`) | `data_ids_april7.zip` linked from https://github.com/Alab-NII/2wikimultihop (version with fixed sentence segmentation) | Apache-2.0 (repo) |
| `2wiki_id_aliases.json` (renamed from `id_aliases.json`) | the same `data_ids_april7.zip` | Apache-2.0 (repo) |

## Step 4b complete (2026-11-22): common format and shared corpora

### Decisions

- **One common format** for both datasets, written to `data/processed/` by `scripts/convert_data.py`:
  - `<dataset>_questions.jsonl`: `qid`, `dataset`, `type`, `hops`, `question`, `answers` (gold answer first, then aliases), `gold_pids` (the gold paragraph of each hop, in hop order), `pids` (all of the question's paragraphs).
  - `<dataset>_paragraphs.jsonl`: `pid`, `title`, `text`, each paragraph stored once.
- **Paragraph IDs are hashes of title + text** (`mq_` or `2w_` + 12 hex digits of SHA-1). The same paragraph always gets the same ID, on any machine, so shared paragraphs are stored once and corpus files can list IDs instead of copying text.
- **MuSiQue gold paragraphs per hop** come from `question_decomposition[i].paragraph_support_idx`, cross-checked against the `is_supporting` flags (all 2,417 questions agree).
- **2Wiki gold paragraphs** come from the titles in `supporting_facts`. A question would be skipped only if a gold title pointed to two different paragraphs; none did.
- **2Wiki answer aliases:** aliases + demonyms of the answer's Wikidata ID (`answer_id`) from `id_aliases.json`, the same rule as the official 2Wiki evaluator.
- **One paragraph format in every prompt**, long-context and RAG alike: `# {title}` on one line, the text below, a blank line between paragraphs (`render_paragraph()` in `lcvr/corpus.py`).
- **Corpus sizes N = 16,000 / 64,000 / 256,000 / 512,000 tokens**, measured with the API's token-counting endpoint (free, separate rate limit) as one user message.
- **Corpus design**, per dataset and seed (3 seeds):
  1. Sample 300 questions; the same 300 are used at every size, so differences across sizes can't come from different questions. Measure each one's paragraph set exactly.
  2. At each size, split them into even groups (biggest first, each to the emptiest group with room) whose own paragraphs (gold + their own distractors) fill about 90%, at most 97%, of N.
  3. Each group becomes one corpus: its paragraphs plus filler from questions outside the sample, shuffled with the corpus ID as seed, so gold paragraphs land at random positions.
  4. Measure the corpus and adjust filler at the edge until it is within 1% of N.
- **Corpus files** (`data/processed/corpora/<dataset>_seed<s>.jsonl`, one corpus per line) store paragraph IDs in prompt order, the questions it serves, and its measured size. The text is rebuilt with `render_corpus()`.

### Findings

**1. Conversion reproduced the October counts exactly.**

| | MuSiQue-Ans dev | 2Wiki dev (kept types) |
|---|---|---|
| Questions | 2,417 | 6,785 |
| Unique paragraphs | 21,100 | 24,434 |
| Hops | 2: 1,252 / 3: 760 / 4: 405 | 2: 6,785 |
| Questions with answer aliases | 680 | 4,422 |
| Skipped | 0 | 0 |

120 2Wiki questions list a title twice in their paragraphs, all harmless: either the same paragraph twice, or two different distractors with one title.

**2. Haiku 5.5 uses many more tokens per character than the usual rule of thumb.** Measured on 400 random paragraphs per dataset: 2.78 characters per token (MuSiQue) and 2.56 (2Wiki), versus the ~4 often assumed. Paragraphs average 175 tokens (MuSiQue) and 234 (unique 2Wiki paragraphs); one question's full paragraph set is about 3,773 tokens (MuSiQue) and 1,641 (2Wiki). So a 16K corpus holds only about 4 MuSiQue questions' paragraph sets.

**3. An average ratio can't size small groups.** Text with many numbers or non-English names takes more tokens than the average predicts. The first build packed a 16K group by estimate, and it measured 16,325 tokens with no filler at all; the size check caught it. The fix was to measure instead of estimating where it matters: each sampled question's paragraphs once before grouping, and each corpus's question paragraphs once before adjusting filler. (Finding 4 of Part 1 again: measure tokens, never estimate them.)

**4. The corpora** (ranges over the 3 seeds; 470 corpora in total). Every corpus measured within 1% of its size.

| Size | MuSiQue: corpora, questions each | 2Wiki: corpora, questions each | Filler share of paragraphs |
|---|---|---|---|
| 16K | 79-82, 3-5 | 34, 7-12 | 12-18% |
| 64K | 20-21, 14-16 | 9, 31-37 | 15-26% |
| 256K | 5-6, 49-62 | 3, 96-107 | 21-48% |
| 512K | 3, 96-107 | 2, 142-158 | 34-63% |

**5. Cost estimate for the runs ahead** (frozen prices, still current on 2026-11-22): about $80 for long context on every corpus (512K about $51, 256K about $27), about $44 for RAG if it has 10 configurations (top-k 5/10/20/40/80, reranker on/off), and about $2 for the pilot and closed-book baseline: about $126 in total. The pilot checks this against the real bill first. Sending RAG calls through the Batch API (50% off) would save about $22.

### Code added

| File | Purpose |
|---|---|
| `lcvr/data.py` | Converts both datasets to the common format, with integrity checks |
| `lcvr/corpus.py` | Paragraph format, corpus building, and corpus checks |
| `lcvr/llm.py` | New `count_tokens()`: exact token count of a text (free; not logged) |
| `scripts/convert_data.py` | Writes `data/processed/` from `data/raw/` |
| `scripts/calibrate_tokens.py` | Measures characters per token for each dataset |
| `scripts/build_corpora.py` | Builds and checks all corpora (resumable: existing files are skipped) |
| `tests/test_data.py`, `tests/test_corpus.py` | 16 new checks with made-up data and a fake token counter (23 in total) |

## Steps 4c-6 and the pilot (2026-11-22): scoring, both systems, baseline

### Decisions

- **Scoring** (`lcvr/scoring.py`): exact match (EM) and token F1 with SQuAD normalization (lowercase; drop punctuation and a/an/the), best over all accepted answers: the same rules as the official MuSiQue and 2Wiki evaluators. `clean_prediction()` takes the reply's first line and strips an "Answer:" label.
- **One prompt for every system** (`lcvr/prompts.py`): instructions in the system prompt, then documents, then the question, and a 50-token answer cap. The instructions ask for the answer alone ("a short phrase such as a name, a date or a number, with no explanation") and a best guess when the documents don't settle it. The two systems differ only in which documents they see.
- **Long context** (`lcvr/lc.py`): the whole corpus as the document block, with a cache marker after it; a corpus's questions are asked back to back, so the first call writes the cache and the rest read it.
- **RAG** (`lcvr/rag.py`): chunks are paragraphs (the gold labels' unit, so recall@k is exact). Embeddings: `BAAI/bge-small-en-v1.5`, with BGE's search instruction in front of questions. Exact search in numpy. The top 100 candidates are reordered by the reranker `cross-encoder/ms-marco-MiniLM-L6-v2`. Every paragraph and question is embedded once and reused. 10 settings: top-k 5, 10, 20, 40, 80, each with and without the reranker; documents go in best first, uncached.
- **Closed-book baseline:** each question with no documents, to measure what the model already knows.
- **Local models run on the GPU** (RTX 3060 Ti, CUDA build of PyTorch 2.15 for Python 3.15). The timing records note the device, and the cost model will price GPU time at a cloud GPU rate.
- **All systems answer the same questions:** `pick_questions()` takes the first n at a size, corpus by corpus.
- **Runs show a cost estimate and wait for "y"**, resume after a crash, and stop at the first error that retries can't fix (e.g. no credit), cancelling queued calls.

### Findings

**1. The GPU is about 15x faster.** Embedding every paragraph and question: CPU 434 s (MuSiQue) and 469 s (2Wiki); GPU 29 s and 38 s. The vectors match, and so does the sanity check: 67% (MuSiQue) and 87% (2Wiki) of gold paragraphs rank in the top 5 of their question's own paragraphs, against 25% and 50% for random order.

**2. Retrieval gets steadily harder as the corpus grows.** Share of gold paragraphs in the top 20 (all 300 questions per size, mean of 3 seeds, dense -> reranked):

| Corpus | MuSiQue | 2Wiki |
|---|---|---|
| 16K | 0.94 -> 0.91 | 0.96 -> 0.96 |
| 64K | 0.87 -> 0.86 | 0.89 -> 0.92 |
| 256K | 0.79 -> 0.80 | 0.82 -> 0.84 |
| 512K | 0.74 -> 0.77 | 0.77 -> 0.80 |

At 512K even the top 80 miss 11-13% of gold paragraphs. Seeds agree within a few points.

**3. The reranker only pays off in big corpora.** It slightly lowers recall at 16K and raises it at 256K-512K. A likely reason: it was trained to rank by direct relevance to the question as written, and MuSiQue's later-hop paragraphs are about an entity the question never names. It helps once there are many distractors to filter.

**4. The cost model predicts the bill within about 1%.** Estimates (made before running, from measured corpus sizes and the frozen price sheet) against logged costs: long context +0.6% (both datasets), RAG +0.3% (MuSiQue) and -1.3% (2Wiki), against the proposal's 5% bar. At 256K, one cache write then 19 reads, as designed. The pilot cost $1.43 in total.

**5. Pilot accuracy is low, and more text makes it worse** (20 questions per size; the 16K and 256K questions differ, so compare systems within a size):

| EM / F1 | MuSiQue 16K | MuSiQue 256K | 2Wiki 16K | 2Wiki 256K |
|---|---|---|---|---|
| Closed book | 0.00 / 0.04 (all pilot questions) | | 0.08 / 0.13 (all pilot questions) | |
| Long context | 0.05 / 0.24 | 0.05 / 0.12 | 0.30 / 0.40 | 0.15 / 0.22 |
| RAG, top 5 | 0.05 / 0.20 | 0.15 / 0.22 | 0.65 / 0.66 | 0.30 / 0.43 |
| RAG, top 80 | 0.05 / 0.20 | 0.15 / 0.24 | 0.30 / 0.48 | 0.25 / 0.39 |

(RAG rows: without the reranker.) On 2Wiki, fewer paragraphs mean better answers: RAG with 5 beats RAG with 80, which roughly matches long context at 16K. MuSiQue sits near the floor for every system. The smoke test's replies were short phrases, as asked, so the likelier cause is answering in one step, with no room to work through 2-4 hops. **Open decision:** keep answer-only, or let both systems reason briefly before answering (the extra output tokens are billed and counted in the matched cost).

### Code added

| File | Purpose |
|---|---|
| `lcvr/scoring.py` | EM and F1, best over accepted answers |
| `lcvr/prompts.py` | The shared instructions and question format |
| `lcvr/lc.py` | Long context: ask a corpus's questions with the corpus cached |
| `lcvr/rag.py` | RAG: embeddings, search, reranking, recall, RAG answers |
| `lcvr/corpus.py` | New `pick_questions()`, shared by every system |
| `lcvr/llm.py` | New `read_log()`; log writes are safe when calls run in parallel |
| `lcvr/cost.py` | New `lc_estimate()` and `call_estimate()` for pre-run estimates |
| `scripts/embed.py`, `scripts/rank.py` | Embeddings once; dense and reranked top 100 per question and corpus |
| `scripts/run_lc.py`, `scripts/run_rag.py`, `scripts/run_closed_book.py` | Run each system (estimate, confirm, run, summary) |
| `scripts/show_answers.py` | A run's replies next to the gold answers |
| `tests/` | 24 new checks (47 in total), with fake models: no downloads or API calls |

## Status

Weeks 1-5 are complete: data, corpora, scoring, both systems, the closed-book baseline, all rankings, and the 20-question pilot. The pipelines score answers end to end, and the cost model matches the bill.

### Next

Decide the answer format (open decision in finding 5), then the core sweep: every corpus size, 300 questions per size, 3 seeds, both datasets.