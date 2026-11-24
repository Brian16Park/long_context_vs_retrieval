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

## Answer-format test (planned 2026-11-22)

The pilot's low accuracy (finding 5 above) may come from answering multi-hop questions in one step. Before the core sweep we test a second prompt style on a small slice, and decide with a rule set before seeing the results.

### Decisions

- **Two prompt styles** (`lcvr/prompts.py`). `answer_only` is the pilot's. `reasoning` changes only the middle of the instructions: "First, in at most three short sentences, reason step by step and name the facts you use. Then, on a last line of its own, write 'Answer:' followed by only the answer..." Reply cap: 200 tokens (50 for answer-only). A run uses one style for every system, so the systems still differ only in their documents. The reasoning tokens are billed output and count in the matched cost.
- **Reading the answer:** `clean_prediction()` now takes the text after the last "Answer:" label in the reply (the next line if the label stands alone), and the first line if there is no label. Answer-only replies score as before, unless a later line carries a label, which the prompt makes unlikely.
- **Labels:** reasoning calls get `"prompt": "reasoning"`; answer-only calls get no new label, so the pilot's calls keep their keys.
- **The test:** the pilot's 20 questions at 16K on both datasets, in the reasoning style: long context, RAG top 5 and top 20 without the reranker, and closed book. `scripts/compare_runs.py` compares it with the pilot call by call. Cost: about $0.06.
- **Decision rule.** Switch the core sweep to `reasoning` if F1 over all long-context and RAG calls together rises by at least 0.05, and at most 5% of reasoning replies lack an "Answer:" line. Otherwise keep `answer_only`. If the format check fails, fix the cap or instructions first and rerun the test under a new run name. The bar is low on purpose: adopting reasoning needlessly costs about $5 over the whole sweep, while staying at the floor would blur every comparison.
- **Caveat:** the 20 test questions are among the 300 evaluation questions per dataset. The choice is one yes/no, applied to every system alike, so the risk of fitting the test set is small.

### Results (2026-11-23)

The same calls, answer-only (the pilot) -> reasoning. 20 questions per row, one run each; sampling makes single runs noisy, so a difference under about 0.1 in one row means little.

| F1 (EM) | MuSiQue | 2Wiki |
|---|---|---|
| Long context, 16K | 0.24 -> 0.53 (0.05 -> 0.40) | 0.40 -> 0.66 (0.30 -> 0.50) |
| RAG, top 5 | 0.20 -> 0.40 (0.05 -> 0.30) | 0.66 -> 0.62 (0.65 -> 0.60) |
| RAG, top 20 | 0.18 -> 0.53 (0.05 -> 0.40) | 0.57 -> 0.61 (0.45 -> 0.50) |
| Closed book | 0.03 -> 0.04 (0.00 -> 0.00) | 0.16 -> 0.29 (0.15 -> 0.20) |

(RAG rows: without the reranker.) **Decision: the full run uses `reasoning`.** Over all 120 long-context and RAG calls, F1 rose from 0.38 to 0.56 (EM 0.26 to 0.45), well over the 0.05 bar, and 4 of 160 reasoning replies (2.5%) had no "Answer:" line, under the 5% limit. The new way of reading answers left the pilot's scores unchanged.

**1. The pilot's answer-only replies often weren't answer-only.** 7-10 of 20 MuSiQue replies per system hit the 50-token cap (2Wiki: 1-6 of 20), and replies averaged 12-32 tokens rather than a few words. So the pilot often scored the first line of an unfinished reply. Haiku 5.5 tends to explain a hard multi-hop answer even when told not to; the reasoning format lets it, then makes it end with a labeled answer. (This corrects the pilot's finding 5, which judged from two smoke-test replies.)

**2. The answer format changes the comparison.** With answer-only, RAG top 5 beat long context on 2Wiki at 16K (F1 0.66 vs 0.40). With reasoning, long context ties or beats RAG on both datasets (MuSiQue 0.53 vs 0.40 and 0.53; 2Wiki 0.66 vs 0.62 and 0.61). Twenty questions make this a hint, not a result, but it is a point for the report: how the model is asked to answer can change which system looks better.

**3. With room to reason, more retrieved text helps.** On MuSiQue, RAG top 20 (gold recall 0.97) beat top 5 (recall 0.66): F1 0.53 vs 0.40. With answer-only the two were level (0.18 vs 0.20).

**4. Reasoning cost 12% more here** ($0.0503 -> $0.0565 for the long-context and RAG calls), all of it output: replies averaged 97-146 tokens instead of 12-32. The extra output is about the same for every call, so it weighs more on cheap calls: RAG top 5 cost 26-41% more, long context 7-10%. For the full run that is about $5.

**5. The 200-token cap cut off 8 of 160 reasoning replies (5%)**, mostly MuSiQue (long context and closed book, 3 of 20 each). **Decision, made after seeing the test:** raise the cap to 400 for the full run. The rule didn't require it (the format check passed). The reason: if the model writes longer replies in bigger corpora, a cap that binds would cut off more long-context answers at large sizes and bias the size comparison. Billing counts only the tokens generated, so a higher cap costs only what longer replies actually use.

### Code added

| File | Purpose |
|---|---|
| `lcvr/prompts.py` | The `reasoning` style; `prompt_settings()`, `style_label()` |
| `lcvr/scoring.py` | `clean_prediction()` reads the last "Answer:" label; `has_answer_label()` |
| `lcvr/lc.py`, `lcvr/rag.py` | A `style` argument for the prompt and the labels |
| `lcvr/cost.py` | `lc_estimate()` can count the reply tokens |
| `scripts/run_*.py` | `--prompt` on every run script; `run_rag` also `--k` and `--reranker` |
| `scripts/compare_runs.py` | Two runs side by side, on the calls they share |
| `tests/` | 10 new checks (57 in total) |

## Full-run preparation (2026-11-23)

### Decisions

- **Long context runs several corpora at once** (`--workers`, default 4). Each corpus's questions still go one after another, so each corpus is written to the cache once and read after that. Sizes run smallest first, so usage grows gradually.
- **Rate limits** (checked 2026-11-23 for Haiku 5.5 on the Start tier): 2M input tokens per minute, which is what limits RAG (cache reads don't count toward it; cache writes do); 400K output tokens per minute and thousands of requests per minute, far above our use. New since October: *ramp limits*, which return 429 errors when usage rises sharply, even below those limits. New organizations can also start on a lower Evaluation tier.
- **Retries:** wait at least as long as the API's retry-after header asks; never retry the monthly spend cap (nothing helps until the next month); 12 attempts (about 7 minutes) instead of 8, so an unattended run rides out rate limits instead of stopping. Ctrl+C stops a parallel long-context run within one call.
- **Our tier:** the organization turned out to be on the Scale tier (10M input tokens per minute for Haiku 5.5), so RAG ran 16 calls at a time instead of 8.
- **The full run** (run name `full`): reasoning prompt with the 400-token cap; long context on every corpus (7,200 calls), RAG in all 10 settings (72,000 calls, 8 at a time), closed book (1,800 calls). Estimated cost about $135: long context about $82, RAG about $52, closed book under $1. Estimated time 7-8 hours, mostly RAG: its roughly 500M input tokens can go through at most 2M per minute, so it needs at least 4 hours.

### Code added

| File | Purpose |
|---|---|
| `scripts/run_lc.py` | `--workers`: several corpora at once; every size and question by default |
| `lcvr/lc.py` | `ask_corpus()` takes a `stop` signal |
| `lcvr/llm.py` | Retries follow retry-after, skip the spend cap, try 12 times; `retry_after()`, `error_code()` |
| `lcvr/prompts.py` | Reasoning reply cap: 400 tokens |
| `tests/` | 5 new checks, including retries with a fake API (62 in total) |

## Full run and analysis (2026-11-24)

### The run

- **81,000 calls answered:** long context 7,200, RAG 72,000, closed book 1,800. Billed $133.95 (long context $81.83, RAG $51.96, closed book $0.15) against an estimate of about $135. 3 failed attempts (2 connection errors, one 500), all retried.
- **Running time:** long context about 2.6 hours (4 corpora at a time), RAG 68 minutes (16 calls at a time), closed book 30 minutes. The run stopped twice (once when the PC crashed) and was resumed each time; no answered call was asked again.
- **Caching worked as the cost model assumes:** 465 of 470 corpora were written to the cache once and only read after that. The other 5 were in progress when the run stopped and were written a second time (+$0.32).
- **The 400-token cap hardly mattered:** long-context replies cut off by it rose with corpus size, as feared, but stayed under 1% (2, 3, 8 and 14 of 1,800 from 16K to 512K); RAG 4 of 72,000; closed book 5 of 1,800. Long-context replies also grew longer with corpus size (MuSiQue: 142 to 166 tokens on average).

### Decisions

- **Analysis code:** `lcvr/analysis.py` (computations, 14 tests) and `scripts/analyze.py`, which writes `results/`: `summary.md` with every table, CSV files, and 5 figures (`lcvr/figures.py`). Everything comes from the log and `data/processed/`. An independent check (a scorer and cost arithmetic written separately) reproduced every accuracy and cost figure exactly.
- **Units and intervals:** a question in one seed's corpora is one instance; each (dataset, size) has 900 (300 per seed x 3 seeds), answered by every system. 95% confidence intervals: bootstrap over instances, 2,000 resamples, within each seed. System comparisons are paired, on the same instances.
- **Long context's cost at query volume Q** comes from the measured tokens (mean cached prefix of instructions + corpus, the question's uncached tokens, the reply). Q = 1 is priced without caching: a cache write costs 25% more than plain input, so a one-off question wouldn't use it.
- **RAG's cost** is the API call as billed plus GPU time at $0.526 per hour (AWS g4dn.xlarge, one NVIDIA T4, on demand, us-east-1; checked 2026-11-24, in `lcvr/prices.py`): per question, embedding it, searching, and reranking the top 100 (if on); per corpus, embedding its paragraphs, spread over Q questions. The timings come from the RTX 3060 Ti, likely faster than a T4, so the GPU cost is a lower bound. It adds at most about 15% to a question's cost, except at Q = 1 in big corpora, where embedding the corpus is most of a cheap RAG setting's cost.
- **Matching rule, as proposed:** in each (dataset, size, Q) cell, long context against the most accurate RAG setting that costs no more per question. A winner needs a paired 95% interval that excludes zero; otherwise it's a tie. Iso-accuracy: the cheapest RAG setting at least as accurate as long context. The best RAG setting is picked on the questions it is scored on, which slightly favors RAG; it is the largest affordable top-k in every cell, so the choice is predictable and the bias small.
- **RAG error classes:** correct means exact match. A wrong answer is a reasoning failure (every gold paragraph was among the k sent), partial retrieval (some were), or a retrieval miss (none were).

### Findings

**1. Long context loses accuracy as the corpus grows; RAG loses less.** F1 from 16K to 512K tokens: long context 0.626 -> 0.461 (MuSiQue) and 0.702 -> 0.582 (2Wiki); RAG with the top 80 reranked 0.642 -> 0.569 and 0.717 -> 0.603. At 16K the two are level.

**2. RAG gets better with every step up in k,** to 80 paragraphs (about 16-20K tokens), at every corpus size; the reranker helps more as corpora grow (MuSiQue 512K, top 20: 0.478 -> 0.532).

**3. At matched cost, the winner depends on corpus size and reuse.**

| Corpus | MuSiQue | 2Wiki |
|---|---|---|
| 16K | tie at Q = 1; LC from Q = 10 (+0.06 to +0.16 F1) | RAG at Q = 1 (+0.02); LC from Q = 10 (+0.06 to +0.16) |
| 64K | RAG at Q = 1-10 (+0.05 to +0.06); tie from Q = 100 | tie at Q = 1-10; LC from Q = 100 (+0.03) |
| 256K | RAG at every Q (+0.08) | tie at every Q |
| 512K | RAG at every Q (+0.11) | tie at every Q |

Long context wins where it is cheap: a small corpus read by many questions costs it $0.0002-0.0004 per question, so RAG has to compete with its top 5 or 10. From 256K up, the cheapest RAG setting that is as accurate as long context costs 0.3-8% as much per question.

**4. Caching cuts long context's cost about 10x, which is not enough above the 100K price cliff.** At 512K a question costs $0.257 with no reuse and $0.026 at Q = 1,000 (MuSiQue); RAG with the top 80 reranked costs $0.0017-0.0027. At 16K with reuse, long context costs about as much as RAG with the top 5.

**5. Small k fails at retrieval, large k at reasoning.** With the top 5, 80-83% (MuSiQue) and 63-79% (2Wiki) of wrong answers had some but not all of their gold paragraphs; missing all of them is rare (at most 3.3% of questions). With the top 80, most wrong answers had every gold paragraph in hand: 99% at 16K and 60% at 512K (MuSiQue). Partial retrieval grows with corpus size at every k.

**6. Retrieval misses the later hops.** MuSiQue, top 20: the first hop's gold paragraph is found 93-99% of the time; the second hop's 61-89% and the fourth's 39-85%, falling as the corpus grows. Later hops are about entities the question never names. The reranker helps hops 3 and 4 in big corpora.

**7. Many right answers come without all the gold paragraphs:** 17-19% of MuSiQue questions with the top 5. The model fills gaps from memory (closed book gets 14% of MuSiQue and 16% of 2Wiki exactly right) or doesn't need every hop.

**8. The systems fail on different questions.** At 512K, 7% of MuSiQue questions are right only with long context and 17% only with RAG (2Wiki: 11% and 14%). A router that picked the right system for each question would beat both; that's an upper bound, not something we built.

**9. Position (secondary): weak signs of lost-in-the-middle at 16K on MuSiQue.** F1 is 0.76 when every gold paragraph sits near an edge of the corpus and 0.57 when one sits in the middle; no consistent pattern at larger sizes or on 2Wiki. The near-edge bins are small (18-40 questions).

**10. Closed book is far above the pilot's near-zero:** F1 0.226 (MuSiQue) and 0.268 (2Wiki). With room to reason, the model recalls more; the pilot measured 20 questions, answer-only.

**11. Latency:** median 1.0-1.5 s per long-context call, rising slightly with corpus size thanks to caching; RAG 0.7-0.9 s, plus up to 0.2 s of reranking.

### Caveats for the report

- One model (Haiku 5.5) and one prompt style; one run per setting at the default temperature.
- The bootstrap resamples questions, not corpora; questions in one corpus share context, so the intervals are a little narrow, most of all at 512K (2-3 corpora per seed).
- Accuracy is measured once per setting; Q enters only through cost, since caching doesn't change the answers.
- GPU time is priced from RTX 3060 Ti timings at a T4 rate.

## Status

The experiment is complete: data, corpora, both systems, the closed-book baseline, the full run and its analysis (`results/`).

### Next

The final report (Introduction, Methods, Results, Discussion and Conclusion) and the 5-15 minute Zoom presentation.
