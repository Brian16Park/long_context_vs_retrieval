# Long Context vs. Retrieval, at Matched Cost

Final project for Big Data Systems (Stony Brook University), Group 9: Brian Park, Vishnu Malladi, Shirisha Gajendra.

We compare two ways of answering questions about a document collection with an LLM, at **equal dollar cost per question**:

- **Long context (LC):** the whole corpus in the prompt, with prompt caching.
- **Retrieval (RAG):** dense search over the corpus (optionally reranked) and only the top k paragraphs in the prompt.

Both use the same model (Claude Haiku 5.5), instructions and questions: multi-hop questions from MuSiQue and 2WikiMultihopQA, in shared corpora of 16K to 512K tokens, priced at 1 to 1,000 questions per cached corpus.

**Main results** (81,000 API calls, $133.95; details in [`results/summary.md`](results/summary.md)):

- Long context lost more accuracy as corpora grew (MuSiQue F1 0.63 to 0.46 from 16K to 512K tokens) than RAG with the top 80 reranked paragraphs (0.64 to 0.57).
- At matched cost, long context won 8 of 32 cells, all small corpora reused by many questions; RAG won 11 and 13 were ties. From 256K tokens up, RAG won or tied, and matched long context's accuracy at 0.3-8% of its cost.
- RAG's errors were mostly partial multi-hop retrieval with few paragraphs and reasoning failures with many.

**Status:** complete: data, corpora, both systems, a closed-book baseline, the full run and its analysis. Not finished (stated in the report): re-pricing the log under other providers' caching schemes; the answer-position study is partial (long context only); the distractor-similarity sweep was dropped.

**Contents:** [1. Repository and frozen version](#1-repository-and-frozen-version) · [2. Architecture and execution path](#2-architecture-and-execution-path) · [3. Environment and dependencies](#3-environment-and-dependencies) · [4. Reproduction procedure](#4-reproduction-procedure) · [5. Verification and ownership](#5-verification-and-ownership) · [Repo layout](#repo-layout) · [Data licenses](#data-licenses)

## 1. Repository and frozen version

- **Repository:** https://github.com/Brian16Park/long_context_vs_retrieval
- **Frozen version:** tag [`v1.0`](https://github.com/Brian16Park/long_context_vs_retrieval/releases/tag/v1.0), the code, results and README that the final report describes. Later commits, if any, do not change the reported results.
- **Release assets** on the `v1.0` release page, for rebuilding the results without new API calls:
  - `log.zip`: `logs/calls.jsonl`, one line per API attempt of every run (pilot, answer-format test, full run), with billed tokens, answers and latency.
  - `data.zip`: `data/processed/` without the embedding vectors (`*.npz`): questions and paragraphs in the common format, the 470 corpora, the dense and reranked rankings, and the GPU timings.

To get exactly the reported version:

```bash
git clone https://github.com/Brian16Park/long_context_vs_retrieval.git
cd long_context_vs_retrieval
git checkout v1.0
```

Not in git: the raw datasets (download them, Section 4), `data/processed/` and `logs/` (release assets, or rebuild them), and `.env` (your API key; never commit it).

## 2. Architecture and execution path

The system is a pipeline of files in three stages. Data is prepared once; three systems ask one model through a single logged wrapper; the analysis reads only the log and makes no API calls.

```
1. Prepare, once (free)
   data/raw/ --convert_data--> data/processed/<dataset>_questions.jsonl, _paragraphs.jsonl
             --build_corpora-> data/processed/corpora/<dataset>_seed<s>.jsonl   470 corpora, 16K-512K tokens
             --embed, rank---> data/processed/embeddings/, rankings/           dense + reranked top 100 (GPU)

2. Ask the model (paid)
   run_lc           (whole corpus, cached)  \
   run_rag          (top k, 10 settings)     >--> call_claude() --> Claude Haiku 5.5 API
   run_closed_book  (no documents)          /          |
                                                       +--> logs/calls.jsonl   one JSON line per attempt

3. Analyze (free)
   analyze --> results/summary.md, results/*.csv, results/figures/*.png
```

| Stage | Command (`python -m ...`) | Reads | Writes | Cost, time |
|---|---|---|---|---|
| Convert | `scripts.convert_data` | `data/raw/` | `data/processed/*_questions.jsonl`, `*_paragraphs.jsonl` | free, seconds |
| Corpora | `scripts.build_corpora` | processed data | `data/processed/corpora/` | free (token counting), minutes |
| Embed | `scripts.embed` | paragraphs, questions | `data/processed/embeddings/` | free, about 1 min on a GPU |
| Rank | `scripts.rank --dataset D --seed S` | embeddings, corpora | `data/processed/rankings/` | free, minutes on a GPU |
| Long context | `scripts.run_lc` | corpora | `logs/calls.jsonl` | $81.83 in the full run |
| RAG | `scripts.run_rag` | rankings | `logs/calls.jsonl` | $51.96 in the full run |
| Closed book | `scripts.run_closed_book` | questions | `logs/calls.jsonl` | $0.15 in the full run |
| Analyze | `scripts.analyze --run full` | log, processed data | `results/` | free, about 15 s |

How the code works:

- **Every model call goes through `call_claude()`** in `lcvr/llm.py`. It builds the prompt (instructions, documents with an optional cache marker, then the question), refuses to run without labels (run, system, dataset, seed, corpus, question, RAG setting), retries rate limits and server errors with exponential backoff, stops at once on errors a retry cannot fix (no credit, a spending cap), and appends one JSON line per attempt to `logs/calls.jsonl`.
- **We log tokens, not dollars.** `lcvr/cost.py` prices token counts with the frozen sheet in `lcvr/prices.py`, so the log can be re-priced without new calls. `lcvr/analysis.py` prices long context at Q = 1, 10, 100 and 1,000 questions per cached corpus (`lc_cost()`), and RAG with its billed tokens plus GPU time at a cloud rate (`rag_cost()`).
- **Runs are resumable.** `load_done_keys()` returns the label sets already answered; every run script skips them, so a stopped run is simply started again.
- **Corpora are measured, not estimated.** Each corpus is built for a target size and checked with the API's token counter to be within 1% of it. Corpus files list paragraph IDs in prompt order, so the exact prompt can always be rebuilt.
- **The systems differ only in their documents.** Same instructions and layout (instructions, documents, question), same prompt style, same questions (`pick_questions()`). Extended thinking is off for every call.
- **Parallel where it is safe.** `run_lc` works on several corpora at once (`--workers`), but asks each corpus's questions in order, so each corpus is written to the cache once and read after that. RAG calls are independent.

## 3. Environment and dependencies

| | Used for the reported run |
|---|---|
| OS | Windows desktop (PowerShell); the tests also pass on Linux |
| Python | 3.15 (3.11 or newer works; tests also pass on 3.13) |
| GPU | NVIDIA RTX 3060 Ti, PyTorch CUDA build; optional, the CPU works about 15x slower for embedding and reranking |
| Model | `claude-haiku-5-5` through the Anthropic API, extended thinking off, default temperature |
| Packages | `requirements.txt`: anthropic 1.17.0, python-dotenv 1.2.4, sentence-transformers 6.3.0, torch 2.15.0, numpy 2, matplotlib 3.11.2, pytest 9.1.1 |
| Local models | `BAAI/bge-small-en-v1.5` (embeddings), `cross-encoder/ms-marco-MiniLM-L6-v2` (reranker), downloaded on first use (about 220 MB) |

You need Git, Python, and for new runs an Anthropic API key with credits ([Claude Console](https://platform.claude.com)). Analysis of the released log needs no key.

### 3.1 Clone and create a virtual environment

Windows (PowerShell):

```powershell
git clone https://github.com/Brian16Park/long_context_vs_retrieval.git
cd long_context_vs_retrieval
git checkout v1.0
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

macOS / Linux:

```bash
git clone https://github.com/Brian16Park/long_context_vs_retrieval.git
cd long_context_vs_retrieval
git checkout v1.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Optional, NVIDIA GPU:** `requirements.txt` installs PyTorch's CPU build. To swap in the CUDA build (needs a driver supporting CUDA 13.2; check with `nvidia-smi`):

```powershell
pip uninstall -y torch
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.15.0+cu132 True
```

The code uses the GPU automatically when PyTorch can see one, and the CPU otherwise.

### 3.2 Add your API key (only for new API calls)

Create a file named `.env` in the repo folder containing one line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is in `.gitignore`. **Never commit it**: this repo is public. Run `git status` before committing and make sure `.env` is not listed.

## 4. Reproduction procedure

### 4.1 Run the tests (free, seconds)

```bash
pytest          # 76 tests, no API calls. Expect "76 passed"
```

### 4.2 Rebuild every table and figure from the released log (free, about a minute)

1. Set up the environment (Section 3.1). No API key is needed.
2. Download `log.zip` and `data.zip` from the `v1.0` release and unzip them in the repo folder, so that `logs/calls.jsonl` and `data/processed/corpora/` exist.
3. Run the analysis and compare with the committed results:

```bash
python -m scripts.analyze --run full
git diff --stat -- results/   # no changes: the rebuilt tables match the committed ones
python -m scripts.compare_runs pilot reasoning-test   # the answer-format ablation
python -m scripts.check_cost                          # total cost of every logged call
```

The analysis is deterministic (the bootstrap uses a fixed seed), so the rebuilt tables match the committed ones exactly.

### 4.3 Re-run the experiment from the raw datasets (about $135 and 8-10 hours)

1. **Download the datasets** (not stored in git) and keep only these three files:

| Save as | Where to get it |
|---|---|
| `data/raw/musique_ans_v1.0_dev.jsonl` | The Google Drive zip linked from [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique). Inside the zip, copy `data/musique_ans_v1.0_dev.jsonl`. |
| `data/raw/2wiki_dev.json` | `data_ids_april7.zip` linked from [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop). Inside the zip, copy `dev.json` and rename it. |
| `data/raw/2wiki_id_aliases.json` | The same `data_ids_april7.zip`. Copy `id_aliases.json` and rename it. |

2. **Check the setup:** `python -m scripts.dataset_stats` (counts should match `lab_notebook.md`) and `python -m tutorial.demo_llm` (one real call pair, under $0.001).

3. **Build the processed data, corpora, embeddings and rankings** (free; each script skips finished work, so after a crash run it again). `rank` prints recall@k at the end, the source of the reranker ablation in the report.

```powershell
python -m scripts.convert_data
python -m scripts.build_corpora
python -m scripts.embed
foreach ($d in "musique", "2wiki") { foreach ($s in 0, 1, 2) { python -m scripts.rank --dataset $d --seed $s } }
```

4. **Pilot and answer-format test** (optional, about $1.50). Each run script shows a cost estimate and waits for `y`:

```powershell
foreach ($d in "musique", "2wiki") {
  python -m scripts.run_lc --run pilot --dataset $d --size 16000 256000 --questions 20
  python -m scripts.run_rag --run pilot --dataset $d --size 16000 256000 --questions 20
  python -m scripts.run_closed_book --run pilot --dataset $d --size 16000 256000 --questions 20
  python -m scripts.run_lc --run reasoning-test --prompt reasoning --dataset $d --size 16000 --questions 20
  python -m scripts.run_rag --run reasoning-test --prompt reasoning --dataset $d --size 16000 --questions 20 --k 5 20 --reranker no
  python -m scripts.run_closed_book --run reasoning-test --prompt reasoning --dataset $d --size 16000 --questions 20
}
python -m scripts.compare_runs pilot reasoning-test
```

5. **The full run:** every corpus, all 300 questions per size, 3 seeds, both datasets, in the reasoning style. The reported run used 4 long-context corpora at once and 16 RAG calls at once; use fewer `--workers` on a lower API rate-limit tier (rate limits are retried, not fatal). If a run stops (no credit, a network outage), run the same commands again: answered calls are skipped.

```powershell
foreach ($d in "musique", "2wiki") { foreach ($s in 0, 1, 2) {
  python -m scripts.run_lc --run full --prompt reasoning --dataset $d --seed $s --workers 4 --yes
} }
foreach ($d in "musique", "2wiki") { foreach ($s in 0, 1, 2) {
  python -m scripts.run_rag --run full --prompt reasoning --dataset $d --seed $s --workers 16 --yes
} }
foreach ($d in "musique", "2wiki") { foreach ($s in 0, 1, 2) {
  python -m scripts.run_closed_book --run full --prompt reasoning --dataset $d --seed $s --yes
} }
python -m scripts.analyze --run full
```

The model samples at its default temperature, so a new run gives slightly different answers: expect numbers within the reported 95% confidence intervals rather than identical ones.

### 4.4 Where each reported result comes from

| Report item | File or command |
|---|---|
| Table 1: accuracy by corpus size | `results/accuracy.csv` |
| Figure 2: accuracy against cost | `results/figures/accuracy_vs_cost.png`, drawn from `results/accuracy.csv` |
| Figure 3: the matched-cost map | `results/matched_cost.csv`, `results/figures/crossover_map.png` |
| Table 2: cost per question across Q | `results/summary.md` (Cost per question) |
| Table 3: RAG error classes | `results/rag_outcomes.csv` |
| Hop recall, overlap, gold position | `results/musique_hop_recall.csv`, `results/overlap.csv`, `results/lc_gold_position.csv` |
| Answer-format ablation | `python -m scripts.compare_runs pilot reasoning-test` |
| Reranker recall | printed by `scripts.rank` for each seed (the report gives the mean) |
| Every other number | `results/summary.md` |
| Figure 1 (architecture) | drawn by hand from this README's Section 2 |

## 5. Verification and ownership

How the results were checked:

- **Unit tests:** 76 tests with fake API clients, a fake token counter and hand-calculated costs, covering the price tiers on both sides of 100K tokens, corpus fitting (including two bugs found while building it), answer extraction, retries, parallel logging and the analysis.
- **Rehearsals:** every run script was run against a fake API through a crash and resume, an empty balance and rate limits before any paid run.
- **Cost model against the bill:** the pilot's pre-run estimates were within 1.3% of the billed cost, and the log shows 465 of 470 corpora written to the cache exactly once, as the model assumes.
- **Corpus checks:** `build_corpora` verifies every corpus is within 1% of its target size and that every question's gold paragraphs are in its corpus.
- **Independent re-scoring:** after the main run, a separately written scorer and cost calculation reproduced every accuracy and cost figure in `results/` exactly.
- **Determinism:** `analyze` uses a fixed bootstrap seed, so anyone with the released log rebuilds the committed tables exactly (Section 4.2).

Ownership:

| Team member | Contribution |
|---|---|
| Brian Park | Design, the Pilot and Full Runs |
| Vishnu Malladi | Data preparation, Report |
| Shirisha Gajendra | Testing, Analysis, Report |

AI assistance: Claude (Anthropic) was used as a coding and writing assistant. Every script was run and tested before use, and every reported number comes from the code and log in this repository.

## Repo layout

```
lcvr/                 project code
  llm.py              call_claude(): every model call (caching, labels, retries, logging);
                      count_tokens(): exact token count of a text (free)
  prices.py           frozen price sheet (date and source inside)
  cost.py             call_cost(): dollar cost of one logged call; estimates before a run
  data.py             converts both datasets to one common format
  corpus.py           paragraph format in prompts; builds and checks the corpora; pick_questions()
  scoring.py          exact match and F1, as the official evaluators score them
  prompts.py          the instructions every system shares, in two styles (answer only, brief reasoning)
  lc.py               long context: the whole corpus, cached, then each question
  rag.py              RAG: embeddings, search, reranking, then the top k paragraphs and the question
  analysis.py         accuracy with confidence intervals, cost across Q, the matched-cost comparison,
                      RAG errors split into retrieval misses and reasoning failures
  figures.py          the result figures (matplotlib)
scripts/              things you run: python -m scripts.<name>
  convert_data.py     data/raw/ -> data/processed/
  calibrate_tokens.py measures characters per token for each dataset (already done)
  build_corpora.py    builds every corpus (2 datasets x 3 seeds x 4 sizes)
  embed.py            embeds every paragraph and question once
  rank.py             dense and reranked top 100 for every question in its corpus; prints recall@k
  run_lc.py           runs long context, several corpora at once
  run_rag.py          runs RAG (10 settings: top-k 5-80, with and without the reranker)
  run_closed_book.py  runs the no-documents baseline
  show_answers.py     a run's replies next to the gold answers
  compare_runs.py     two runs side by side, on the calls they share (e.g. two prompt styles)
  check_cost.py       prices every call in logs/calls.jsonl and totals it
  analyze.py          every table and figure of a finished run -> results/
  dataset_stats.py    counts questions, hops and text in each dataset
tests/                pytest checks (no API calls)
results/              tables and figures of the full run (from scripts/analyze.py)
tutorial/             learning and demo scripts; demo_*.py make real API calls
lab_notebook.md       decisions, findings, data sources, in the order they happened
data/                 raw and processed data, corpora (not in git; see Section 1)
logs/                 call logs (not in git; see Section 1)
```

## Data licenses

- MuSiQue: CC BY 4.0.
- 2WikiMultihopQA: Apache-2.0 (see the [2wikimultihop repository](https://github.com/Alab-NII/2wikimultihop)).

The processed data and call log in the release are derived from these datasets under the same licenses.