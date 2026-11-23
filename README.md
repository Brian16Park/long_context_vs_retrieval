# Long Context vs. Retrieval, at Matched Cost

Semester project for the Big Data Systems course (Stony Brook University).
Team: Brian Park, Vishnu Malladi, Shirisha Gajendra.

We compare two ways of answering questions about a document collection with an LLM:

- **Long context:** put the whole corpus in the prompt, using prompt caching.
- **Retrieval (RAG):** search the corpus and send only the most relevant chunks.

Both use the same model (Claude Haiku 5.5). We compare them at **equal dollar cost per question**, across corpus size and query volume, on multi-hop questions from MuSiQue and 2WikiMultihopQA.

**Status:** Weeks 1-5 done: data, corpora, scoring, both systems, a closed-book baseline, and a 20-question pilot (the cost model matched the bill within about 1%). Next: the core sweep. Decisions and findings so far are in [`lab_notebook.md`](lab_notebook.md).

## Setup

You need Python 3.11 or newer, Git, and an Anthropic API key with credits ([Claude Console](https://platform.claude.com)). API usage is billed separately from a Claude subscription.

### 1. Clone and create a virtual environment

Windows (PowerShell):

```powershell
git clone https://github.com/Brian16Park/long_context_vs_retrieval.git
cd long_context_vs_retrieval
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

**Optional, NVIDIA GPU:** the embedding model and reranker run about 15x faster on a GPU. `requirements.txt` installs PyTorch's CPU build; to swap in the CUDA build (needs a driver supporting CUDA 13.2; check with `nvidia-smi`):

```powershell
pip uninstall -y torch
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.15.0+cu132 True
```

The code uses the GPU automatically when PyTorch can see one, and the CPU otherwise.

macOS / Linux:

```bash
git clone https://github.com/Brian16Park/long_context_vs_retrieval.git
cd long_context_vs_retrieval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API key

Create a file named `.env` in the repo folder containing one line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is in `.gitignore`. **Never commit it**: this repo is public. Run `git status` before committing and make sure `.env` is not listed.

### 3. Download the datasets

Datasets are not stored in git. Download them from the official sources and keep only these three files:

| Save as | Where to get it |
|---|---|
| `data/raw/musique_ans_v1.0_dev.jsonl` | The Google Drive zip linked from [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique). Inside the zip, copy `data/musique_ans_v1.0_dev.jsonl`. |
| `data/raw/2wiki_dev.json` | `data_ids_april7.zip` linked from [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop). Inside the zip, copy `dev.json` and rename it. |
| `data/raw/2wiki_id_aliases.json` | The same `data_ids_april7.zip`. Copy `id_aliases.json` and rename it. |

Delete the zips afterwards; we only use the dev splits.

### 4. Check that everything works

Run these from the repo folder:

```bash
pytest                          # free, no API calls. Expect "47 passed"
python -m scripts.dataset_stats  # dataset counts; should match the table in lab_notebook.md
python -m tutorial.demo_llm     # one real API call pair; costs under $0.001
```

### 5. Build the processed data, corpora, embeddings and rankings

All free (token counting and local models only). Each script skips work that's already done, so after a crash just run it again.

```powershell
python -m scripts.convert_data   # both datasets -> data/processed/ (seconds)
python -m scripts.build_corpora  # every corpus -> data/processed/corpora/ (minutes)
python -m scripts.embed          # every paragraph and question -> data/processed/embeddings/ (about 1 min on a GPU, 15 on a CPU)
foreach ($d in "musique", "2wiki") { foreach ($s in 0, 1, 2) { python -m scripts.rank --dataset $d --seed $s } }
```

`embed` and `rank` download two small models the first time (about 220 MB). `scripts/calibrate_tokens.py` measured the characters-per-token ratios the corpus builder uses; you don't need to rerun it.

## Running the systems

Each run script shows a cost estimate and waits for `y` before spending anything, logs every call to `logs/calls.jsonl`, and skips calls already answered when run again. The pilot, for example:

```powershell
python -m scripts.run_lc --run pilot --dataset musique --size 16000 256000 --questions 20
python -m scripts.run_rag --run pilot --dataset musique --size 16000 256000 --questions 20
python -m scripts.run_closed_book --run pilot --dataset musique --size 16000 256000 --questions 20
python -m scripts.show_answers pilot     # replies next to the gold answers
python -m scripts.check_cost             # total cost of everything logged
```

## Repo layout

```
lcvr/                 project code
  llm.py              call_claude(): every model call goes through here (caching, retries, logging);
                      count_tokens(): exact token count of a text (free)
  prices.py           frozen price sheet (date and source inside)
  cost.py             call_cost(): dollar cost of one logged call; estimates before a run
  data.py             converts both datasets to one common format
  corpus.py           how paragraphs look in a prompt; builds and checks the corpora; pick_questions()
  scoring.py          exact match and F1, as the official evaluators score them
  prompts.py          the instructions every system shares
  lc.py               long context: the whole corpus, cached, then each question
  rag.py              RAG: embeddings, search, reranking, then the top k paragraphs and the question
scripts/              things you run: python -m scripts.<name>
  convert_data.py     data/raw/ -> data/processed/
  calibrate_tokens.py measures characters per token for each dataset
  build_corpora.py    builds every corpus (2 datasets x 3 seeds x 4 sizes)
  embed.py            embeds every paragraph and question once
  rank.py             dense and reranked top 100 for every question in its corpus; prints recall@k
  run_lc.py           runs long context
  run_rag.py          runs RAG (10 settings: top-k 5-80, with and without the reranker)
  run_closed_book.py  runs the no-documents baseline
  show_answers.py     a run's replies next to the gold answers
  check_cost.py       prices every call in logs/calls.jsonl and totals it
  dataset_stats.py    counts questions, hops and text in each dataset
tests/                pytest checks (no API calls)
tutorial/             learning and demo scripts; demo_*.py make real API calls
lab_notebook.md       decisions, findings, data sources
data/                 raw and processed data, corpora (not in git)
logs/                 call logs (not in git)
```

## How the code works, in brief

- **Every model call goes through `call_claude()`** in `lcvr/llm.py`. It takes required `labels` (run, system, corpus, question ID, ...) and appends one JSON line per attempt to `logs/calls.jsonl` with the billed token counts, the answer and the latency.
- **We log tokens, not dollars.** `lcvr/cost.py` turns token counts into dollars using the frozen price sheet in `lcvr/prices.py`, so runs can be re-priced without re-running them.
- **Runs are resumable.** `load_done_keys()` returns the label combinations that already succeeded; scripts skip those, so a crashed run can simply be started again.
- **Thinking is disabled** for all calls, so both systems run under identical settings.
- **One data format for both datasets.** Every question lists its accepted answers and its gold paragraph for each hop; every paragraph has an ID made from its own text, so the same paragraph gets the same ID on any machine.
- **Corpora are measured, not estimated.** Each corpus is built for a target size (16K to 512K tokens) and checked with the exact token counter to be within 1% of it. Corpus files list paragraph IDs in prompt order, so the exact prompt text can always be rebuilt.
- **The systems differ only in their documents.** Every system uses the same instructions and layout (instructions, documents, question). Long context sends the whole corpus, cached; RAG sends the top k paragraphs of the question's ranking; the closed-book baseline sends none. All of them answer the same questions, picked by `pick_questions()`.

## Data licenses

- MuSiQue: CC BY 4.0.
- 2WikiMultihopQA: see the [2wikimultihop repository](https://github.com/Alab-NII/2wikimultihop) (Apache-2.0).