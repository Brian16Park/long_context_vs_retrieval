# Long Context vs. Retrieval, at Matched Cost

Semester project for the Big Data Systems course (Stony Brook University).
Team: Brian Park, Vishnu Malladi, Shirisha Gajendra.

We compare two ways of answering questions about a document collection with an LLM:

- **Long context:** put the whole corpus in the prompt, using prompt caching.
- **Retrieval (RAG):** search the corpus and send only the most relevant chunks.

Both use the same model (Claude Haiku 5.5). We compare them at **equal dollar cost per question**, across corpus size and query volume, on multi-hop questions from MuSiQue and 2WikiMultihopQA.

**Status:** Weeks 1-2 done (model, logging, price sheet, cost model, dataset choice). Next: building the corpora and the two pipelines. Decisions and findings so far are in [`lab_notebook.md`](lab_notebook.md).

## Setup

You need Python 3.11 or newer, Git, and an Anthropic API key with credits ([Claude Console](https://platform.claude.com)). API usage is billed separately from a Claude subscription.

### 1. Clone and create a virtual environment

Windows (PowerShell):

```powershell
git clone https://github.com/Brian16Park/<long_context_vs_retrieval>.git
cd <repo>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

macOS / Linux:

```bash
git clone https://github.com/Brian16Park/<long_context_vs_retrieval>.git
cd <repo>
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

Datasets are not stored in git. Download them from the official sources and keep only these two files:

| Save as | Where to get it |
|---|---|
| `data/raw/musique_ans_v1.0_dev.jsonl` | The Google Drive zip linked from [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique). Inside the zip, copy `data/musique_ans_v1.0_dev.jsonl`. |
| `data/raw/2wiki_dev.json` | `data_ids_april7.zip` linked from [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop). Inside the zip, copy `dev.json` and rename it. |

Delete the zips afterwards; we only use the dev splits.

### 4. Check that everything works

Run these from the repo folder:

```bash
pytest                          # cost-model tests; free, no API calls. Expect "7 passed"
python dataset_stats.py         # dataset counts; should match the table in notes/lab_notebook.md
python -m tutorial.demo_llm     # one real API call pair; costs under $0.001
```

## Repo layout

```
lcvr/                 project code
  llm.py              call_claude(): every model call goes through here (caching, retries, logging)
  prices.py           frozen price sheet (date and source inside)
  cost.py             call_cost(): dollar cost of one logged call
tests/                pytest checks (no API calls)
tutorial/             learning and demo scripts; demo_*.py make real API calls
notes/lab_notebook.md decisions, findings, data sources
check_cost.py         prices every call in logs/calls.jsonl and totals it
dataset_stats.py      counts questions, hops and text in each dataset
data/                 datasets (not in git)
logs/                 call logs (not in git)
```

## How the code works, in brief

- **Every model call goes through `call_claude()`** in `lcvr/llm.py`. It takes required `labels` (run, system, corpus, question ID, ...) and appends one JSON line per attempt to `logs/calls.jsonl` with the billed token counts, the answer and the latency.
- **We log tokens, not dollars.** `lcvr/cost.py` turns token counts into dollars using the frozen price sheet in `lcvr/prices.py`, so runs can be re-priced without re-running them.
- **Runs are resumable.** `load_done_keys()` returns the label combinations that already succeeded; scripts skip those, so a crashed run can simply be started again.
- **Thinking is disabled** for all calls, so both systems run under identical settings.

## Data licenses

- MuSiQue: CC BY 4.0.
- 2WikiMultihopQA: see the [2wikimultihop repository](https://github.com/Alab-NII/2wikimultihop) (Apache-2.0).