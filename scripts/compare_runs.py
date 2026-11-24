"""Compare two runs on the calls they have in common: the same system, dataset, corpus,
question and RAG setting. Shows what a change, such as the prompt style, did to accuracy
and cost, on exactly the same questions.

Run from the repo folder, for example:
  python -m scripts.compare_runs pilot reasoning-test

Reads the log only: it makes no API calls.
"""
import json
import sys
from statistics import mean

from lcvr.cost import call_cost
from lcvr.data import read_jsonl
from lcvr.llm import read_log
from lcvr.scoring import has_answer_label, score

if len(sys.argv) != 3:
    raise SystemExit("Usage: python -m scripts.compare_runs RUN_A RUN_B")
run_a, run_b = sys.argv[1], sys.argv[2]


def pair_key(record):
    """What two calls must share to be compared: their labels, minus the run name and prompt style."""
    labels = json.loads(record["key"])            # the labels the call was made with
    labels.pop("run")
    labels.pop("prompt", None)
    return json.dumps(labels, sort_keys=True)


calls = {run_a: {}, run_b: {}}
for r in read_log():
    if r.get("ok") and r.get("run") in calls:
        calls[r["run"]][pair_key(r)] = r          # a call made twice: the later one counts
shared = calls[run_a].keys() & calls[run_b].keys()
if not shared:
    raise SystemExit(f"Runs '{run_a}' and '{run_b}' have no calls in common.")

questions = {}
for dataset in {calls[run_a][key]["dataset"] for key in shared}:
    questions.update({q["qid"]: q for q in read_jsonl(f"data/processed/{dataset}_questions.jsonl")})

SYSTEM_ORDER = {"lc": 0, "rag": 1, "closed_book": 2}


def row_of(r):
    """Which table row a call belongs to: (dataset, system, corpus size, k, reranked)."""
    return (r["dataset"], r["system"], r.get("corpus_tokens", 0), r.get("k", 0), r.get("reranked", False))


def row_sort(row):
    dataset, system, size, k, reranked = row
    return dataset, SYSTEM_ORDER.get(system, 9), size, k, reranked


def summary(pairs):
    """Mean EM, mean F1, total cost, mean reply tokens and replies cut off, for A and for B."""
    out = {}
    for side in (0, 1):
        records = [pair[side] for pair in pairs]
        scores = [score(r["answer"], questions[r["question_id"]]["answers"]) for r in records]
        out[side] = (mean(s[0] for s in scores), mean(s[1] for s in scores),
                     sum(call_cost(r) for r in records), mean(r["output_tokens"] for r in records),
                     sum(r["stop_reason"] == "max_tokens" for r in records))
    return out


rows = {}
for key in shared:
    a, b = calls[run_a][key], calls[run_b][key]
    rows.setdefault(row_of(a), []).append((a, b))

print(f"A = run '{run_a}', B = run '{run_b}': {len(shared)} calls in both, compared one to one.\n")
print("dataset  system        size  setting       calls   EM A -> B     F1 A -> B     "
      "cost A -> B            reply tokens   cut off A/B")
for row in sorted(rows, key=row_sort):
    dataset, system, size, k, reranked = row
    s = summary(rows[row])
    size_text = f"{size // 1000}K" if size else "-"
    setting = f"k={k}{' reranked' if reranked else ''}" if system == "rag" else ""
    print(f"{dataset:<8} {system:<12} {size_text:>5}  {setting:<12} {len(rows[row]):>5}   "
          f"{s[0][0]:.2f} -> {s[1][0]:.2f}  {s[0][1]:.2f} -> {s[1][1]:.2f}  "
          f"${s[0][2]:.4f} -> ${s[1][2]:.4f}  {s[0][3]:>5.0f} -> {s[1][3]:<5.0f}  "
          f"{s[0][4]:>4}/{s[1][4]}")

# Everything with documents (long context and RAG) pooled: one number per side
with_documents = [pair for row, pairs in rows.items() if row[1] != "closed_book" for pair in pairs]
if with_documents:
    s = summary(with_documents)
    print(f"\nLong context and RAG together: {len(with_documents)} calls, "
          f"EM {s[0][0]:.2f} -> {s[1][0]:.2f}, F1 {s[0][1]:.2f} -> {s[1][1]:.2f}, "
          f"cost ${s[0][2]:.4f} -> ${s[1][2]:.4f}")

# Reasoning replies should end with an "Answer:" line; count the ones that don't
for run in (run_a, run_b):
    reasoning = [calls[run][key] for key in shared if calls[run][key].get("prompt") == "reasoning"]
    if reasoning:
        missing = sum(not has_answer_label(r["answer"]) for r in reasoning)
        print(f"Run '{run}': {missing} of {len(reasoning)} replies have no 'Answer:' line "
              "(they are scored from their first line).")
