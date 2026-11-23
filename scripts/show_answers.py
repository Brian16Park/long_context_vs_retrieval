"""Show a run's replies next to the gold answers, with their scores.

Run from the repo folder:  python -m scripts.show_answers smoke
(The optional second argument limits how many to show, e.g.  ... pilot 30)
"""
import sys

from lcvr.data import read_jsonl
from lcvr.llm import read_log
from lcvr.scoring import clean_prediction, score

run = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
records = [r for r in read_log() if r.get("ok") and r.get("run") == run][:limit]
if not records:
    raise SystemExit(f"No successful calls logged for run '{run}'.")

questions = {}
for dataset in {r["dataset"] for r in records}:
    questions.update({q["qid"]: q for q in read_jsonl(f"data/processed/{dataset}_questions.jsonl")})

for r in records:
    q = questions[r["question_id"]]
    em, f1 = score(r["answer"], q["answers"])
    size = f", {r['corpus_tokens']:,} tokens" if "corpus_tokens" in r else ""
    print(f"[{r['system']}{size}] {q['question']}")
    print(f"   reply:  {r['answer']!r}")
    print(f"   scored: {clean_prediction(r['answer'])!r}")
    print(f"   gold:   {q['answers']}   EM {em:.0f}, F1 {f1:.2f}\n")