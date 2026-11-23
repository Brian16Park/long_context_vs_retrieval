"""The closed-book baseline: each question with no documents at all, to measure how much the
model already knows from its training (Wikipedia is in it).

Run from the repo folder, for example:
  python -m scripts.run_closed_book --run pilot --dataset musique --size 16000 256000 --questions 20

Asks each question picked at those sizes once: with no documents, corpus size doesn't
matter. Shows a cost estimate and waits for "y"; running it again skips answered questions.
"""
import argparse
from statistics import mean

from lcvr.corpus import SIZES, pick_questions
from lcvr.cost import call_cost, call_estimate
from lcvr.data import read_jsonl
from lcvr.llm import call_claude, label_key, load_done_keys, read_log
from lcvr.prompts import CLOSED_BOOK_INSTRUCTIONS, MAX_ANSWER_TOKENS, question_text
from lcvr.scoring import score

parser = argparse.ArgumentParser(description="Run the closed-book baseline.")
parser.add_argument("--run", required=True, help="a name for this run, e.g. pilot")
parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, nargs="+", default=list(SIZES), help="corpus sizes (default: all)")
parser.add_argument("--questions", type=int, default=None, help="questions per size (default: all)")
parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
args = parser.parse_args()

questions = {q["qid"]: q for q in read_jsonl(f"data/processed/{args.dataset}_questions.jsonl")}
corpora = read_jsonl(f"data/processed/corpora/{args.dataset}_seed{args.seed}.jsonl")

# The questions the other systems answer at these sizes, each once
qids = list(dict.fromkeys(qid for size in args.size
                          for _, chosen in pick_questions(corpora, size, args.questions)
                          for qid in chosen))


def labels(qid):
    return {"run": args.run, "system": "closed_book", "dataset": args.dataset,
            "seed": args.seed, "question_id": qid}


done = load_done_keys()
todo = [qid for qid in qids if label_key(labels(qid)) not in done]
estimate = len(todo) * call_estimate(input_tokens=100, output_tokens=10)
print(f"{len(qids)} questions, {len(todo)} still to do. Estimated cost: ${estimate:.4f}")
if todo and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was spent.")

for qid in todo:
    call_claude(question_text(questions[qid]), labels(qid),
                system=CLOSED_BOOK_INSTRUCTIONS, max_tokens=MAX_ANSWER_TOKENS)

records = [r for r in read_log()
           if r.get("ok") and r.get("run") == args.run and r.get("system") == "closed_book"
           and r.get("dataset") == args.dataset and r.get("seed") == args.seed and r["question_id"] in qids]
scores = [score(r["answer"], questions[r["question_id"]]["answers"]) for r in records]
print(f"\nClosed book, {args.dataset} seed {args.seed}: {len(records)} answers, "
      f"EM {mean(s[0] for s in scores):.2f}, F1 {mean(s[1] for s in scores):.2f}, "
      f"cost ${sum(call_cost(r) for r in records):.4f}")