"""Run the long-context system: each corpus in the prompt (cached), then its questions one by one.

Run from the repo folder, for example:
  python -m scripts.run_lc --run smoke --dataset musique --size 16000 --questions 2
  python -m scripts.run_lc --run pilot --dataset musique --size 16000 256000 --questions 20

Asks up to --questions questions at each size, corpus by corpus (seed 0 unless --seed).
Shows a cost estimate and waits for "y" before spending anything. Every call is logged
to logs/calls.jsonl, so running the same command again skips questions already answered.
"""
import argparse
from statistics import mean

from lcvr.cost import call_cost, lc_estimate
from lcvr.data import read_jsonl
from lcvr.lc import ask_corpus
from lcvr.llm import load_done_keys, read_log
from lcvr.scoring import score

parser = argparse.ArgumentParser(description="Run the long-context system.")
parser.add_argument("--run", required=True, help="a name for this run, e.g. smoke or pilot")
parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, nargs="+", required=True, help="corpus sizes, e.g. 16000 256000")
parser.add_argument("--questions", type=int, required=True, help="questions to ask at each size")
parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
args = parser.parse_args()

questions_by_qid = {q["qid"]: q for q in read_jsonl(f"data/processed/{args.dataset}_questions.jsonl")}
paragraphs_by_pid = {p["pid"]: p for p in read_jsonl(f"data/processed/{args.dataset}_paragraphs.jsonl")}
corpora = read_jsonl(f"data/processed/corpora/{args.dataset}_seed{args.seed}.jsonl")

# ---------------- plan: which corpora, and how many of each one's questions ----------------
plan = []
for size in args.size:
    remaining = args.questions
    for corpus in corpora:
        if corpus["target_tokens"] == size and remaining > 0:
            n = min(remaining, len(corpus["qids"]))
            plan.append((corpus, n))
            remaining -= n
    if remaining == args.questions:
        raise SystemExit(f"No corpora of size {size} for {args.dataset} seed {args.seed}.")

estimate = sum(lc_estimate(corpus["tokens"], n) for corpus, n in plan)
print(f"{len(plan)} corpora, {sum(n for _, n in plan)} questions. Estimated cost: ${estimate:.4f}")
print("(less if some questions were already answered, or a corpus is still cached)")
if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was spent.")

# ---------------- run ----------------
done = load_done_keys()
for corpus, n in plan:
    asked = ask_corpus(corpus, questions_by_qid, paragraphs_by_pid, args.run, done, limit=n)
    print(f"{corpus['corpus_id']}: asked {asked} of {n}")

# ---------------- results of this run, from the log ----------------
records = [r for r in read_log()
           if r.get("ok") and r.get("run") == args.run and r.get("system") == "lc"
           and r.get("dataset") == args.dataset and r.get("seed") == args.seed]
print()
for size in args.size:
    rows = [r for r in records if r["corpus_tokens"] == size]
    if not rows:
        continue
    scores = [score(r["answer"], questions_by_qid[r["question_id"]]["answers"]) for r in rows]
    writes = sum(r["cache_write_tokens"] > 0 for r in rows)
    print(f"{size:>7,} tokens: {len(rows)} answers, EM {mean(s[0] for s in scores):.2f}, "
          f"F1 {mean(s[1] for s in scores):.2f}, cost ${sum(call_cost(r) for r in rows):.4f} "
          f"(cache writes: {writes}, cache reads: {len(rows) - writes})")