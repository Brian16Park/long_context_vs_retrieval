"""Step 6c: RAG answers. For each question: the top k paragraphs from its ranking, then the question.

Run from the repo folder, for example:
  python -m scripts.run_rag --run pilot --dataset musique --size 16000 256000 --questions 20

Tries every setting in RAG_CONFIGS (top-k 5, 10, 20, 40, 80, each with and without the
reranker), so that's 10 calls per question. Needs the rankings from scripts/rank.py.
Shows a cost estimate and waits for "y" before spending anything. Every call is logged,
so running the same command again skips calls already answered. Calls run --workers at
a time (default 4). If a call fails for good (for example, no credit left), the calls
still waiting are cancelled and the script stops; fix the problem and run it again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from lcvr.corpus import CHARS_PER_TOKEN, SIZES, estimate_sizes, pick_questions
from lcvr.cost import call_cost, call_estimate
from lcvr.data import read_jsonl
from lcvr.llm import label_key, load_done_keys, read_log
from lcvr.rag import RAG_CONFIGS, ask_rag, rag_labels, read_rankings, recall
from lcvr.scoring import score

parser = argparse.ArgumentParser(description="Run the RAG system.")
parser.add_argument("--run", required=True, help="a name for this run, e.g. pilot")
parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, nargs="+", default=list(SIZES), help="corpus sizes (default: all)")
parser.add_argument("--questions", type=int, default=None, help="questions per size (default: all)")
parser.add_argument("--workers", type=int, default=4, help="calls running at the same time")
parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
args = parser.parse_args()

PROCESSED = Path("data/processed")
questions = {q["qid"]: q for q in read_jsonl(PROCESSED / f"{args.dataset}_questions.jsonl")}
paragraphs = read_jsonl(PROCESSED / f"{args.dataset}_paragraphs.jsonl")
paragraphs_by_pid = {p["pid"]: p for p in paragraphs}
est = estimate_sizes(paragraphs, CHARS_PER_TOKEN[args.dataset])
corpora = read_jsonl(PROCESSED / "corpora" / f"{args.dataset}_seed{args.seed}.jsonl")
rankings = {(r["corpus_id"], r["qid"]): r
            for r in read_rankings(PROCESSED / "rankings" / f"{args.dataset}_seed{args.seed}.jsonl")}

# ---------------- plan: the same questions as every other system, 10 settings each ----------------
jobs, unranked = [], 0
for size in args.size:
    for corpus, qids in pick_questions(corpora, size, args.questions):
        for qid in qids:
            ranking = rankings.get((corpus["corpus_id"], qid))
            if ranking is None:
                unranked += 1
                continue
            jobs += [(ranking, k, reranked) for k, reranked in RAG_CONFIGS]
if unranked:
    raise SystemExit(f"{unranked} of these questions have no ranking yet. Run scripts.rank first, "
                     "with the same --dataset, --seed, --size and --questions.")

done = load_done_keys()
todo = [(ranking, k, reranked) for ranking, k, reranked in jobs
        if label_key(rag_labels(ranking, args.dataset, args.seed, args.run, k, reranked)) not in done]


def input_estimate(ranking, k, reranked):
    """Prompt size: the k paragraphs, plus about 150 tokens of instructions and question."""
    order = ranking["reranked"] if reranked else ranking["dense"]
    return sum(est[pid] for pid in order[:k]) + 150


estimate = sum(call_estimate(input_estimate(*job), output_tokens=10) for job in todo)
print(f"{len(jobs)} calls planned, {len(todo)} still to do. Estimated cost: ${estimate:.4f}")
if todo and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was spent.")


# ---------------- run ----------------
def answer(job):
    ranking, k, reranked = job
    labels = rag_labels(ranking, args.dataset, args.seed, args.run, k, reranked)
    return ask_rag(ranking, k, reranked, questions[ranking["qid"]], paragraphs_by_pid, labels)


with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(answer, job) for job in todo]
    try:
        for n, future in enumerate(as_completed(futures), 1):
            future.result()                       # a call that failed for good raises its error here
            if n % 50 == 0 or n == len(todo):
                print(f"   {n}/{len(todo)} answered", flush=True)
    except BaseException:                         # that error (e.g. no credit left) or Ctrl+C:
        for future in futures:                    # cancel the calls still waiting to start,
            future.cancel()                       # let the few already running finish, then stop
        raise

# ---------------- results of this run, from the log ----------------
records = [r for r in read_log()
           if r.get("ok") and r.get("run") == args.run and r.get("system") == "rag"
           and r.get("dataset") == args.dataset and r.get("seed") == args.seed]
print(f"\nRAG, {args.dataset} seed {args.seed}, run '{args.run}':")
print("   size    k  reranker  answers    EM    F1  recall  avg prompt tokens      cost")
for size in args.size:
    for k, reranked in RAG_CONFIGS:
        rows = [r for r in records if r["corpus_tokens"] == size and r["k"] == k and r["reranked"] == reranked]
        if not rows:
            continue
        scores = [score(r["answer"], questions[r["question_id"]]["answers"]) for r in rows]
        found = mean(recall(rankings[(r["corpus_id"], r["question_id"])]["reranked" if reranked else "dense"],
                            questions[r["question_id"]]["gold_pids"], k) for r in rows)
        print(f"   {size // 1000:>3}K  {k:>3}  {'yes' if reranked else 'no':<8}  {len(rows):>7}  "
              f"{mean(s[0] for s in scores):.2f}  {mean(s[1] for s in scores):.2f}  {found:>6.2f}  "
              f"{mean(r['input_tokens'] for r in rows):>17,.0f}  ${sum(call_cost(r) for r in rows):.4f}")