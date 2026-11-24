"""Run the long-context system: each corpus in the prompt (cached), then its questions one by one.

Run from the repo folder, for example:
  python -m scripts.run_lc --run smoke --dataset musique --size 16000 --questions 2
  python -m scripts.run_lc --run pilot --dataset musique --size 16000 256000 --questions 20
  python -m scripts.run_lc --run reasoning-test --prompt reasoning --dataset musique --size 16000 --questions 20
  python -m scripts.run_lc --run full --prompt reasoning --dataset musique --seed 1 --workers 4

Asks up to --questions questions at each size (default: every size, every question; seed 0
unless --seed). --workers corpora run at the same time (default 4); each corpus's questions
go one after another, so each corpus is written to the cache once. Sizes run in the order
given, smallest first by default: usage then grows gradually, which the API's rate limits prefer.
--prompt picks the prompt style (lcvr/prompts.py); the default is answer_only.
Shows a cost estimate and waits for "y" before spending anything. Every call is logged
to logs/calls.jsonl, so running the same command again skips questions already answered.
"""
import argparse
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from statistics import mean

from lcvr.corpus import SIZES, pick_questions
from lcvr.cost import call_cost, lc_estimate
from lcvr.data import read_jsonl
from lcvr.lc import ask_corpus
from lcvr.llm import load_done_keys, read_log
from lcvr.prompts import DEFAULT_STYLE, OUTPUT_ESTIMATE, STYLES
from lcvr.scoring import has_answer_label, score

parser = argparse.ArgumentParser(description="Run the long-context system.")
parser.add_argument("--run", required=True, help="a name for this run, e.g. smoke or pilot")
parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, nargs="+", default=list(SIZES), help="corpus sizes (default: all)")
parser.add_argument("--questions", type=int, default=None, help="questions per size (default: all)")
parser.add_argument("--prompt", choices=STYLES, default=DEFAULT_STYLE, help="prompt style (default: answer_only)")
parser.add_argument("--workers", type=int, default=4, help="corpora running at the same time")
parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
args = parser.parse_args()

questions_by_qid = {q["qid"]: q for q in read_jsonl(f"data/processed/{args.dataset}_questions.jsonl")}
paragraphs_by_pid = {p["pid"]: p for p in read_jsonl(f"data/processed/{args.dataset}_paragraphs.jsonl")}
corpora = read_jsonl(f"data/processed/corpora/{args.dataset}_seed{args.seed}.jsonl")

# ---------------- plan: which corpora, and how many of each one's questions ----------------
plan = []
for size in args.size:
    picked = pick_questions(corpora, size, args.questions)     # the same questions every system uses
    if not picked:
        raise SystemExit(f"No corpora of size {size} for {args.dataset} seed {args.seed}.")
    plan += [(corpus, len(qids)) for corpus, qids in picked]

estimate = sum(lc_estimate(corpus["tokens"], n, OUTPUT_ESTIMATE[args.prompt]) for corpus, n in plan)
print(f"{len(plan)} corpora, {sum(n for _, n in plan)} questions. Estimated cost: ${estimate:.4f}")
print("(less if some questions were already answered, or a corpus is still cached)")
if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was spent.")

# ---------------- run: several corpora at once, each one's questions in order ----------------
done = load_done_keys()
stop = threading.Event()    # set when a corpus fails for good: the others stop before their next question


def run_corpus(job):
    corpus, n = job
    asked = ask_corpus(corpus, questions_by_qid, paragraphs_by_pid, args.run, done, limit=n,
                       style=args.prompt, stop=stop)
    return corpus["corpus_id"], asked, n


with ThreadPoolExecutor(max_workers=args.workers) as executor:
    pending = [executor.submit(run_corpus, job) for job in plan]      # started in plan order
    finished = 0
    try:
        while pending:
            # Wait in 1-second steps: on Windows, Ctrl+C can't interrupt one long wait
            ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                corpus_id, asked, n = future.result()   # a corpus that failed for good raises here
                finished += 1
                print(f"[{finished}/{len(plan)}] {corpus_id}: asked {asked} of {n}", flush=True)
    except BaseException:                           # that error (e.g. no credit left) or Ctrl+C:
        stop.set()                                  # running corpora stop before their next question,
        for future in pending:                      # and the ones still waiting never start
            future.cancel()
        raise

# ---------------- results of this run, from the log ----------------
records = [r for r in read_log()
           if r.get("ok") and r.get("run") == args.run and r.get("system") == "lc"
           and r.get("dataset") == args.dataset and r.get("seed") == args.seed
           and r.get("prompt", DEFAULT_STYLE) == args.prompt]
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
    cut_off = sum(r["stop_reason"] == "max_tokens" for r in rows)
    unlabeled = sum(not has_answer_label(r["answer"]) for r in rows)
    if args.prompt == "reasoning" and (cut_off or unlabeled):
        print(f"         replies cut off by the token cap: {cut_off}, with no 'Answer:' line: {unlabeled}")
