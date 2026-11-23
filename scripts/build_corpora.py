"""Step 4b-3: build the shared corpora for both datasets, every seed and every size.

Run from the repo folder:  python -m scripts.build_corpora
Writes data/processed/corpora/<dataset>_seed<seed>.jsonl, one corpus per line.
Free (token counting only). Takes several minutes; a file that already exists is
skipped, so after a crash just run it again.
"""
from pathlib import Path

from lcvr.corpus import CHARS_PER_TOKEN, SEEDS, build_corpora, check_corpora, estimate_sizes
from lcvr.data import read_jsonl, write_jsonl
from lcvr.llm import count_tokens

PROCESSED = Path("data/processed")
OUT = PROCESSED / "corpora"


def summarize(corpora):
    for size in sorted({c["target_tokens"] for c in corpora}):
        group = [c for c in corpora if c["target_tokens"] == size]
        n_questions = [len(c["qids"]) for c in group]
        tokens = [c["tokens"] for c in group]
        filler = sum(c["n_filler"] for c in group) / sum(len(c["pids"]) for c in group)
        print(f"   {size // 1000:>3}K: {len(group):>2} corpora, {min(n_questions)}-{max(n_questions)} questions each, "
              f"{min(tokens):,}-{max(tokens):,} tokens, filler {filler:.0%} of paragraphs")


for dataset in ("musique", "2wiki"):
    questions = read_jsonl(PROCESSED / f"{dataset}_questions.jsonl")
    paragraphs = read_jsonl(PROCESSED / f"{dataset}_paragraphs.jsonl")
    by_pid = {p["pid"]: p for p in paragraphs}
    est = estimate_sizes(paragraphs, CHARS_PER_TOKEN[dataset])

    for seed in SEEDS:
        out = OUT / f"{dataset}_seed{seed}.jsonl"
        if out.exists():
            print(f"{out} already exists, skipping")
            continue
        corpora = build_corpora(dataset, seed, questions, by_pid, est, count_tokens)
        problems = check_corpora(corpora, {q["qid"]: q for q in questions})
        if problems:
            print(f"\n{len(problems)} problems, so {out} was NOT written. First few:")
            for problem in problems[:10]:
                print("  ", problem)
            raise SystemExit(1)
        write_jsonl(out, corpora)
        print(f"{dataset} seed {seed}: all checks passed, saved {out}")
        summarize(corpora)