"""Step 6b: rank every question's corpus for RAG. Free: runs locally (the reranker uses the GPU
if PyTorch can use one, otherwise the CPU).

For each question, in its corpus at each size: the 100 paragraphs closest to it by
embedding ("dense"), and the same 100 reordered by the reranker ("reranked").

Run from the repo folder, for example:
  python -m scripts.rank --dataset musique --size 16000 256000 --questions 20   (the pilot's questions)
  python -m scripts.rank --dataset musique                                      (every size, every question)

Appends to data/processed/rankings/<dataset>_seed<seed>.jsonl. Questions already ranked
are skipped, so it can be stopped (Ctrl+C) and restarted. Prints recall@k at the end.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path
from statistics import mean

from lcvr.corpus import SIZES, pick_questions
from lcvr.data import read_jsonl
from lcvr.llm import ends_with_newline
from lcvr.rag import (CANDIDATES, describe_device, load_reranker, load_vectors, read_rankings,
                      recall, rerank, top_k)

parser = argparse.ArgumentParser(description="Rank paragraphs for RAG.")
parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, nargs="+", default=list(SIZES), help="corpus sizes (default: all)")
parser.add_argument("--questions", type=int, default=None, help="questions per size (default: all)")
args = parser.parse_args()

PROCESSED = Path("data/processed")
OUT = PROCESSED / "rankings" / f"{args.dataset}_seed{args.seed}.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

questions = {q["qid"]: q for q in read_jsonl(PROCESSED / f"{args.dataset}_questions.jsonl")}
paragraphs = {p["pid"]: p for p in read_jsonl(PROCESSED / f"{args.dataset}_paragraphs.jsonl")}
corpora = read_jsonl(PROCESSED / "corpora" / f"{args.dataset}_seed{args.seed}.jsonl")
pids, p_vectors = load_vectors(PROCESSED / "embeddings" / f"{args.dataset}_paragraphs.npz")
qids, q_vectors = load_vectors(PROCESSED / "embeddings" / f"{args.dataset}_questions.npz")
p_row = {pid: i for i, pid in enumerate(pids)}
q_row = {qid: i for i, qid in enumerate(qids)}

done = {(r["corpus_id"], r["qid"]) for r in read_rankings(OUT)}
work = [(corpus, qid)
        for size in args.size
        for corpus, chosen in pick_questions(corpora, size, args.questions)
        for qid in chosen if (corpus["corpus_id"], qid) not in done]
print(f"{len(work)} questions to rank ({len(done)} already done)")

if work:
    reranker = load_reranker()
    device = describe_device()
    print(f"reranking on {device}")
    scores = {}                                  # (question, paragraph) -> reranker score
    dense_seconds = rerank_seconds = 0.0         # wall-clock time: what a rented machine bills for
    corpus_id, matrix = None, None
    if not ends_with_newline(OUT):               # a line cut off by Ctrl+C: close it off
        with open(OUT, "a", encoding="utf-8") as f:
            f.write("\n")
    with open(OUT, "a", encoding="utf-8") as f:
        for n, (corpus, qid) in enumerate(work, 1):
            start = time.perf_counter()
            if corpus["corpus_id"] != corpus_id:     # this corpus's paragraph vectors, built once
                corpus_id = corpus["corpus_id"]
                matrix = p_vectors[[p_row[p] for p in corpus["pids"]]]
            dense = [corpus["pids"][i] for i in top_k(q_vectors[q_row[qid]], matrix, CANDIDATES)]
            middle = time.perf_counter()
            reranked = rerank(reranker, questions[qid], dense, paragraphs, scores)
            dense_seconds += middle - start
            rerank_seconds += time.perf_counter() - middle

            f.write(json.dumps({"corpus_id": corpus["corpus_id"], "corpus_tokens": corpus["target_tokens"],
                                "qid": qid, "dense": dense, "reranked": reranked}) + "\n")
            f.flush()
            if n % 100 == 0 or n == len(work):
                print(f"   {n}/{len(work)} ranked ({len(scores):,} pairs scored by the reranker)", flush=True)

    timing = {"date": date.today().isoformat(), "device": device, "questions": len(work),
              "dense_seconds": round(dense_seconds, 2),
              "rerank_pairs": len(scores), "rerank_seconds": round(rerank_seconds, 1)}
    with open(OUT.with_name(OUT.stem + "_timing.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(timing) + "\n")

# ---------------- recall@k over everything ranked so far ----------------
rankings = read_rankings(OUT)
ks = (5, 10, 20, 40, 80)
print(f"\nShare of gold paragraphs in the top k, {args.dataset} seed {args.seed} (dense / reranked):")
print("   size " + "".join(f"{f'k={k}':>14}" for k in ks))
for size in sorted({r["corpus_tokens"] for r in rankings}):
    rows = [r for r in rankings if r["corpus_tokens"] == size]
    cells = []
    for k in ks:
        dense = mean(recall(r["dense"], questions[r["qid"]]["gold_pids"], k) for r in rows)
        reranked = mean(recall(r["reranked"], questions[r["qid"]]["gold_pids"], k) for r in rows)
        cells.append(f"{dense:.2f} / {reranked:.2f}")
    print(f"   {size // 1000:>3}K " + "".join(f"{c:>14}" for c in cells) + f"   ({len(rows)} questions)")