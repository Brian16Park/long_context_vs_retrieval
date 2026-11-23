"""Step 6a: embed every paragraph and every question once, for retrieval. Free: runs locally
(on the GPU if PyTorch can use one, otherwise the CPU).

Run from the repo folder:  python -m scripts.embed
Writes to data/processed/embeddings/: <dataset>_paragraphs.npz, <dataset>_questions.npz, and
<dataset>_timing.json (time and device, for the cost model). A dataset that already has its
timing file is skipped. The first run downloads the embedding model (about 130 MB).
"""
import json
import time
from datetime import date
from pathlib import Path
from statistics import mean

from lcvr.corpus import render_paragraph
from lcvr.data import read_jsonl
from lcvr.rag import (EMBEDDING_MODEL, QUERY_PROMPT, describe_device, embed, load_embedder,
                      save_vectors, top_k)

PROCESSED = Path("data/processed")
OUT = PROCESSED / "embeddings"


def timed_embed(model, texts, prompt=None):
    """Embed, and measure CPU time (all cores added up) and wall-clock time."""
    cpu, wall = time.process_time(), time.perf_counter()
    vectors = embed(model, texts, prompt)
    return vectors, time.process_time() - cpu, time.perf_counter() - wall


def sanity_check(questions, pids, p_vectors, q_vectors, k=5):
    """Within each question's own paragraphs, what share of its gold ones rank in the top k?"""
    row = {pid: i for i, pid in enumerate(pids)}
    found, random_share = [], []
    for q, q_vector in zip(questions, q_vectors):
        own = [row[p] for p in q["pids"]]
        best = {q["pids"][i] for i in top_k(q_vector, p_vectors[own], k)}
        gold = set(q["gold_pids"])
        found.append(len(gold & best) / len(gold))
        random_share.append(min(k, len(own)) / len(own))
    return mean(found), mean(random_share)


todo = [d for d in ("musique", "2wiki") if not (OUT / f"{d}_timing.json").exists()]
if not todo:
    raise SystemExit("Both datasets are already embedded.")
model = load_embedder()

for dataset in todo:
    paragraphs = read_jsonl(PROCESSED / f"{dataset}_paragraphs.jsonl")
    questions = read_jsonl(PROCESSED / f"{dataset}_questions.jsonl")

    print(f"{dataset}: embedding {len(paragraphs):,} paragraphs")
    texts = [render_paragraph(p) for p in paragraphs]                  # the same text the model sees
    p_vectors, p_cpu, p_wall = timed_embed(model, texts)
    pids = [p["pid"] for p in paragraphs]
    save_vectors(OUT / f"{dataset}_paragraphs.npz", pids, p_vectors)

    print(f"{dataset}: embedding {len(questions):,} questions")
    q_vectors, q_cpu, q_wall = timed_embed(model, [q["question"] for q in questions], QUERY_PROMPT)
    save_vectors(OUT / f"{dataset}_questions.npz", [q["qid"] for q in questions], q_vectors)

    timing = {
        "model": EMBEDDING_MODEL, "date": date.today().isoformat(), "device": describe_device(),
        "paragraphs": len(paragraphs), "paragraph_chars": sum(len(t) for t in texts),
        "paragraph_cpu_seconds": round(p_cpu, 1), "paragraph_wall_seconds": round(p_wall, 1),
        "questions": len(questions),
        "question_cpu_seconds": round(q_cpu, 1), "question_wall_seconds": round(q_wall, 1),
    }
    (OUT / f"{dataset}_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")   # written last
    print(f"{dataset}: done in {p_wall + q_wall:.0f} s ({p_cpu + q_cpu:.0f} CPU-seconds)")

    share, random_share = sanity_check(questions, pids, p_vectors, q_vectors)
    print(f"   sanity check: {share:.0%} of gold paragraphs rank in the top 5 of their question's "
          f"own paragraphs (random order would give {random_share:.0%})\n")