"""Retrieval (RAG): turn paragraphs and questions into vectors, then find the closest paragraphs.

All of this runs locally (on the GPU if PyTorch can use one, otherwise the CPU), so it
costs no API money. The cost model prices it as machine time instead (proposal section 4),
which is why the scripts measure it and record which device did the work.
"""
from pathlib import Path

import numpy as np

from lcvr.corpus import render_paragraph

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # small (33M parameters); 384 numbers per vector
# BGE's recommended instruction in front of short search queries (paragraphs get none)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"   # small model trained to rank search results
CANDIDATES = 100     # paragraphs the reranker looks at: enough for the largest top-k we use (80)


def pick_device():
    """'cuda' (the GPU) if PyTorch can use one, otherwise 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device():
    """Which device does the work, for the timing records, e.g. 'cuda (NVIDIA GeForce RTX 3060)'."""
    import torch
    return f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"


def load_embedder():
    """The embedding model, on the GPU if there is one. Downloaded the first time (about 130 MB)."""
    from sentence_transformers import SentenceTransformer   # slow to import, so only when needed
    return SentenceTransformer(EMBEDDING_MODEL, device=pick_device())


def embed(model, texts, prompt=None):
    """One vector of length 1 per text, so the dot product of two vectors is their cosine similarity."""
    vectors = model.encode(texts, prompt=prompt, batch_size=64, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    return np.asarray(vectors, dtype=np.float32)


def save_vectors(path, ids, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=vectors)


def load_vectors(path):
    """-> (list of ids, matrix with one row per id, in the same order)"""
    data = np.load(path)
    return [str(i) for i in data["ids"]], data["vectors"]


def top_k(query_vector, vectors, k):
    """Row numbers of the k rows of `vectors` most similar to `query_vector`, most similar first."""
    scores = vectors @ query_vector
    return np.argsort(-scores, kind="stable")[:k]


def load_reranker():
    """The reranker, on the GPU if there is one. Downloaded the first time (about 90 MB)."""
    from sentence_transformers import CrossEncoder         # slow to import, so only when needed
    return CrossEncoder(RERANK_MODEL, device=pick_device())


def rerank(reranker, question, candidates, paragraphs_by_pid, scores):
    """`candidates` (paragraph IDs) reordered by the reranker, best first.

    The reranker reads the question and a paragraph together, which is slower than
    comparing two vectors but more accurate. `scores` remembers every (question, paragraph)
    pair already scored, so a pair that turns up at several corpus sizes is scored once.
    """
    qid = question["qid"]
    new = [pid for pid in candidates if (qid, pid) not in scores]
    if new:
        pairs = [(question["question"], render_paragraph(paragraphs_by_pid[pid])) for pid in new]
        for pid, score in zip(new, reranker.predict(pairs, batch_size=64, show_progress_bar=False)):
            scores[(qid, pid)] = float(score)
    return sorted(candidates, key=lambda pid: -scores[(qid, pid)])    # ties keep their dense order


def recall(ranking, gold_pids, k):
    """Share of the gold paragraphs found in the first k of `ranking`."""
    gold = set(gold_pids)
    return len(gold & set(ranking[:k])) / len(gold)