"""Retrieval (RAG): turn paragraphs and questions into vectors, then find the closest paragraphs.

All of this runs locally on the CPU, so it costs no API money. The cost model prices
it as CPU time instead (proposal section 4), which is why the scripts measure it.
"""
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # small (33M parameters); 384 numbers per vector
# BGE's recommended instruction in front of short search queries (paragraphs get none)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def load_embedder():
    """The embedding model, on the CPU. Downloaded the first time (about 130 MB)."""
    from sentence_transformers import SentenceTransformer   # slow to import, so only when needed
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


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