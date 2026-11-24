"""The RAG system: find the paragraphs closest to a question, then answer from the top k.

Retrieval (embedding, search, reranking) runs locally, on the GPU if PyTorch can use one,
otherwise the CPU, so it costs no API money. The cost model prices it as machine time
instead (proposal section 4), which is why the scripts measure it and record the device.
Answering is an API call, priced like any other.
"""
import json
from pathlib import Path

import numpy as np

from lcvr.corpus import render_corpus, render_paragraph
from lcvr.llm import call_claude
from lcvr.prompts import DEFAULT_STYLE, prompt_settings, question_text, style_label

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # small (33M parameters); 384 numbers per vector
# BGE's recommended instruction in front of short search queries (paragraphs get none)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"   # small model trained to rank search results
CANDIDATES = 100     # paragraphs the reranker looks at: enough for the largest top-k we use (80)

# Every RAG setting we try: how many paragraphs go in the prompt, and whether they're reranked
RAG_CONFIGS = [(k, reranked) for k in (5, 10, 20, 40, 80) for reranked in (False, True)]


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


def read_rankings(path):
    """Rankings saved by scripts/rank.py (a line cut off by Ctrl+C is skipped)."""
    rankings = []
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rankings.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rankings


# ---------------- answering ----------------

def rag_labels(ranking, dataset, seed, run, k, reranked, style=DEFAULT_STYLE):
    return {
        "run": run,
        "system": "rag",
        "dataset": dataset,
        "seed": seed,
        "corpus_tokens": ranking["corpus_tokens"],
        "corpus_id": ranking["corpus_id"],
        "question_id": ranking["qid"],
        "k": k,
        "reranked": reranked,
        **style_label(style),
    }


def rag_documents(ranking, k, reranked, paragraphs_by_pid):
    """The top k paragraphs of a question's ranking, best first, as prompt text."""
    order = ranking["reranked"] if reranked else ranking["dense"]
    return render_corpus(paragraphs_by_pid[pid] for pid in order[:k])


def ask_rag(ranking, k, reranked, question, paragraphs_by_pid, labels, style=DEFAULT_STYLE):
    """One RAG answer: the same instructions and layout as long context, with fewer documents.

    No caching: each question gets its own documents, so there is nothing to reuse.
    `style` is the prompt style; make `labels` with the same one (rag_labels(..., style)).
    """
    instructions, max_tokens = prompt_settings(style)
    documents = rag_documents(ranking, k, reranked, paragraphs_by_pid)
    return call_claude(question_text(question), labels, context=documents,
                       system=instructions, max_tokens=max_tokens)
