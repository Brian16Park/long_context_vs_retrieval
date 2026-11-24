"""Checks for the RAG system (lcvr/rag.py), with fake models: no downloads, no API calls."""
import json

import numpy as np

from lcvr import rag
from lcvr.prompts import INSTRUCTIONS, MAX_ANSWER_TOKENS, MAX_REASONING_TOKENS, REASONING_INSTRUCTIONS
from lcvr.rag import (embed, load_vectors, rag_documents, rag_labels, read_rankings, recall, rerank,
                      save_vectors, top_k)


def test_vectors_survive_saving_and_loading(tmp_path):
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_vectors(tmp_path / "v.npz", ["mq_a", "mq_b"], vectors)
    ids, loaded = load_vectors(tmp_path / "v.npz")
    assert ids == ["mq_a", "mq_b"]
    assert np.array_equal(loaded, vectors)


def test_top_k_most_similar_first():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    query = np.array([0, 1], dtype=np.float32)
    assert list(top_k(query, vectors, 2)) == [1, 2]           # scores: 0, 1, 0.8
    assert list(top_k(query, vectors, 10)) == [1, 2, 0]       # k larger than the rows: all of them


def test_embed_asks_for_unit_vectors_and_passes_the_prompt():
    seen = {}

    class FakeModel:
        def encode(self, texts, **options):
            seen.update(options)
            return np.ones((len(texts), 4))

    vectors = embed(FakeModel(), ["a question"], prompt="Search: ")
    assert seen["prompt"] == "Search: " and seen["normalize_embeddings"] is True
    assert vectors.dtype == np.float32 and vectors.shape == (1, 4)


class FakeReranker:
    """Scores a (question, paragraph) pair by how many of the question's words the paragraph has."""

    def __init__(self):
        self.pairs_scored = 0

    def predict(self, pairs, **options):
        self.pairs_scored += len(pairs)
        return [sum(word in text for word in question.split()) for question, text in pairs]


PARAGRAPHS = {"p1": {"pid": "p1", "title": "A", "text": "red"},
              "p2": {"pid": "p2", "title": "B", "text": "red blue"},
              "p3": {"pid": "p3", "title": "C", "text": "green"}}


def test_rerank_puts_the_best_scored_paragraphs_first():
    question = {"qid": "q1", "question": "red blue"}
    assert rerank(FakeReranker(), question, ["p1", "p3", "p2"], PARAGRAPHS, {}) == ["p2", "p1", "p3"]


def test_rerank_scores_each_pair_only_once():
    question = {"qid": "q1", "question": "red blue"}
    reranker, scores = FakeReranker(), {}
    rerank(reranker, question, ["p1", "p2"], PARAGRAPHS, scores)
    rerank(reranker, question, ["p1", "p2", "p3"], PARAGRAPHS, scores)   # e.g. a bigger corpus
    assert reranker.pairs_scored == 3                                     # p1 and p2 weren't redone


def test_recall():
    assert recall(["a", "b", "c", "d"], ["b", "d"], k=2) == 0.5           # b found, d not yet
    assert recall(["a", "b", "c", "d"], ["b", "d"], k=4) == 1.0


RANKING = {"corpus_id": "c1", "corpus_tokens": 16000, "qid": "q1",
           "dense": ["p1", "p2", "p3"], "reranked": ["p3", "p1", "p2"]}


def test_rag_documents_take_the_top_k_of_the_chosen_ranking():
    assert rag_documents(RANKING, 2, False, PARAGRAPHS) == "# A\nred\n\n# B\nred blue"
    assert rag_documents(RANKING, 1, True, PARAGRAPHS) == "# C\ngreen"


def test_ask_rag_sends_the_documents_uncached_with_the_shared_instructions(monkeypatch):
    seen = {}

    def fake_call_claude(question, labels, **options):
        seen.update(question=question, labels=labels, **options)
        return "an answer", {}

    monkeypatch.setattr(rag, "call_claude", fake_call_claude)
    labels = rag_labels(RANKING, "musique", 0, "test", k=2, reranked=True)
    rag.ask_rag(RANKING, 2, True, {"qid": "q1", "question": "What?"}, PARAGRAPHS, labels)
    assert seen["question"] == "Question: What?"
    assert seen["context"] == "# C\ngreen\n\n# A\nred"
    assert seen.get("cache_context", False) is False                     # nothing to reuse
    assert seen["system"] == INSTRUCTIONS and seen["max_tokens"] == MAX_ANSWER_TOKENS
    assert seen["labels"]["k"] == 2 and seen["labels"]["reranked"] is True
    assert "prompt" not in seen["labels"]                                # answer-only: labels as before


def test_ask_rag_reasoning_style(monkeypatch):
    seen = {}

    def fake_call_claude(question, labels, **options):
        seen.update(labels=labels, **options)
        return "an answer", {}

    monkeypatch.setattr(rag, "call_claude", fake_call_claude)
    labels = rag_labels(RANKING, "musique", 0, "test", k=2, reranked=False, style="reasoning")
    rag.ask_rag(RANKING, 2, False, {"qid": "q1", "question": "What?"}, PARAGRAPHS, labels, style="reasoning")
    assert seen["system"] == REASONING_INSTRUCTIONS and seen["max_tokens"] == MAX_REASONING_TOKENS
    assert seen["labels"]["prompt"] == "reasoning"


def test_read_rankings_skips_a_cut_off_line(tmp_path):
    path = tmp_path / "rankings.jsonl"
    path.write_text(json.dumps(RANKING) + '\n{"corpus_id": "c2", "qi\n', encoding="utf-8")
    assert [r["qid"] for r in read_rankings(path)] == ["q1"]
