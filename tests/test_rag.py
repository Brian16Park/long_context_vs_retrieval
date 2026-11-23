"""Checks for retrieval helpers (lcvr/rag.py), with a fake embedding model: no downloads."""
import numpy as np

from lcvr.rag import embed, load_vectors, save_vectors, top_k


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