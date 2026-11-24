"""Checks for the analysis (lcvr/analysis.py) with small made-up data and hand-calculated costs."""
import pytest

from lcvr import analysis
from lcvr.analysis import (bootstrap_ci, compare_at_q, gold_positions, hop_recall, lc_cost, load_calls,
                           paired_difference, rag_cost, retrieval_outcome)

RATES = {"embed_per_paragraph": 0.001, "embed_per_question": 0.0005, "search_per_query": 0.0005,
         "rerank_per_pair": 0.002}


def test_lc_cost_by_hand():
    # 16K corpus, 10 questions per cache: (16,000 x (0.125 + 9 x 0.01) / 10 + 40 x 0.10 + 140 x 0.50) / 1M
    assert lc_cost(16_000, 40, 140, 10) == pytest.approx((344 + 4 + 70) / 1e6)
    # 512K corpus (long-prompt prices), 100 questions: 512,000 x (0.625 + 99 x 0.05) / 100 + 40 x 0.5 + 160 x 2.5
    assert lc_cost(512_000, 40, 160, 100) == pytest.approx((28_544 + 20 + 400) / 1e6)


def test_lc_cost_at_q1_skips_the_cache():
    # A cache write (0.125) costs more than plain input (0.10), so one question pays plain input
    assert lc_cost(16_000, 40, 140, 1) == pytest.approx((1_600 + 4 + 70) / 1e6)


def test_rag_cost_adds_gpu_time():
    # 50 paragraphs, all reranked; seconds = 0.0005 + 0.0005 + 50 x 0.002 + 50 x 0.001 / 10 = 0.106
    assert rag_cost(0.0002, True, 50, 10, RATES) == pytest.approx(0.0002 + 0.106 * 0.526 / 3600)
    # no reranker: 0.0005 + 0.0005 + 0.005 = 0.006 s
    assert rag_cost(0.0002, False, 50, 10, RATES) == pytest.approx(0.0002 + 0.006 * 0.526 / 3600)
    # a big corpus reranks only the top 100 candidates
    assert rag_cost(0, True, 1000, 1000, RATES) == pytest.approx((0.001 + 100 * 0.002 + 0.001) * 0.526 / 3600)


def test_bootstrap_ci():
    assert bootstrap_ci([0.5] * 30, [0] * 30) == (0.5, 0.5)
    values = [0, 1] * 150
    low, high = bootstrap_ci(values, [0] * 150 + [1] * 150)
    assert low < 0.5 < high and high - low < 0.15
    assert bootstrap_ci(values, [0] * 300) == bootstrap_ci(values, [0] * 300)     # same seed, same interval


def call(seed, qid, f1, em=None, **labels):
    return {"seed": seed, "question_id": qid, "dataset": "musique", "f1": f1, "em": f1 if em is None else em, **labels}


def calls_for(f1s):
    return {("musique", i % 3, f"q{i}"): call(i % 3, f"q{i}", f1) for i, f1 in enumerate(f1s)}


def test_paired_difference():
    a, b = calls_for([1.0, 0.5, 1.0, 0.0]), calls_for([0.5, 0.5, 0.0, 0.0])
    diff, (low, high), n = paired_difference(a, b)
    assert diff == pytest.approx(0.375) and n == 4 and low <= diff <= high


def test_matched_cost_rule():
    lc_calls = calls_for([0.6] * 60)
    lc_row = {"f1": 0.6, "cost_at": {1: 0.01, 1000: 0.0005}}
    rag_rows = [{"k": 5, "reranked": False, "f1": 0.5, "cost": 0.001},
                {"k": 80, "reranked": True, "f1": 0.7, "cost": 0.004}]
    rag_calls = {(5, False): calls_for([0.5] * 60), (80, True): calls_for([0.7] * 60)}
    price = lambda row, q: row["cost"]  # noqa: E731

    # At Q = 1 both RAG settings are cheaper: the best (top 80) beats long context
    cell = compare_at_q(lc_row, lc_calls, rag_rows, rag_calls, 1, price)
    assert (cell["rag_k"], cell["winner"]) == (80, "RAG") and cell["diff"] == pytest.approx(-0.1)
    assert (cell["iso_k"], cell["iso_ratio"]) == (80, pytest.approx(0.4))

    # At Q = 1,000 long context costs less than every RAG setting
    cell = compare_at_q(lc_row, lc_calls, rag_rows, rag_calls, 1000, price)
    assert cell["rag_k"] is None and cell["winner"].startswith("LC")

    # Only top 5 fits a middle budget, and long context beats it
    lc_row["cost_at"][10] = 0.002
    cell = compare_at_q(lc_row, lc_calls, rag_rows, rag_calls, 10, price)
    assert (cell["rag_k"], cell["winner"]) == (5, "LC")


def test_matched_cost_tie():
    noisy = [0.0, 1.0] * 30
    lc_row = {"f1": 0.5, "cost_at": {1: 0.01}}
    rag_rows = [{"k": 5, "reranked": False, "f1": 0.5, "cost": 0.001}]
    cell = compare_at_q(lc_row, calls_for(noisy), rag_rows, {(5, False): calls_for(noisy[::-1])}, 1,
                        lambda row, q: row["cost"])
    assert cell["winner"] == "tie"


RANKING = {"dense": ["p1", "g1", "p2", "g2", "p3"], "reranked": ["g1", "g2", "p1", "p2", "p3"]}


@pytest.mark.parametrize("k, reranked, em, expected", [
    (2, True, 1, ("correct", 2, 2)),
    (2, True, 0, ("reasoning failure", 2, 2)),
    (2, False, 0, ("partial retrieval", 1, 2)),
    (1, False, 0, ("retrieval miss", 0, 2)),
])
def test_retrieval_outcome(k, reranked, em, expected):
    c = {"k": k, "reranked": reranked, "em": em}
    assert retrieval_outcome(c, RANKING, ["g1", "g2"]) == expected


def test_hop_recall_and_positions():
    assert hop_recall(RANKING, ["g1", "g2"], 2, False) == [True, False]
    assert gold_positions(["a", "g1", "b", "c", "g2"], ["g1", "g2"]) == [0.25, 1.0]


def test_load_calls_keeps_the_run_and_the_last_answer():
    records = [{"run": "full", "ok": True, "key": "a", "answer": "first"},
               {"run": "full", "ok": False, "key": "b"},
               {"run": "pilot", "ok": True, "key": "c"},
               {"run": "full", "ok": True, "key": "a", "answer": "second"}]
    assert [c["answer"] for c in load_calls(records, "full")] == ["second"]


def test_summarize_one_setting():
    tokens = {"ok": True, "input_tokens": 1000, "cache_write_tokens": 0, "cache_read_tokens": 0,
              "output_tokens": 100, "latency_s": 1.0, "stop_reason": "end_turn"}
    cs = [{**call(s, f"q{i}", f1, system="rag", corpus_tokens=16000, k=5, reranked=False), **tokens}
          for i, (s, f1) in enumerate([(0, 1.0), (0, 0.0), (1, 0.5), (2, 0.5)])]
    (row,) = analysis.summarize(cs)
    assert (row["n"], row["f1"], row["k"]) == (4, 0.5, 5)
    assert row["api_cost"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    assert row["f1_by_seed"] == {0: 0.5, 1: 0.5, 2: 0.5}
