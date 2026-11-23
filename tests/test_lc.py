"""Checks for the long-context system (lcvr/lc.py) and logging, with a fake model: no API calls."""
import json
import threading

from lcvr import lc, llm
from lcvr.llm import label_key
from lcvr.prompts import INSTRUCTIONS

CORPUS = {"corpus_id": "c1", "dataset": "musique", "seed": 0, "target_tokens": 16000,
          "qids": ["q1", "q2", "q3"], "pids": ["p1", "p2"]}
QUESTIONS = {q: {"qid": q, "question": f"What is {q}?"} for q in ["q1", "q2", "q3"]}
PARAGRAPHS = {"p1": {"pid": "p1", "title": "A", "text": "a"}, "p2": {"pid": "p2", "title": "B", "text": "b"}}


def fake_model(monkeypatch):
    """Replace the real API call with one that just records what it was asked."""
    calls = []

    def fake_call_claude(question, labels, **options):
        calls.append({"question": question, "labels": labels, **options})
        return "an answer", {}

    monkeypatch.setattr(lc, "call_claude", fake_call_claude)
    return calls


def test_asks_in_order_with_the_corpus_cached(monkeypatch):
    calls = fake_model(monkeypatch)
    asked = lc.ask_corpus(CORPUS, QUESTIONS, PARAGRAPHS, "test", done=set(), limit=2)
    assert asked == 2
    assert [c["question"] for c in calls] == ["Question: What is q1?", "Question: What is q2?"]
    assert calls[0]["context"] == "# A\na\n\n# B\nb"           # the whole corpus, every time
    assert calls[0]["cache_context"] is True
    assert calls[0]["system"] == INSTRUCTIONS
    assert calls[1]["labels"]["question_id"] == "q2"
    assert calls[1]["labels"]["corpus_tokens"] == 16000


def test_skips_questions_already_answered(monkeypatch):
    calls = fake_model(monkeypatch)
    done = {label_key(lc.lc_labels(CORPUS, "q1", "test"))}
    assert lc.ask_corpus(CORPUS, QUESTIONS, PARAGRAPHS, "test", done) == 2
    assert [c["labels"]["question_id"] for c in calls] == ["q2", "q3"]


def test_log_reading_skips_half_written_lines(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"key": "a", "ok": True}) + "\n"
                   + '{"key": "b", "ok": tr\n'                       # cut off by a crash
                   + json.dumps({"key": "c", "ok": False}) + "\n", encoding="utf-8")
    monkeypatch.setattr(llm, "LOG_FILE", log)
    assert [r["key"] for r in llm.read_log()] == ["a", "c"]
    assert llm.load_done_keys() == {"a"}                            # only successful calls count


def test_calls_finishing_at_the_same_time_dont_mix_their_log_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LOG_FILE", tmp_path / "calls.jsonl")
    big = "x" * 5000                                                # long lines are easier to mix up

    def write_many(worker):
        for i in range(50):
            llm.save_to_log({"key": f"{worker}-{i}", "ok": True, "answer": big})

    threads = [threading.Thread(target=write_many, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(llm.read_log()) == 400                               # every line still whole