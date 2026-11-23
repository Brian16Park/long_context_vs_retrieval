"""Checks for corpus building (lcvr/corpus.py), with a fake token counter: no API calls."""
import random

from lcvr.corpus import (TOLERANCE, build_corpora, check_corpora, estimate_sizes, fit_corpus,
                         group_questions, pick_filler, pick_questions, render_corpus)


def test_render_format():
    paragraphs = [{"pid": "a", "title": "A", "text": "first"}, {"pid": "b", "title": "B", "text": "second"}]
    assert render_corpus(paragraphs) == "# A\nfirst\n\n# B\nsecond"


def test_groups_fit_and_come_out_even():
    # 12 questions, 4 paragraphs of 10 tokens each -> 40 tokens per question
    questions = [{"qid": f"q{i}", "pids": [f"p{i}_{j}" for j in range(4)]} for i in range(12)]
    est = {p: 10 for q in questions for p in q["pids"]}
    groups = group_questions(questions, est, size=100)
    assert sorted(qid for g in groups for qid in g.qids) == sorted(q["qid"] for q in questions)
    assert all(g.tokens <= 97 for g in groups)               # MAX_FILL leaves room for the exact fit
    assert [len(g.qids) for g in groups] == [2] * 6           # 2 questions (80 tokens) per group


def test_shared_paragraphs_count_once():
    questions = [{"qid": "q1", "pids": ["a", "b", "c", "d"]}, {"qid": "q2", "pids": ["c", "d", "e", "f"]}]
    est = {p: 10 for p in "abcdef"}
    groups = group_questions(questions, est, size=100)
    assert len(groups) == 1
    assert groups[0].tokens == 60                             # c and d counted once


def test_pick_filler_fills_the_budget_without_going_over():
    est = {"c0": 10, "f0": 10, "f1": 30, "f2": 10, "f3": 10}
    picked = pick_filler(["c0", "f0", "f1", "f2", "f3"], {"c0"}, est, budget=35, ratio=1.0)
    assert picked == ["f0", "f2", "f3"]                       # c0 is used; f1 (30) no longer fits


def test_fit_corpus_corrects_wrong_estimates():
    rng = random.Random(0)
    est = {f"p{i}": rng.uniform(5, 20) for i in range(300)}
    true = {p: size * 1.15 for p, size in est.items()}       # estimates are 15% too low

    def measure(pids):
        return round(sum(true[p] for p in pids))

    core = [f"p{i}" for i in range(20)]
    pool = [f"p{i}" for i in range(20, 300)]
    order, tokens = fit_corpus(core, pool, est, 1000, measure, "seed")
    assert abs(tokens - 1000) <= TOLERANCE * 1000
    assert set(core) <= set(order)
    assert fit_corpus(core, pool, est, 1000, measure, "seed")[0] == order   # same inputs, same order


def fake_dataset(n_questions=40, n_extra=300):
    rng = random.Random(1)
    paragraphs, questions = [], []

    def new_paragraph():
        pid = f"p{len(paragraphs)}"
        paragraphs.append({"pid": pid, "title": f"Title {pid}", "text": "w" * rng.randint(10, 120)})
        return pid

    for i in range(n_questions):
        pids = [new_paragraph() for _ in range(3)]
        questions.append({"qid": f"q{i}", "pids": pids, "gold_pids": pids[:2]})
    for _ in range(n_extra):
        new_paragraph()
    return questions, paragraphs


def test_build_corpora_end_to_end():
    questions, paragraphs = fake_dataset()
    est = estimate_sizes(paragraphs, chars_per_token=2.0)

    def fake_count_tokens(text):                               # the "real" ratio is 1.8, not 2.0
        return round(len(text) / 1.8)

    by_qid = {q["qid"]: q for q in questions}
    corpora = build_corpora("fake", 0, questions, {p["pid"]: p for p in paragraphs}, est,
                            fake_count_tokens, sizes=(2000, 6000), n_questions=30)
    assert check_corpora(corpora, by_qid) == []
    assert {c["target_tokens"] for c in corpora} == {2000, 6000}

    # and the checks do catch a broken corpus: remove one question's gold paragraph
    first = corpora[0]
    first["pids"].remove(by_qid[first["qids"][0]]["gold_pids"][0])
    assert any("gold paragraph" in problem for problem in check_corpora(corpora, by_qid))


def test_hard_to_estimate_text_still_fits():
    # The real failure: some questions' paragraphs take far more tokens than the average
    # ratio predicts (here every "Ж" is a whole token, like many non-English names).
    questions, paragraphs = fake_dataset()
    heavy = {p for q in questions[:20] for p in q["pids"]}
    for p in paragraphs:
        if p["pid"] in heavy:
            p["text"] = "Ж" * len(p["text"])
    est = estimate_sizes(paragraphs, chars_per_token=2.0)

    def fake_count_tokens(text):
        return round(text.count("Ж") + (len(text) - text.count("Ж")) / 1.8)

    # at 1,500 tokens the groups are packed close to full, as they were in the real failure
    corpora = build_corpora("fake", 0, questions, {p["pid"]: p for p in paragraphs}, est,
                            fake_count_tokens, sizes=(1500, 6000), n_questions=30)
    assert check_corpora(corpora, {q["qid"]: q for q in questions}) == []


def test_pick_questions_takes_the_first_n_corpus_by_corpus():
    corpora = [{"corpus_id": "a", "target_tokens": 16000, "qids": ["q1", "q2", "q3"]},
               {"corpus_id": "b", "target_tokens": 64000, "qids": ["q9"]},
               {"corpus_id": "c", "target_tokens": 16000, "qids": ["q4", "q5"]},
               {"corpus_id": "d", "target_tokens": 16000, "qids": ["q6"]}]
    picked = pick_questions(corpora, 16000, n=4)
    assert [(c["corpus_id"], q) for c, q in picked] == [("a", ["q1", "q2", "q3"]), ("c", ["q4"])]
    assert len(pick_questions(corpora, 16000)) == 3                      # n=None: every corpus of that size