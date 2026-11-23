"""Checks for the dataset converter (lcvr/data.py), using tiny made-up records."""
import pytest

from lcvr.data import convert_2wiki, convert_musique, paragraph_id


def musique_record(qid, paragraphs, support, answer="Paris", aliases=()):
    """paragraphs: list of (title, text); support: which paragraph answers each hop."""
    return {
        "id": qid,
        "paragraphs": [{"idx": i, "title": t, "paragraph_text": x, "is_supporting": i in support}
                       for i, (t, x) in enumerate(paragraphs)],
        "question": "Where is it?",
        "question_decomposition": [{"id": n, "question": "...", "answer": "...", "paragraph_support_idx": i}
                                   for n, i in enumerate(support)],
        "answer": answer,
        "answer_aliases": list(aliases),
        "answerable": True,
    }


def wiki_record(qid, qtype, answer_id="Q1", extra_context=()):
    return {
        "_id": qid,
        "type": qtype,
        "question": "Who is the mother of the director of film X?",
        "context": [["Film X", ["Film X is a film.", " It was directed by Y."]],   # note the leading space
                    ["Y", ["Y is a director.", "Y's mother is Z."]],
                    ["Other", ["An unrelated paragraph."]]] + list(extra_context),
        "supporting_facts": [["Film X", 1], ["Y", 0], ["Y", 1]],
        "answer": "Z",
        "answer_id": answer_id,
    }


def test_musique_gold_paragraphs_in_hop_order():
    # hop 1 is answered by paragraph 2, hop 2 by paragraph 0
    record = musique_record("2hop__1_2", [("A", "a text"), ("B", "b text"), ("C", "c text")], support=[2, 0])
    q = convert_musique([record])[0][0]
    assert q["gold_pids"] == [paragraph_id("mq", "C", "c text"), paragraph_id("mq", "A", "a text")]
    assert q["hops"] == 2
    assert q["type"] == "2hop"
    assert len(q["pids"]) == 3


def test_shared_paragraph_is_stored_once():
    first = musique_record("2hop__1_2", [("A", "a text"), ("B", "b text")], support=[0, 1])
    second = musique_record("2hop__3_4", [("B", "b text"), ("D", "d text")], support=[0, 1])
    _, paragraphs, _ = convert_musique([first, second])
    assert len(paragraphs) == 3                       # A, B, D: B is shared


def test_musique_answer_first_and_no_repeats():
    record = musique_record("2hop__1_2", [("A", "a"), ("B", "b")], support=[0, 1],
                            answer="NYC", aliases=["New York City", "NYC"])
    q = convert_musique([record])[0][0]
    assert q["answers"] == ["NYC", "New York City"]


def test_musique_mismatched_flags_are_caught():
    record = musique_record("2hop__1_2", [("A", "a"), ("B", "b"), ("C", "c")], support=[0, 1])
    record["paragraphs"][2]["is_supporting"] = True   # flagged as gold, but no hop uses it
    with pytest.raises(ValueError):
        convert_musique([record])


def test_2wiki_keeps_only_fact_chain_types():
    records = [wiki_record("a", "compositional"), wiki_record("b", "comparison"),
               wiki_record("c", "inference"), wiki_record("d", "bridge_comparison")]
    questions, _, _ = convert_2wiki(records)
    assert [q["qid"] for q in questions] == ["a", "c"]


def test_2wiki_gold_paragraphs_and_aliases():
    questions, paragraphs, _ = convert_2wiki([wiki_record("a", "compositional")], aliases={"Q1": ["Zed", "Z"]})
    q = questions[0]
    assert q["gold_pids"] == [paragraph_id("2w", "Film X", "Film X is a film. It was directed by Y."),
                              paragraph_id("2w", "Y", "Y is a director. Y's mother is Z.")]
    assert q["hops"] == 2
    assert q["answers"] == ["Z", "Zed"]
    assert len(paragraphs) == 3


def test_2wiki_same_paragraph_listed_twice_is_fine():
    record = wiki_record("a", "compositional", extra_context=[["Y", ["Y is a director.", "Y's mother is Z."]]])
    questions, paragraphs, skipped = convert_2wiki([record])
    assert skipped == []
    assert len(questions[0]["pids"]) == 3             # the repeat is one paragraph, stored once
    assert len(paragraphs) == 3


def test_2wiki_repeated_distractor_title_is_fine():
    record = wiki_record("a", "compositional", extra_context=[["Other", ["A different unrelated paragraph."]]])
    questions, _, skipped = convert_2wiki([record])
    assert skipped == []
    assert len(questions[0]["pids"]) == 4             # two different "Other" paragraphs, both kept


def test_2wiki_ambiguous_gold_title_is_skipped():
    # two DIFFERENT paragraphs titled "Y", and "Y" is gold: we can't tell which one is meant
    record = wiki_record("a", "compositional", extra_context=[["Y", ["A different Y entirely."]]])
    questions, paragraphs, skipped = convert_2wiki([record])
    assert questions == []
    assert skipped == ["a"]
    assert paragraphs == []                           # nothing stored for a skipped question
