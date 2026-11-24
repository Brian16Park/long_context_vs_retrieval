"""Scoring answers: exact match (EM) and token F1, the way MuSiQue and 2Wiki score them.

Both official evaluators use the SQuAD rules: normalize the prediction and the gold
answer (lowercase, drop punctuation and the articles a/an/the, collapse spaces), then
compare. A question gets its best score over all its accepted answers.
"""
import re
import string
from collections import Counter


def normalize(text):
    """'The Eiffel Tower.' -> 'eiffel tower'"""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return float(normalize(prediction) == normalize(gold))


def f1(prediction, gold):
    """Overlap of the two answers' words: 2 x precision x recall / (precision + recall)."""
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())   # words in common
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# "Answer:" anywhere in a line, also "**Answer:**" or "Final answer:", but not "Answering"
ANSWER_LABEL = re.compile(r"\banswer\W*?:", re.IGNORECASE)


def clean_prediction(reply):
    """The model's reply -> just the answer.

    Reasoning replies end with a line "Answer: ...", so if the reply has an "Answer:"
    label, the answer is what follows the last one (or the next line, if the label stands
    alone). Answer-only replies are the answer itself: their first non-empty line.
    """
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    for i in reversed(range(len(lines))):
        labels = list(ANSWER_LABEL.finditer(lines[i]))
        if labels:
            answer = re.sub(r"^\W+", "", lines[i][labels[-1].end():]).strip()
            if not answer and i + 1 < len(lines):     # "Answer:" alone, the answer below it
                answer = lines[i + 1]
            return answer
    return lines[0] if lines else ""


def has_answer_label(reply):
    """Does the reply contain an "Answer:" label? Reasoning replies without one were
    usually cut off by the token cap, and get scored from their first line."""
    return ANSWER_LABEL.search(reply) is not None


def score(reply, answers):
    """(EM, F1) of a model reply: the best over all accepted answers."""
    prediction = clean_prediction(reply)
    return (max(exact_match(prediction, a) for a in answers),
            max(f1(prediction, a) for a in answers))
