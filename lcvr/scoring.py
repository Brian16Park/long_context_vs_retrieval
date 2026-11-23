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


def clean_prediction(reply):
    """The model's reply -> just the answer: its first non-empty line, without an 'Answer:' label.

    The prompt asks for the answer alone; this only catches the common slips.
    """
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return ""
    return re.sub(r"^\W*answer\W*:\W*", "", lines[0], flags=re.IGNORECASE).strip()


def score(reply, answers):
    """(EM, F1) of a model reply: the best over all accepted answers."""
    prediction = clean_prediction(reply)
    return (max(exact_match(prediction, a) for a in answers),
            max(f1(prediction, a) for a in answers))