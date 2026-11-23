"""Convert MuSiQue and 2WikiMultihopQA into one common format.

Each dataset becomes two files in data/processed/:

  <name>_questions.jsonl   one question per line:
      qid        the dataset's own question ID
      dataset    "musique" or "2wiki"
      type       2hop, 3hop1, ... (MuSiQue) or compositional, inference (2Wiki)
      hops       number of reasoning steps
      question   the question text
      answers    accepted answers: the gold answer first, then its aliases
      gold_pids  the gold paragraph of each hop, in hop order
      pids       every paragraph that comes with the question (gold + distractors)

  <name>_paragraphs.jsonl  one paragraph per line: pid, title, text

A paragraph shared by several questions is stored once, under one ID.
"""
import hashlib
import json
from pathlib import Path

KEEP_2WIKI_TYPES = ("compositional", "inference")   # fact chains; the "X or Y?" types are excluded


# ---------------- reading and writing ----------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # create data/processed/ if it's missing
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------- paragraph IDs ----------------

def paragraph_id(prefix, title, text):
    """An ID made from the paragraph itself: same title and text -> same ID, on any machine."""
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def store_paragraph(paragraphs, prefix, title, text):
    """Add a paragraph to `paragraphs` (a dict: pid -> paragraph) once, and return its ID."""
    pid = paragraph_id(prefix, title, text)
    paragraph = {"pid": pid, "title": title, "text": text}
    if pid in paragraphs and paragraphs[pid] != paragraph:
        raise ValueError(f"two different paragraphs got the same ID {pid}")
    paragraphs[pid] = paragraph
    return pid


def unique(items):
    """Remove repeats, keep the order."""
    return list(dict.fromkeys(items))


# ---------------- MuSiQue ----------------

def convert_musique(raw_questions):
    """MuSiQue records -> (questions, paragraphs, skipped) in the common format.

    skipped: IDs of questions left out (unanswerable ones; the Ans dev split has none).
    """
    questions = []
    paragraphs = {}
    skipped = []
    for raw in raw_questions:
        if not raw.get("answerable", True):
            skipped.append(raw["id"])
            continue
        pid_by_idx = {}
        flagged = set()                              # paragraphs the dataset marks as gold
        for position, p in enumerate(raw["paragraphs"]):
            pid = store_paragraph(paragraphs, "mq", p["title"], p["paragraph_text"])
            pid_by_idx[p.get("idx", position)] = pid
            if p["is_supporting"]:
                flagged.add(pid)

        # Each hop names the paragraph that answers it
        gold_pids = [pid_by_idx[hop["paragraph_support_idx"]] for hop in raw["question_decomposition"]]

        # Cross-check: the hops must point at exactly the flagged paragraphs
        if set(gold_pids) != flagged:
            raise ValueError(f"{raw['id']}: hop paragraphs don't match the is_supporting flags")

        questions.append({
            "qid": raw["id"],
            "dataset": "musique",
            "type": raw["id"].split("__")[0],          # e.g. "2hop", "3hop1", "4hop3"
            "hops": len(raw["question_decomposition"]),
            "question": raw["question"],
            "answers": unique([raw["answer"]] + raw.get("answer_aliases", [])),
            "gold_pids": gold_pids,
            "pids": unique(pid_by_idx.values()),
        })
    return questions, list(paragraphs.values()), skipped


# ---------------- 2WikiMultihopQA ----------------

def read_2wiki_aliases(path):
    """id_aliases.json (one JSON object per line) -> {Wikidata ID: other accepted names}.

    Same rule as the official 2Wiki evaluator: aliases plus demonyms.
    """
    return {r["Q_id"]: r["aliases"] + r["demonyms"] for r in read_jsonl(path)}


def gold_pids_by_title(context, gold_titles):
    """The paragraph ID for each gold title.

    context: list of (title, text, pid). Returns None if a gold title matches
    no paragraph, or two DIFFERENT paragraphs (then we can't tell which is gold).
    The same paragraph repeated twice is fine: both copies have the same ID.
    """
    gold_pids = []
    for gold_title in gold_titles:
        matches = unique(pid for title, _, pid in context if title == gold_title)
        if len(matches) != 1:
            return None
        gold_pids.append(matches[0])
    return gold_pids


def convert_2wiki(raw_questions, aliases=None):
    """2Wiki records -> (questions, paragraphs, skipped), compositional and inference questions only.

    skipped: IDs of questions whose gold paragraph is ambiguous (see gold_pids_by_title).
    """
    aliases = aliases or {}
    questions = []
    paragraphs = {}
    skipped = []
    for raw in raw_questions:
        if raw["type"] not in KEEP_2WIKI_TYPES:
            continue
        context = []                                 # (title, text, pid), in the dataset's order
        for title, sentences in raw["context"]:
            text = " ".join(s.strip() for s in sentences)    # list of sentences -> one paragraph
            context.append((title, text, paragraph_id("2w", title, text)))

        # Gold paragraphs are named by title in supporting_facts (one entry per sentence used)
        gold_titles = unique(title for title, _ in raw["supporting_facts"])
        gold_pids = gold_pids_by_title(context, gold_titles)
        if gold_pids is None:
            skipped.append(raw["_id"])
            continue                                 # store nothing for a skipped question

        for title, text, _ in context:
            store_paragraph(paragraphs, "2w", title, text)

        questions.append({
            "qid": raw["_id"],
            "dataset": "2wiki",
            "type": raw["type"],
            "hops": len(gold_pids),
            "question": raw["question"],
            "answers": unique([raw["answer"]] + aliases.get(raw.get("answer_id"), [])),
            "gold_pids": gold_pids,
            "pids": unique(pid for _, _, pid in context),
        })
    return questions, list(paragraphs.values()), skipped
