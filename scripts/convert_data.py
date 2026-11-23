"""Step 4b-1: convert both datasets to the common format described in lcvr/data.py.

Run from the repo folder:  python -m scripts.convert_data
Reads data/raw/, writes data/processed/. Free: no API calls.
"""
from collections import Counter
from pathlib import Path

from lcvr.data import (KEEP_2WIKI_TYPES, convert_2wiki, convert_musique, read_2wiki_aliases,
                       read_json, read_jsonl, write_jsonl)

RAW = Path("data/raw")
OUT = Path("data/processed")


def report(name, questions, paragraphs, skipped):
    hops = Counter(q["hops"] for q in questions)
    with_aliases = sum(len(q["answers"]) > 1 for q in questions)
    print(f"{name}: {len(questions)} questions, {len(paragraphs)} unique paragraphs")
    print(f"   hops (count: questions): {dict(sorted(hops.items()))}")
    print(f"   questions with answer aliases: {with_aliases}")
    print(f"   skipped: {len(skipped)}  {skipped[:3]}")


# ---------------- MuSiQue ----------------
questions, paragraphs, skipped = convert_musique(read_jsonl(RAW / "musique_ans_v1.0_dev.jsonl"))
write_jsonl(OUT / "musique_questions.jsonl", questions)
write_jsonl(OUT / "musique_paragraphs.jsonl", paragraphs)
report("musique", questions, paragraphs, skipped)

# ---------------- 2WikiMultihopQA ----------------
alias_file = RAW / "2wiki_id_aliases.json"
if alias_file.exists():
    aliases = read_2wiki_aliases(alias_file)
else:
    aliases = {}
    print(f"\nNOTE: {alias_file} not found, so 2Wiki answers have no aliases yet.")

raw_2wiki = read_json(RAW / "2wiki_dev.json")
questions, paragraphs, skipped = convert_2wiki(raw_2wiki, aliases)
write_jsonl(OUT / "2wiki_questions.jsonl", questions)
write_jsonl(OUT / "2wiki_paragraphs.jsonl", paragraphs)
report("2wiki", questions, paragraphs, skipped)

# How common are repeated titles? (Most should be the same paragraph listed twice, which is harmless.)
repeats = sum(len({title for title, _ in r["context"]}) < len(r["context"])
              for r in raw_2wiki if r["type"] in KEEP_2WIKI_TYPES)
print(f"   questions whose paragraph list repeats a title: {repeats}")
