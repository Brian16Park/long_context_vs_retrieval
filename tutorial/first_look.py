"""Print the first example of each dataset, with long text shortened."""
import json


def shorten(value, max_chars=80, max_items=3):
    """Return a copy of `value` that is small enough to read on screen."""
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + "..."
        return value
    if isinstance(value, list):
        short_list = [shorten(item) for item in value[:max_items]]
        if len(value) > max_items:
            short_list.append(f"... ({len(value) - max_items} more)")
        return short_list
    if isinstance(value, dict):
        return {key: shorten(item) for key, item in value.items()}
    return value                               # numbers, True/False, None: keep as they are


# MuSiQue is JSON Lines: one question per line. Read just the first line.
with open("data/raw/musique_ans_v1.0_dev.jsonl", encoding="utf-8") as f:
    musique_first = json.loads(f.readline())

print("=== MuSiQue: first example ===")
print(json.dumps(shorten(musique_first), indent=2))

# 2WikiMultihopQA is ONE big JSON list: we have to load the whole file.
with open("data/raw/2wiki_dev.json", encoding="utf-8") as f:
    wiki_all = json.load(f)

print("\n=== 2WikiMultihopQA: first example ===")
print(json.dumps(shorten(wiki_all[0]), indent=2))