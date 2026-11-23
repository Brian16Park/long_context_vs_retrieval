"""Step 4b-2: measure how many characters make one token in each dataset's paragraphs.

The corpus builder (next step) uses this ratio to ESTIMATE sizes while choosing
paragraphs, then MEASURES every finished corpus exactly. Token counting is free.
Run from the repo folder:  python -m scripts.calibrate_tokens
"""
import random
from statistics import mean

from lcvr.corpus import render_corpus, render_paragraph
from lcvr.data import read_jsonl
from lcvr.llm import count_tokens

SAMPLE = 400        # paragraphs measured per dataset
BATCH = 100         # paragraphs per count_tokens call (roughly 10K tokens)

for name in ("musique", "2wiki"):
    paragraphs = read_jsonl(f"data/processed/{name}_paragraphs.jsonl")
    sample = random.Random(0).sample(paragraphs, SAMPLE)

    chars = 0
    tokens = 0
    for start in range(0, SAMPLE, BATCH):
        text = render_corpus(sample[start:start + BATCH])
        chars += len(text)
        tokens += count_tokens(text)
    ratio = chars / tokens

    # What one question brings to a corpus: its whole paragraph set (gold + distractors)
    chars_by_pid = {p["pid"]: len(render_paragraph(p)) for p in paragraphs}
    questions = read_jsonl(f"data/processed/{name}_questions.jsonl")
    per_question = mean(sum(chars_by_pid[pid] for pid in q["pids"]) for q in questions) / ratio

    print(f"{name}: {ratio:.3f} characters per token ({tokens / SAMPLE:.0f} tokens per paragraph on average)")
    print(f"   one question's paragraph set: about {per_question:,.0f} tokens")