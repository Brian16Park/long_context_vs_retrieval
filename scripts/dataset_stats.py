"""Count what's in each dataset, so we can choose what to use."""
import json
from collections import Counter
from statistics import mean


def words(text):
    """Number of words in a piece of text (a rough size measure; exact tokens come later)."""
    return len(text.split())


# ---------------- MuSiQue ----------------
musique = []
with open("data/raw/musique_ans_v1.0_dev.jsonl", encoding="utf-8") as f:
    for line in f:
        musique.append(json.loads(line))

hops = Counter(len(q["question_decomposition"]) for q in musique)
paragraphs_per_q = [len(q["paragraphs"]) for q in musique]
gold_per_q = [sum(p["is_supporting"] for p in q["paragraphs"]) for q in musique]
paragraph_words = [words(p["paragraph_text"]) for q in musique for p in q["paragraphs"]]
unique_paragraphs = {(p["title"], p["paragraph_text"]) for q in musique for p in q["paragraphs"]}

print("=== MuSiQue-Ans dev ===")
print("questions:              ", len(musique))
print("answerable:             ", sum(q["answerable"] for q in musique))
print("hops (count: questions):", dict(sorted(hops.items())))
print("paragraphs per question:", min(paragraphs_per_q), "to", max(paragraphs_per_q))
print("gold per question:      ", dict(sorted(Counter(gold_per_q).items())))
print("words per paragraph:     avg", round(mean(paragraph_words)), " max", max(paragraph_words))
print("unique paragraphs:      ", len(unique_paragraphs),
      " (out of", len(paragraph_words), "total)")
print("words in unique paras:  ", sum(words(text) for _, text in unique_paragraphs))

# ---------------- 2WikiMultihopQA ----------------
with open("data/raw/2wiki_dev.json", encoding="utf-8") as f:
    wiki = json.load(f)

types = Counter(q["type"] for q in wiki)
gold_titles_per_q = [len({title for title, _ in q["supporting_facts"]}) for q in wiki]
wiki_paragraph_words = [words(" ".join(sentences)) for q in wiki for _, sentences in q["context"]]
unique_wiki = {(title, " ".join(sentences)) for q in wiki for title, sentences in q["context"]}

# Data check: is every gold title actually present among the question's paragraphs?
missing_gold = 0
for q in wiki:
    context_titles = {title for title, _ in q["context"]}
    for title, _ in q["supporting_facts"]:
        if title not in context_titles:
            missing_gold += 1

print("\n=== 2WikiMultihopQA dev ===")
print("questions:              ", len(wiki))
print("types:                  ", dict(types.most_common()))
print("paragraphs per question:", min(len(q["context"]) for q in wiki), "to",
      max(len(q["context"]) for q in wiki))
print("gold per question:      ", dict(sorted(Counter(gold_titles_per_q).items())))
print("words per paragraph:     avg", round(mean(wiki_paragraph_words)), " max", max(wiki_paragraph_words))
print("unique paragraphs:      ", len(unique_wiki), " (out of", len(wiki_paragraph_words), "total)")
print("words in unique paras:  ", sum(words(text) for _, text in unique_wiki))
print("gold titles missing from context:", missing_gold)