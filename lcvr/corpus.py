"""Corpora: groups of paragraphs that many questions share, and how they look in a prompt.

How corpora are built, for each dataset and seed:
  1. Sample 300 questions. The same 300 are used at every corpus size. Measure each
     one's paragraph set exactly, so the groups below are sized with real counts.
  2. At each size, split them into even groups whose paragraph sets (gold + their
     own distractors) fill about 90% of the size.
  3. Each group becomes one corpus: its questions' paragraphs, topped up with
     "filler" paragraphs from questions outside the sample, then shuffled.
  4. Every corpus is measured with the exact token counter, and the filler is
     adjusted until the corpus is within 1% of its size.
"""
import math
import random
from concurrent.futures import ThreadPoolExecutor

SIZES = (16_000, 64_000, 256_000, 512_000)    # corpus sizes N, in tokens
SEEDS = (0, 1, 2)
QUESTIONS_PER_SEED = 300
FILL = 0.9          # a group's own paragraphs fill about 90% of its corpus; filler does the rest
MAX_FILL = 0.97     # ...and never more than 97%, leaving room to adjust to the exact size
TOLERANCE = 0.01    # every corpus must measure within 1% of its size

# Measured 2026-11-22 with scripts/calibrate_tokens.py (rendered paragraphs, Haiku 5.5 tokenizer)
CHARS_PER_TOKEN = {"musique": 2.781, "2wiki": 2.556}


# ---------------- how paragraphs look in a prompt ----------------

def render_paragraph(paragraph):
    """How one paragraph appears in EVERY prompt, long-context and RAG alike: a title line, then the text."""
    return f"# {paragraph['title']}\n{paragraph['text']}"


def render_corpus(paragraphs):
    """Paragraphs in order, separated by a blank line."""
    return "\n\n".join(render_paragraph(p) for p in paragraphs)


def estimate_sizes(paragraphs, chars_per_token):
    """Estimated tokens of each paragraph in a corpus (its blank-line separator included)."""
    return {p["pid"]: (len(render_paragraph(p)) + 2) / chars_per_token for p in paragraphs}


def rescale_to_exact(est, questions, exact_tokens):
    """New estimates in which each question's paragraphs add up to its exact measured count.

    The average characters-per-token ratio is off for some questions (numbers and
    non-English names take more tokens), so we measure each sampled question's
    paragraph set once and scale its paragraphs' estimates to match.
    """
    new = dict(est)
    for q, tokens in zip(questions, exact_tokens):
        factor = tokens / sum(est[p] for p in q["pids"])
        for p in q["pids"]:
            new[p] = est[p] * factor
    return new


# ---------------- step 2: groups of questions ----------------

class Group:
    """Questions that will share one corpus, and the paragraphs they bring."""

    def __init__(self):
        self.qids = []
        self.pids = []           # in the order they were added
        self.pid_set = set()     # the same paragraphs, for fast "already have it?" checks
        self.tokens = 0.0        # estimated

    def extra_tokens(self, question, est):
        """Tokens this question would add: only the paragraphs the group doesn't have yet."""
        return sum(est[p] for p in question["pids"] if p not in self.pid_set)

    def add(self, question, est):
        self.tokens += self.extra_tokens(question, est)
        self.qids.append(question["qid"])
        for p in question["pids"]:
            if p not in self.pid_set:
                self.pids.append(p)
                self.pid_set.add(p)


def group_questions(questions, est, size):
    """Split questions into even groups whose paragraphs fit in a corpus of `size` tokens.

    Uses just enough groups to fill each about FILL full, then hands questions out
    biggest first, each to the emptiest group with room (at most MAX_FILL full).
    Big ones first means the small ones at the end can even out the groups.
    """
    def own_tokens(q):
        return sum(est[p] for p in q["pids"])

    total = sum(own_tokens(q) for q in questions)
    groups = [Group() for _ in range(math.ceil(total / (FILL * size)))]
    for q in sorted(questions, key=own_tokens, reverse=True):
        fits = [g for g in groups if g.tokens + g.extra_tokens(q, est) <= MAX_FILL * size]
        if not fits:                              # fits in no group: open a new one
            groups.append(Group())
            fits = [groups[-1]]
        min(fits, key=lambda g: g.tokens).add(q, est)
    return [g for g in groups if g.qids]


# ---------------- steps 3-4: filler, shuffle, exact size ----------------

def pick_filler(pool, used, est, budget, ratio):
    """Paragraphs from `pool`, in order and not in `used`, whose sizes (ratio x estimate)
    fill as much of `budget` tokens as possible without going over."""
    picked = []
    for pid in pool:
        size = ratio * est[pid]
        if size <= budget and pid not in used:
            picked.append(pid)
            budget -= size
    return picked


def fit_corpus(core, pool, est, target, measure, shuffle_seed, max_rounds=8):
    """Core paragraphs + filler, shuffled and measured exactly.

    measure(pids) gives the exact tokens of those paragraphs as a corpus. The core is
    measured once, so whatever a full corpus adds beyond it is the filler's real size.
    Each round then drops the most recently added filler if the corpus is too big, or
    adds new filler if it's too small. A dropped paragraph is never picked again, so
    the rounds can't swing back and forth.
    Returns (pids in prompt order, measured tokens).
    """
    core_tokens = measure(core)
    ratio = 1.0                                         # real tokens per estimated token, for filler
    dropped = set()
    filler = pick_filler(pool, set(core), est, target - core_tokens, ratio)
    for _ in range(max_rounds):
        order = core + filler
        random.Random(shuffle_seed).shuffle(order)     # same seed + same paragraphs -> same order
        tokens = measure(order)
        gap = target - tokens                           # above 0: too small; below 0: too big
        if abs(gap) <= TOLERANCE * target:
            break
        filler_tokens = tokens - core_tokens            # the filler's real size
        if filler and filler_tokens > 0:
            ratio = filler_tokens / sum(est[p] for p in filler)
        if gap < 0:                                     # too big: drop the latest filler
            if not filler:
                break                                   # the core alone is too big; nothing to drop
            while filler and gap < 0:
                pid = filler.pop()
                dropped.add(pid)
                gap += ratio * est[pid]
        else:                                           # too small: add new filler that fits the gap
            more = pick_filler(pool, set(core) | set(filler) | dropped, est, gap, ratio)
            if not more:
                break                                   # nothing left that fits
            filler += more
    return order, tokens


def build_corpora(dataset, seed, questions, paragraphs_by_pid, est, count_tokens,
                  sizes=SIZES, n_questions=QUESTIONS_PER_SEED, workers=8):
    """All corpora for one dataset and seed (the design is at the top of this file).

    Token counts run `workers` at a time: each one mostly waits on the network,
    so sending several at once makes the build several times faster.
    """
    sample = random.Random(f"{dataset}-{seed}").sample(questions, n_questions)
    in_sample = {p for q in sample for p in q["pids"]}
    filler = sorted(p for p in paragraphs_by_pid if p not in in_sample)

    def measure(pids):
        return count_tokens(render_corpus(paragraphs_by_pid[p] for p in pids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Step 1: the exact size of each sampled question's paragraph set
        print(f"{dataset} seed {seed}: measuring {len(sample)} questions", flush=True)
        exact = list(executor.map(lambda q: measure(q["pids"]), sample))
        est = rescale_to_exact(est, sample, exact)

        corpora = []
        for size in sizes:
            groups = group_questions(sample, est, size)
            ids = [f"{dataset}-s{seed}-{size // 1000:03d}k-{i:03d}" for i in range(len(groups))]
            print(f"{dataset} seed {seed}, {size:,} tokens: {len(groups)} corpora", flush=True)

            def fit(group, corpus_id):
                order = filler[:]
                random.Random(corpus_id).shuffle(order)      # each corpus gets its own random filler
                return fit_corpus(group.pids, order, est, size, measure, corpus_id)

            for group, corpus_id, (pids, tokens) in zip(groups, ids, executor.map(fit, groups, ids)):
                corpora.append({
                    "corpus_id": corpus_id,
                    "dataset": dataset,
                    "seed": seed,
                    "target_tokens": size,
                    "tokens": tokens,                       # measured with the exact token counter
                    "qids": group.qids,
                    "n_filler": len(pids) - len(group.pids),
                    "pids": pids,                           # in prompt order
                })
    return corpora


def pick_questions(corpora, size, n=None):
    """The first n questions at one corpus size (all if n is None), corpus by corpus in file
    order, as [(corpus, its chosen question IDs)]. Every system picks questions with this,
    so they all answer the same questions in the same corpora."""
    picked = []
    for corpus in corpora:
        if corpus["target_tokens"] != size:
            continue
        qids = corpus["qids"] if n is None else corpus["qids"][:n - sum(len(q) for _, q in picked)]
        if not qids:
            break
        picked.append((corpus, qids))
    return picked


# ---------------- checks ----------------

def check_corpora(corpora, questions_by_qid):
    """A list of problems; an empty list means every check passed."""
    problems = []
    qids_by_seed_size = {}
    for c in corpora:
        name = c["corpus_id"]
        pids = set(c["pids"])
        if len(pids) != len(c["pids"]):
            problems.append(f"{name}: a paragraph appears twice")
        if abs(c["tokens"] - c["target_tokens"]) > TOLERANCE * c["target_tokens"]:
            problems.append(f"{name}: measured {c['tokens']:,} tokens, target {c['target_tokens']:,}")
        for qid in c["qids"]:
            if not set(questions_by_qid[qid]["gold_pids"]) <= pids:
                problems.append(f"{name}: a gold paragraph of {qid} is missing")
        qids_by_seed_size.setdefault((c["seed"], c["target_tokens"]), []).extend(c["qids"])

    for (seed, size), qids in qids_by_seed_size.items():
        if len(qids) != len(set(qids)):
            problems.append(f"seed {seed}, {size:,} tokens: a question is in two corpora")
    for seed in {seed for seed, _ in qids_by_seed_size}:
        question_sets = {frozenset(q) for (s, _), q in qids_by_seed_size.items() if s == seed}
        if len(question_sets) != 1:
            problems.append(f"seed {seed}: the sizes don't all have the same questions")
    return problems