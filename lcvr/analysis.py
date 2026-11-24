"""Results of a run: accuracy with confidence intervals, cost per question across query volume Q,
long context vs. RAG at matched cost, and RAG's errors split into retrieval misses and
reasoning failures.

Everything here works on records already in memory (the call log, data/processed): no API
calls. scripts/analyze.py loads the files and writes the tables and figures.

Terms used throughout:
  setting   (dataset, system, corpus size, k, reranked): one row of results
  instance  (dataset, seed, question ID): one question in one seed's corpora. Every setting
            of a dataset and size answers the same instances, so settings compare one to one.
  Q         questions served per corpus while it stays cached (for RAG: per index)
"""
from collections import defaultdict
from statistics import mean, median

import numpy as np

from lcvr.cost import call_cost
from lcvr.prices import GPU, HAIKU_5_5
from lcvr.scoring import score

Q_VALUES = (1, 10, 100, 1000)
N_BOOT = 2000                    # bootstrap resamples for each 95% interval
OUTCOMES = ("correct", "reasoning failure", "partial retrieval", "retrieval miss")


# ---------------- loading ----------------

def load_calls(records, run):
    """The run's successful calls, one per label key (a call made twice: the later one counts)."""
    latest = {}
    for record in records:
        if record.get("ok") and record.get("run") == run:
            latest[record["key"]] = record
    return list(latest.values())


def score_calls(calls, questions_by_qid):
    """Add each call's exact match and F1, best over the question's accepted answers."""
    for call in calls:
        call["em"], call["f1"] = score(call["answer"], questions_by_qid[call["question_id"]]["answers"])
    return calls


def setting_of(call):
    return (call["dataset"], call["system"], call.get("corpus_tokens", 0),
            call.get("k", 0), call.get("reranked", False))


def instance_of(call):
    return call["dataset"], call["seed"], call["question_id"]


def by_setting(calls):
    """{setting: {instance: call}}"""
    groups = defaultdict(dict)
    for call in calls:
        groups[setting_of(call)][instance_of(call)] = call
    return groups


# ---------------- confidence intervals ----------------

def bootstrap_ci(values, strata, n_boot=N_BOOT, seed=0):
    """95% interval for the mean of `values`, resampling with replacement within each stratum
    (each seed's questions), so every resample keeps the design: 300 questions per seed."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    sums = np.zeros(n_boot)
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        draws = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        sums += values[rows][draws].sum(axis=1)
    low, high = np.percentile(sums / len(values), [2.5, 97.5])
    return float(low), float(high)


def paired_difference(a, b, metric="f1"):
    """Mean of metric(a) - metric(b) over the instances both settings answered, with its 95%
    interval. a and b are {instance: call}. Returns (difference, (low, high), n)."""
    shared = sorted(a.keys() & b.keys())
    diffs = [a[i][metric] - b[i][metric] for i in shared]
    return mean(diffs), bootstrap_ci(diffs, [i[1] for i in shared]), len(shared)


# ---------------- accuracy ----------------

def summarize(calls):
    """One row per setting: accuracy with 95% intervals, billed cost, latency, reply length."""
    rows = []
    for setting, group in sorted(by_setting(calls).items()):
        cs = list(group.values())
        strata = [c["seed"] for c in cs]
        em, f1 = [c["em"] for c in cs], [c["f1"] for c in cs]
        dataset, system, size, k, reranked = setting
        (em_low, em_high), (f1_low, f1_high) = bootstrap_ci(em, strata), bootstrap_ci(f1, strata)
        rows.append({
            "dataset": dataset, "system": system, "size": size, "k": k, "reranked": reranked,
            "n": len(cs),
            "em": mean(em), "em_low": em_low, "em_high": em_high,
            "f1": mean(f1), "f1_low": f1_low, "f1_high": f1_high,
            "f1_by_seed": {s: mean(c["f1"] for c in cs if c["seed"] == s) for s in sorted(set(strata))},
            "api_cost": mean(call_cost(c) for c in cs),
            "prompt_tokens": mean(c["input_tokens"] + c["cache_write_tokens"] + c["cache_read_tokens"] for c in cs),
            "cached_tokens": mean(c["cache_write_tokens"] + c["cache_read_tokens"] for c in cs),
            "input_tokens": mean(c["input_tokens"] for c in cs),
            "output_tokens": mean(c["output_tokens"] for c in cs),
            "latency_s": median(c["latency_s"] for c in cs),
            "cut_off": sum(c["stop_reason"] == "max_tokens" for c in cs),
        })
    return rows


# ---------------- cost per question across Q ----------------

def tier_prices(prompt_tokens, prices=HAIKU_5_5):
    return prices["long" if prompt_tokens > prices["long_prompt_threshold"] else "short"]


def lc_cost(cached_tokens, input_tokens, output_tokens, q, prices=HAIKU_5_5):
    """Long context's dollars per question when q questions share one cached corpus.

    The corpus (with the instructions before it) is written to the cache once, then read by
    the other q - 1 questions; each question also pays for its own uncached tokens and its
    reply. At q = 1 a cache write costs 25% more than sending the corpus uncached, so the
    cheaper of the two counts (a deployment with no reuse wouldn't cache).
    """
    p = tier_prices(cached_tokens + input_tokens, prices)
    per_question = input_tokens * p["input"] + output_tokens * p["output"]
    cached = cached_tokens * (p["cache_write_5m"] + (q - 1) * p["cache_read"]) / q
    uncached = cached_tokens * p["input"]
    return (min(cached, uncached) + per_question) / 1_000_000


def gpu_dollars(seconds, gpu=GPU):
    return seconds * gpu["dollars_per_hour"] / 3600


def gpu_rates(embed_timing, rank_timings):
    """Seconds of GPU work per unit, from the timing records of scripts/embed.py and rank.py."""
    pairs = sum(t["rerank_pairs"] for t in rank_timings)
    queries = sum(t["questions"] for t in rank_timings)
    return {
        "embed_per_paragraph": embed_timing["paragraph_wall_seconds"] / embed_timing["paragraphs"],
        "embed_per_question": embed_timing["question_wall_seconds"] / embed_timing["questions"],
        "search_per_query": sum(t["dense_seconds"] for t in rank_timings) / queries,
        "rerank_per_pair": sum(t["rerank_seconds"] for t in rank_timings) / pairs,
    }


def rag_cost(api_cost, reranked, corpus_paragraphs, q, rates, candidates=100, gpu=GPU):
    """RAG's dollars per question when q questions share one index: the API call as billed, plus
    GPU time to embed the question, search, rerank (if on), and to embed the corpus once,
    spread over q questions. Reranking scores the top `candidates` (fewer in a small corpus)."""
    seconds = rates["embed_per_question"] + rates["search_per_query"]
    if reranked:
        seconds += min(candidates, corpus_paragraphs) * rates["rerank_per_pair"]
    seconds += corpus_paragraphs * rates["embed_per_paragraph"] / q
    return api_cost + gpu_dollars(seconds, gpu)


# ---------------- long context vs. RAG at matched cost ----------------

def compare_at_q(lc_row, lc_calls, rag_rows, rag_calls, q, rag_cost_at):
    """The proposal's matching rule in one (dataset, size, Q) cell.

    Iso-cost: long context against the most accurate RAG setting that costs no more per
    question. The winner needs a 95% interval for the F1 difference that excludes zero;
    otherwise it's a tie. If no RAG setting is that cheap, RAG can't compete at that budget.
    Iso-accuracy: the cheapest RAG setting at least as accurate as long context, if any.
    `rag_cost_at(row, q)` prices a RAG row at q; lc_row["cost_at"][q] prices long context.
    """
    lc_dollars = lc_row["cost_at"][q]
    priced = [(rag_cost_at(row, q), row) for row in rag_rows]
    eligible = [(cost, row) for cost, row in priced if cost <= lc_dollars]
    cell = {"q": q, "lc_cost": lc_dollars, "lc_f1": lc_row["f1"]}
    if eligible:
        cost, best = max(eligible, key=lambda pair: (pair[1]["f1"], -pair[0]))
        diff, (low, high), n = paired_difference(lc_calls, rag_calls[(best["k"], best["reranked"])])
        winner = "LC" if low > 0 else "RAG" if high < 0 else "tie"
        cell.update(rag_k=best["k"], rag_reranked=best["reranked"], rag_cost=cost, rag_f1=best["f1"],
                    diff=diff, diff_low=low, diff_high=high, winner=winner)
    else:
        cell.update(rag_k=None, winner="LC (RAG can't be this cheap)")
    as_good = [(cost, row) for cost, row in priced if row["f1"] >= lc_row["f1"]]
    if as_good:
        cost, row = min(as_good, key=lambda pair: pair[0])
        cell.update(iso_k=row["k"], iso_reranked=row["reranked"], iso_cost=cost, iso_ratio=cost / lc_dollars)
    else:
        cell.update(iso_k=None)
    return cell


# ---------------- RAG errors: retrieval miss or reasoning failure ----------------

def retrieval_outcome(call, ranking, gold_pids):
    """What happened in one RAG call: 'correct' (exact match), or for a wrong answer, why.

    reasoning failure : every gold paragraph was among the k sent, and the answer was still wrong
    partial retrieval : some gold paragraphs were sent, others missed (a multi-hop failure mode)
    retrieval miss    : none of the gold paragraphs was sent
    Also returns how many of the gold paragraphs were sent, and how many there are.
    """
    order = ranking["reranked"] if call["reranked"] else ranking["dense"]
    gold = set(gold_pids)
    found = len(gold & set(order[:call["k"]]))
    if call["em"] == 1:
        outcome = "correct"
    elif found == len(gold):
        outcome = "reasoning failure"
    elif found:
        outcome = "partial retrieval"
    else:
        outcome = "retrieval miss"
    return outcome, found, len(gold)


def hop_recall(ranking, gold_pids, k, reranked):
    """For each hop's gold paragraph, in hop order: was it among the first k? (MuSiQue lists
    gold paragraphs in hop order.)"""
    top = set((ranking["reranked"] if reranked else ranking["dense"])[:k])
    return [pid in top for pid in gold_pids]


# ---------------- long context: where the gold paragraphs sit ----------------

def gold_positions(corpus_pids, gold_pids):
    """Relative position (0 = start of the corpus, 1 = end) of each gold paragraph."""
    where = {pid: i for i, pid in enumerate(corpus_pids)}
    last = max(len(corpus_pids) - 1, 1)
    return [where[pid] / last for pid in gold_pids if pid in where]
