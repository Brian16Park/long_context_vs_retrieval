"""Analyze a finished run: accuracy, cost across query volume Q, long context vs. RAG at
matched cost, and RAG's errors split into retrieval misses and reasoning failures.

Run from the repo folder, for example:
  python -m scripts.analyze --run full

Reads logs/calls.jsonl and data/processed/ (questions, corpora, rankings and timing files)
and writes results/: summary.md (every table), CSV files, and figures/ (needs matplotlib:
pip install matplotlib). Makes no API calls.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from lcvr.analysis import (OUTCOMES, Q_VALUES, by_setting, compare_at_q, gold_positions, gpu_rates, hop_recall,
                           lc_cost, load_calls, rag_cost, retrieval_outcome, score_calls, summarize, tier_prices)
from lcvr.corpus import SIZES
from lcvr.cost import call_cost
from lcvr.data import read_jsonl
from lcvr.llm import read_log
from lcvr.prices import GPU, PRICE_SHEET_DATE

parser = argparse.ArgumentParser(description="Tables and figures for a finished run.")
parser.add_argument("--run", default="full", help="run name (default: full)")
parser.add_argument("--log", default="logs/calls.jsonl", help="call log")
parser.add_argument("--data", default="data/processed", help="questions, corpora, rankings, timing files")
parser.add_argument("--out", default="results", help="output folder")
args = parser.parse_args()

DATA, OUT = Path(args.data), Path(args.out)
DATASETS = ("musique", "2wiki")
NAMES = {"musique": "MuSiQue", "2wiki": "2WikiMultihopQA"}


def size_label(size):
    return f"{size // 1000}K"


def setting_label(row):
    if row["system"] == "lc":
        return "Long context"
    if row["system"] == "closed_book":
        return "Closed book"
    return f"RAG top {row['k']}{', reranked' if row['reranked'] else ''}"


# ---------------- load ----------------
questions = {}
corpora, rankings, rates, paragraphs_per_corpus = {}, {}, {}, {}
for dataset in DATASETS:
    questions.update({q["qid"]: q for q in read_jsonl(DATA / f"{dataset}_questions.jsonl")})
    embed_timing = json.loads((DATA / "embeddings" / f"{dataset}_timing.json").read_text(encoding="utf-8"))
    rank_timings = []
    for seed in (0, 1, 2):
        corpora.update({c["corpus_id"]: c for c in read_jsonl(DATA / "corpora" / f"{dataset}_seed{seed}.jsonl")})
        rankings.update({(r["corpus_id"], r["qid"]): r
                         for r in read_jsonl(DATA / "rankings" / f"{dataset}_seed{seed}.jsonl")})
        rank_timings += read_jsonl(DATA / "rankings" / f"{dataset}_seed{seed}_timing.jsonl")
    rates[dataset] = gpu_rates(embed_timing, rank_timings)
    for size in SIZES:
        sized = [c for c in corpora.values() if c["dataset"] == dataset and c["target_tokens"] == size]
        paragraphs_per_corpus[(dataset, size)] = mean(len(c["pids"]) for c in sized)

records = read_log(args.log)
calls = score_calls(load_calls(records, args.run), questions)
failed = [r for r in records if r.get("run") == args.run and not r.get("ok")]
if not calls:
    raise SystemExit(f"No successful calls for run '{args.run}' in {args.log}.")
rows = summarize(calls)
groups = by_setting(calls)
print(f"{len(calls):,} calls in run '{args.run}', {len(rows)} settings")


def find(dataset, system, size=0, k=0, reranked=False):
    return next(r for r in rows if (r["dataset"], r["system"], r["size"], r["k"], r["reranked"])
                == (dataset, system, size, k, reranked))


# ---------------- cost per question across Q ----------------
def lc_cost_at(row, q):
    return lc_cost(row["cached_tokens"], row["input_tokens"], row["output_tokens"], q)


def rag_cost_at(row, q):
    return rag_cost(row["api_cost"], row["reranked"], paragraphs_per_corpus[(row["dataset"], row["size"])],
                    q, rates[row["dataset"]])


for row in rows:
    if row["system"] == "lc":
        row["cost_at"] = {q: lc_cost_at(row, q) for q in Q_VALUES}
    elif row["system"] == "rag":
        row["cost_at"] = {q: rag_cost_at(row, q) for q in Q_VALUES}

# Cost model check: one cache write per corpus, then reads, as the model assumes
writes = Counter(c["corpus_id"] for c in calls if c["system"] == "lc" and c["cache_write_tokens"] > 0)
lc_corpora = {c["corpus_id"] for c in calls if c["system"] == "lc"}
extra_writes = {cid: n - 1 for cid, n in writes.items() if n > 1}
extra_write_cost, written = 0.0, set()     # what each write after a corpus's first cost beyond a read
for c in sorted((c for c in calls if c["system"] == "lc" and c["cache_write_tokens"] > 0), key=lambda c: c["time"]):
    if c["corpus_id"] in written:
        p = tier_prices(c["cache_write_tokens"] + c["input_tokens"])
        extra_write_cost += c["cache_write_tokens"] * (p["cache_write_5m"] - p["cache_read"]) / 1_000_000
    written.add(c["corpus_id"])

# ---------------- matched cost ----------------
cells = []
for dataset in DATASETS:
    for size in SIZES:
        lc = find(dataset, "lc", size)
        rag_rows = [r for r in rows if r["dataset"] == dataset and r["system"] == "rag" and r["size"] == size]
        rag_calls = {(r["k"], r["reranked"]): groups[(dataset, "rag", size, r["k"], r["reranked"])] for r in rag_rows}
        for q in Q_VALUES:
            cell = compare_at_q(lc, groups[(dataset, "lc", size, 0, False)], rag_rows, rag_calls, q, rag_cost_at)
            cells.append({"dataset": dataset, "size": size, **cell})

# ---------------- RAG outcomes ----------------
outcome_rows = []
for (dataset, system, size, k, reranked), group in sorted(groups.items()):
    if system != "rag":
        continue
    counts = Counter()
    for c in group.values():
        outcome, found, total = retrieval_outcome(c, rankings[(c["corpus_id"], c["question_id"])],
                                                  questions[c["question_id"]]["gold_pids"])
        counts[outcome] += 1
        counts["all gold sent"] += found == total
        counts["correct without all gold"] += outcome == "correct" and found < total
    n = len(group)
    outcome_rows.append({"dataset": dataset, "size": size, "k": k, "reranked": reranked, "n": n,
                         **{name: counts[name] / n for name in OUTCOMES},
                         "all gold sent": counts["all gold sent"] / n,
                         "correct without all gold": counts["correct without all gold"] / n})

# ---------------- multi-hop detail ----------------
hop_rows = []          # MuSiQue F1 by hop count
for size in SIZES:
    for hops in (2, 3, 4):
        entry = {"size": size, "hops": hops}
        for name, key in (("lc", ("musique", "lc", size, 0, False)),
                          ("rag_top80_reranked", ("musique", "rag", size, 80, True)),
                          ("rag_top20_reranked", ("musique", "rag", size, 20, True))):
            picked = [c for c in groups[key].values() if questions[c["question_id"]]["hops"] == hops]
            entry[name], entry["n"] = mean(c["f1"] for c in picked), len(picked)
        hop_rows.append(entry)

hop_recall_rows = []   # MuSiQue: is each hop's gold paragraph in the top 20?
for size in SIZES:
    for reranked in (False, True):
        hits = defaultdict(list)
        for c in groups[("musique", "rag", size, 20, reranked)].values():
            ranking = rankings[(c["corpus_id"], c["question_id"])]
            for hop, found in enumerate(hop_recall(ranking, questions[c["question_id"]]["gold_pids"], 20, reranked), 1):
                hits[hop].append(found)
        hop_recall_rows.append({"size": size, "reranked": reranked,
                                **{f"hop {h}": mean(v) for h, v in sorted(hits.items())},
                                **{f"n hop {h}": len(v) for h, v in sorted(hits.items())}})

overlap_rows = []      # which questions each system gets right (exact match)
for dataset in DATASETS:
    for size in SIZES:
        a, b = groups[(dataset, "lc", size, 0, False)], groups[(dataset, "rag", size, 80, True)]
        t = Counter((a[i]["em"] == 1, b[i]["em"] == 1) for i in a)
        n = len(a)
        overlap_rows.append({"dataset": dataset, "size": size, "both": t[(True, True)] / n,
                             "only long context": t[(True, False)] / n, "only RAG": t[(False, True)] / n,
                             "neither": t[(False, False)] / n})

position_rows = []     # long context: F1 by how deep the most buried gold paragraph sits
for dataset in DATASETS:
    for size in SIZES:
        bins = defaultdict(list)
        for c in groups[(dataset, "lc", size, 0, False)].values():
            positions = gold_positions(corpora[c["corpus_id"]]["pids"], questions[c["question_id"]]["gold_pids"])
            depth = max(min(p, 1 - p) for p in positions)       # 0 = at an edge, 0.5 = the middle
            bins[min(int(depth * 10), 4)].append(c["f1"])
        for b, values in sorted(bins.items()):
            position_rows.append({"dataset": dataset, "size": size, "depth": f"{b * 10}-{b * 10 + 10}%",
                                  "f1": mean(values), "n": len(values)})

# ---------------- write CSV files ----------------
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(name, table):
    fields = []
    for entry in table:
        fields += [f for f in entry if f not in fields]
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in table:
            writer.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in entry.items()})


write_csv("accuracy.csv", rows)
write_csv("matched_cost.csv", cells)
write_csv("rag_outcomes.csv", outcome_rows)
write_csv("musique_hops.csv", hop_rows)
write_csv("musique_hop_recall.csv", hop_recall_rows)
write_csv("overlap.csv", overlap_rows)
write_csv("lc_gold_position.csv", position_rows)

# ---------------- write summary.md ----------------
lines = []
add = lines.append


def table(header, body):
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for entry in body:
        add("| " + " | ".join(str(x) for x in entry) + " |")
    add("")


by_system = Counter()
for c in calls:
    by_system[c["system"]] += call_cost(c)
total = sum(by_system.values())
add(f"# Results: run `{args.run}`")
add("")
add("Generated by `python -m scripts.analyze`. Every number comes from the call log and `data/processed/`; "
    "95% confidence intervals are bootstrapped over questions (2,000 resamples, within each seed). "
    f"API prices: the frozen sheet of {PRICE_SHEET_DATE} (`lcvr/prices.py`). "
    f"GPU work for RAG is priced at ${GPU['dollars_per_hour']}/hour ({GPU['instance']}).")
add("")
add("## The run")
add("")
counts = Counter(c["system"] for c in calls)
add(f"- {len(calls):,} answered calls: long context {counts['lc']:,}, RAG {counts['rag']:,}, "
    f"closed book {counts['closed_book']:,}; {len(failed)} failed attempts, all retried.")
add(f"- Billed: ${total:,.2f} (long context ${by_system['lc']:,.2f}, RAG ${by_system['rag']:,.2f}, "
    f"closed book ${by_system['closed_book']:,.2f}).")
cut = Counter(c["system"] for c in calls if c["stop_reason"] == "max_tokens")
add(f"- Replies cut off by the 400-token cap: long context {cut['lc']}, RAG {cut['rag']}, closed book {cut['closed_book']}.")
add(f"- Caching worked as the cost model assumes: {len(lc_corpora) - len(extra_writes)} of {len(lc_corpora)} corpora "
    f"were written to the cache once and read after that. {len(extra_writes)} were written twice, after the run "
    f"was interrupted (about ${extra_write_cost:.2f} extra).")
add("")

for dataset in DATASETS:
    add(f"## Accuracy: {NAMES[dataset]}")
    add("")
    add("F1 (95% CI), then exact match, over 900 questions per cell (300 per seed x 3 seeds). "
        "The same questions at every corpus size; closed book doesn't depend on size.")
    add("")
    body = []
    settings = [("lc", 0, False)] + [("rag", k, rr) for rr in (False, True) for k in (5, 10, 20, 40, 80)]
    for system, k, rr in settings:
        cells_f1 = []
        for size in SIZES:
            r = find(dataset, system, size, k, rr)
            cells_f1.append(f"{r['f1']:.3f} ({r['f1_low']:.2f}-{r['f1_high']:.2f}) / {r['em']:.3f}")
        body.append([setting_label(find(dataset, system, SIZES[0], k, rr))] + cells_f1)
    closed = find(dataset, "closed_book")
    body.append(["Closed book"] + [f"{closed['f1']:.3f} ({closed['f1_low']:.2f}-{closed['f1_high']:.2f}) / {closed['em']:.3f}"] * 4)
    table(["System"] + [f"{size_label(s)}: F1 (CI) / EM" for s in SIZES], body)

add("## Cost per question")
add("")
add("Long context: dollars per question when Q questions share one cached corpus (at Q = 1, sending the "
    "corpus uncached is cheaper, so that price counts). RAG: the API call as billed plus GPU time "
    "(question embedding, search, reranking if on, and the corpus embedding spread over Q questions).")
add("")
for dataset in DATASETS:
    body = []
    for size in SIZES:
        lc = find(dataset, "lc", size)
        r5, r80 = find(dataset, "rag", size, 5, True), find(dataset, "rag", size, 80, True)
        body.append([size_label(size)] + [f"${lc['cost_at'][q]:.5f}" for q in Q_VALUES]
                    + [f"${r5['cost_at'][q]:.5f}" for q in (1, 1000)] + [f"${r80['cost_at'][q]:.5f}" for q in (1, 1000)]
                    + [f"${lc['api_cost']:.5f}"])
    add(f"**{NAMES[dataset]}**")
    add("")
    table(["Corpus"] + [f"LC, Q = {q:,}" for q in Q_VALUES] + ["RAG top 5 rr, Q = 1", "Q = 1,000",
                                                                "RAG top 80 rr, Q = 1", "Q = 1,000",
                                                                "LC as run (billed)"], body)

add("## Long context vs. RAG at matched cost")
add("")
add("In each cell, long context is compared with the most accurate RAG setting that costs no more per "
    "question. Winner: the 95% CI of the F1 difference excludes zero; otherwise a tie. The last column is "
    "the cheapest RAG setting at least as accurate as long context, and its cost as a share of long context's.")
add("")
for dataset in DATASETS:
    body = []
    for cell in (c for c in cells if c["dataset"] == dataset):
        if cell["rag_k"] is None:
            best, diff = "none this cheap", ""
        else:
            name = f"top {cell['rag_k']}{' rr' if cell['rag_reranked'] else ''}"
            best = f"{name}: ${cell['rag_cost']:.5f}, F1 {cell['rag_f1']:.3f}"
            diff = f"{cell['diff']:+.3f} ({cell['diff_low']:+.3f} to {cell['diff_high']:+.3f})"
        iso = (f"top {cell['iso_k']}{' rr' if cell['iso_reranked'] else ''}: {cell['iso_ratio']:.1%}"
               if cell["iso_k"] is not None else "none")
        body.append([size_label(cell["size"]), f"{cell['q']:,}", f"${cell['lc_cost']:.5f}", f"{cell['lc_f1']:.3f}",
                     best, diff, cell["winner"], iso])
    add(f"**{NAMES[dataset]}**")
    add("")
    table(["Corpus", "Q", "LC $/question", "LC F1", "Best RAG at <= LC's cost", "LC - RAG F1 (95% CI)",
           "Winner", "Cheapest RAG as accurate as LC"], body)

add("## RAG errors: retrieval or reasoning")
add("")
add("Each RAG answer: correct (exact match), or wrong with every gold paragraph among the k sent "
    "(reasoning failure), with some (partial retrieval), or with none (retrieval miss). Shares of all questions.")
add("")
for dataset in DATASETS:
    body = []
    for k, rr in ((5, True), (20, False), (20, True), (80, True)):
        for size in SIZES:
            o = next(x for x in outcome_rows if (x["dataset"], x["size"], x["k"], x["reranked"]) == (dataset, size, k, rr))
            wrong = 1 - o["correct"]
            body.append([f"top {k}{', reranked' if rr else ''}", size_label(size)]
                        + [f"{o[name]:.1%}" for name in OUTCOMES]
                        + [f"{o['partial retrieval'] / wrong:.0%}" if wrong else "-", f"{o['correct without all gold']:.1%}"])
    add(f"**{NAMES[dataset]}**")
    add("")
    table(["RAG setting", "Corpus", "Correct", "Reasoning failure", "Partial retrieval", "Retrieval miss",
           "Partial, share of wrong", "Correct without all gold"], body)

add("## Multi-hop detail (MuSiQue)")
add("")
add("F1 by number of hops: long context | RAG top 80 reranked | RAG top 20 reranked.")
add("")
body = []
for size in SIZES:
    entry = [size_label(size)]
    for hops in (2, 3, 4):
        h = next(x for x in hop_rows if x["size"] == size and x["hops"] == hops)
        entry.append(f"{h['lc']:.2f} \\| {h['rag_top80_reranked']:.2f} \\| {h['rag_top20_reranked']:.2f} (n={h['n']})")
    body.append(entry)
table(["Corpus", "2-hop", "3-hop", "4-hop"], body)
add("Share of each hop's gold paragraph found in the top 20 (dense -> reranked). MuSiQue lists gold paragraphs in hop order.")
add("")
body = []
for size in SIZES:
    dense = next(x for x in hop_recall_rows if x["size"] == size and not x["reranked"])
    reranked = next(x for x in hop_recall_rows if x["size"] == size and x["reranked"])
    body.append([size_label(size)] + [f"{dense[f'hop {h}']:.2f} -> {reranked[f'hop {h}']:.2f}" for h in (1, 2, 3, 4)])
table(["Corpus", "Hop 1", "Hop 2", "Hop 3", "Hop 4"], body)

add("## Long context and RAG get different questions right")
add("")
add("Exact match, long context vs. RAG top 80 reranked, on the same questions.")
add("")
table(["Dataset", "Corpus", "Both right", "Only long context", "Only RAG", "Neither"],
      [[NAMES[o["dataset"]], size_label(o["size"])] + [f"{o[k]:.1%}" for k in ("both", "only long context", "only RAG", "neither")]
       for o in overlap_rows])

add("## Long context: where the gold paragraphs sit")
add("")
add("F1 by how far the most deeply buried gold paragraph is from the nearer edge of the corpus "
    "(0% = at the start or end, 50% = the middle). Small bins are noisy.")
add("")
body = []
for dataset in DATASETS:
    for size in SIZES:
        entry = [NAMES[dataset], size_label(size)]
        for b in range(5):
            label = f"{b * 10}-{b * 10 + 10}%"
            p = next((x for x in position_rows if x["dataset"] == dataset and x["size"] == size and x["depth"] == label), None)
            entry.append(f"{p['f1']:.2f} (n={p['n']})" if p else "-")
        body.append(entry)
table(["Dataset", "Corpus"] + [f"{b * 10}-{b * 10 + 10}%" for b in range(5)], body)

add("## Latency")
add("")
add("Median seconds per API call. RAG's local retrieval adds about "
    f"{rates['musique']['rerank_per_pair'] * 100:.2f} s (MuSiQue) or {rates['2wiki']['rerank_per_pair'] * 100:.2f} s "
    "(2Wiki) of reranking when the reranker is on.")
add("")
body = []
for dataset in DATASETS:
    body.append([NAMES[dataset]] + [f"{find(dataset, 'lc', s)['latency_s']:.2f}" for s in SIZES]
                + [f"{find(dataset, 'rag', s, 20, True)['latency_s']:.2f}" for s in SIZES])
table(["Dataset"] + [f"LC {size_label(s)}" for s in SIZES] + [f"RAG top 20 {size_label(s)}" for s in SIZES], body)

(OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT / 'summary.md'} and the CSV files")

# ---------------- figures ----------------
try:
    from lcvr import figures
except ImportError:
    print("matplotlib isn't installed, so no figures (pip install matplotlib)")
else:
    (OUT / "figures").mkdir(exist_ok=True)
    figures.accuracy_vs_cost(rows, lc_cost_at, rag_cost_at, OUT / "figures" / "accuracy_vs_cost.png")
    figures.accuracy_vs_size(rows, OUT / "figures" / "accuracy_vs_size.png")
    figures.crossover_map(cells, OUT / "figures" / "crossover_map.png")
    figures.rag_outcomes(outcome_rows, OUT / "figures" / "rag_outcomes.png")
    figures.cost_vs_q([r for r in rows if r["system"] == "lc"], [r for r in rows if r["system"] == "rag"],
                      lc_cost_at, rag_cost_at, OUT / "figures" / "cost_vs_q.png")
    print(f"Wrote 5 figures to {OUT / 'figures'}")
