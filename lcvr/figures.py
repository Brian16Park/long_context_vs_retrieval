"""Figures for the report and slides (matplotlib), drawn from the tables scripts/analyze.py builds.

Colors follow the system, the same in every figure: long context blue, RAG orange (dense
search) or aqua (reranked), closed book gray. They come from a validated colorblind-safe
palette; every figure also has a legend or direct labels, so color never carries meaning alone.
"""
import matplotlib

matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LC, RAG, RAG_RR, CLOSED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
OUTCOME_COLORS = {"correct": "#c3c2b7", "reasoning failure": "#6250d6",
                  "partial retrieval": "#eda100", "retrieval miss": "#e34948"}
SIZE_RAMP = {16000: "#86b6ef", 64000: "#3987e5", 256000: "#1c5cab", 512000: "#0d366b"}
DATASET_NAMES = {"musique": "MuSiQue", "2wiki": "2WikiMultihopQA"}
SIZES = (16000, 64000, 256000, 512000)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlesize": 10, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})
MARKER = {"marker": "o", "markersize": 5.5, "markeredgecolor": SURFACE, "markeredgewidth": 1.0}


def size_label(size):
    return f"{size // 1000}K"


def dollars(x, _=None):
    return f"${x:g}" if x >= 0.01 else f"${x:.4f}".rstrip("0")


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def accuracy_vs_cost(rows, lc_cost_at, rag_cost_at, path, q_values=(1, 10, 100, 1000)):
    """F1 against dollars per question, one panel per dataset and corpus size.

    RAG traces a curve over top-k (5 to 80); long context is one accuracy at a cost that falls
    as more questions share the cached corpus: a horizontal segment from Q = 1,000 to Q = 1.
    Shaded bands are 95% confidence intervals.
    """
    datasets = ("musique", "2wiki")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey="row")
    for r, dataset in enumerate(datasets):
        closed = next(x for x in rows if x["dataset"] == dataset and x["system"] == "closed_book")
        for c, size in enumerate(SIZES):
            ax = axes[r, c]
            ax.set_xscale("log")
            for reranked, color, name in ((False, RAG, "RAG, dense search"), (True, RAG_RR, "RAG, reranked")):
                pts = sorted((x for x in rows if x["dataset"] == dataset and x["system"] == "rag"
                              and x["size"] == size and x["reranked"] == reranked), key=lambda x: x["k"])
                xs = [rag_cost_at(x, 100) for x in pts]
                ax.fill_between(xs, [x["f1_low"] for x in pts], [x["f1_high"] for x in pts], color=color, alpha=0.1, lw=0)
                ax.plot(xs, [x["f1"] for x in pts], color=color, label=name, **MARKER, markerfacecolor=color)
                if (r, c) == (0, 3):          # label the ends once, in the panel with room
                    for x, xv in zip(pts, xs):
                        if x["k"] in (5, 80) and reranked == (x["k"] == 80):
                            ax.annotate(f"top {x['k']}", (xv, x["f1"]), textcoords="offset points",
                                        xytext=(0, 8 if x["k"] == 80 else -13), ha="center", fontsize=7, color=MUTED)
            lc = next(x for x in rows if x["dataset"] == dataset and x["system"] == "lc" and x["size"] == size)
            xs = [lc_cost_at(lc, q) for q in q_values]
            ax.fill_between([min(xs), max(xs)], lc["f1_low"], lc["f1_high"], color=LC, alpha=0.1, lw=0)
            ax.plot(xs, [lc["f1"]] * len(xs), color=LC, label="Long context, Q = 1,000 … 1", **MARKER, markerfacecolor=LC)
            for q, xv in zip(q_values, xs):
                if (r, c) == (0, 3) and q in (1, 1000):
                    ax.annotate(f"Q={q:,}", (xv, lc["f1"]), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7, color=MUTED)
            ax.axhline(closed["f1"], color=CLOSED, lw=1.0, label="Closed book (no documents)")
            if r == 0:
                ax.set_title(f"{size_label(size)}-token corpus")
            if c == 0:
                ax.set_ylabel(f"{DATASET_NAMES[dataset]}\nF1")
            if r == 1:
                ax.set_xlabel("Dollars per question (log scale)")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
            ax.xaxis.set_major_formatter(FuncFormatter(dollars))
            ax.set_xlim(8e-5, 0.5)
    axes[0, 0].set_ylim(0.15, 0.75)
    axes[1, 0].set_ylim(0.15, 0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(n) for n in ("Long context, Q = 1,000 … 1", "RAG, dense search", "RAG, reranked",
                                       "Closed book (no documents)")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def accuracy_vs_size(rows, path):
    """F1 against corpus size: long context falls faster than RAG as the corpus grows."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, dataset in zip(axes, ("musique", "2wiki")):
        lines = [("lc", 0, False, LC, "Long context", "o"),
                 ("rag", 80, True, RAG_RR, "RAG, top 80 reranked", "o"),
                 ("rag", 20, True, RAG_RR, "RAG, top 20 reranked", "s")]
        for system, k, reranked, color, name, marker in lines:
            pts = [next(x for x in rows if x["dataset"] == dataset and x["system"] == system and x["size"] == s
                        and x["k"] == k and x["reranked"] == reranked) for s in SIZES]
            ax.fill_between(SIZES, [p["f1_low"] for p in pts], [p["f1_high"] for p in pts], color=color, alpha=0.1, lw=0)
            ax.plot(SIZES, [p["f1"] for p in pts], color=color, label=name,
                    **{**MARKER, "marker": marker}, markerfacecolor=color)
        closed = next(x for x in rows if x["dataset"] == dataset and x["system"] == "closed_book")
        ax.axhline(closed["f1"], color=CLOSED, lw=1.0, label="Closed book (no documents)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [size_label(s) for s in SIZES])
        ax.minorticks_off()
        ax.set_xlim(12000, 700_000)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("Corpus size (tokens)")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.15, 0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, path)


def crossover_map(cells, path, q_values=(1, 10, 100, 1000)):
    """Who wins at matched cost in each (corpus size, Q) cell, and by how much F1."""
    fills = {"LC": "#b7d3f6", "RAG": "#f8c9b5", "tie": "#f0efec"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, dataset in zip(axes, ("musique", "2wiki")):
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row, size in enumerate(SIZES):
            for col, q in enumerate(q_values):
                cell = next(c for c in cells if c["dataset"] == dataset and c["size"] == size and c["q"] == q)
                winner = cell["winner"] if cell["winner"] in fills else "LC"
                ax.add_patch(plt.Rectangle((col + 0.04, row + 0.04), 0.92, 0.92, color=fills[winner], lw=0))
                note = ""
                if cell["rag_k"] is None:
                    text, detail = "LC", "RAG can't be this cheap"
                elif winner == "tie":
                    leader = "LC" if cell["diff"] > 0 else "RAG"
                    text, detail, note = "Tie", f"{leader} +{abs(cell['diff']):.2f} F1", "not significant"
                else:
                    text, detail = winner, f"+{abs(cell['diff']):.2f} F1"
                ax.text(col + 0.5, row + 0.64, text, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
                ax.text(col + 0.5, row + 0.37, detail, ha="center", va="center", fontsize=7.5, color=INK_2)
                if note:
                    ax.text(col + 0.5, row + 0.18, note, ha="center", va="center", fontsize=6.5, color=MUTED)
        ax.set_xlim(0, len(q_values))
        ax.set_ylim(0, len(SIZES))
        ax.set_xticks([i + 0.5 for i in range(len(q_values))], [f"{q:,}" for q in q_values])
        ax.set_yticks([i + 0.5 for i in range(len(SIZES))], [size_label(s) for s in SIZES])
        ax.tick_params(length=0)
        ax.set_xlabel("Q: questions per cached corpus")
        ax.set_title(DATASET_NAMES[dataset])
    axes[0].set_ylabel("Corpus size (tokens)")
    fig.tight_layout()
    save(fig, path)


def rag_outcomes(outcome_rows, path, ks=(5, 20, 80), outcomes=tuple(OUTCOME_COLORS)):
    """What happened to each RAG answer, by corpus size: right, or wrong with all the gold
    paragraphs in hand (reasoning), with some (partial retrieval), or with none."""
    fig, axes = plt.subplots(2, len(ks), figsize=(9, 5.4), sharey=True)
    for r, dataset in enumerate(("musique", "2wiki")):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            ax.grid(axis="x", visible=False)
            bottoms = [0.0] * len(SIZES)
            for outcome in outcomes:
                shares = [next(o for o in outcome_rows if o["dataset"] == dataset and o["size"] == s and o["k"] == k
                               and o["reranked"])[outcome] for s in SIZES]
                bars = ax.bar(range(len(SIZES)), shares, bottom=bottoms, width=0.62, color=OUTCOME_COLORS[outcome],
                              edgecolor=SURFACE, linewidth=1.2, label=outcome[0].upper() + outcome[1:])
                for bar, share, bottom in zip(bars, shares, bottoms):
                    if share >= 0.08:
                        ink = INK if outcome in ("correct", "partial retrieval") else "#ffffff"
                        ax.text(bar.get_x() + bar.get_width() / 2, bottom + share / 2, f"{share:.0%}",
                                ha="center", va="center", fontsize=7, color=ink)
                bottoms = [b + s for b, s in zip(bottoms, shares)]
            ax.set_xticks(range(len(SIZES)), [size_label(s) for s in SIZES])
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
            if r == 0:
                ax.set_title(f"Top {k}, reranked")
            if c == 0:
                ax.set_ylabel(f"{DATASET_NAMES[dataset]}\nShare of questions")
            if r == 1:
                ax.set_xlabel("Corpus size (tokens)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def end_labels(ax, items, x, min_gap=0.13):
    """Label line ends on a log y axis, spread at least `min_gap` decades apart. A label that
    had to move gets a thin leader line to its line's end. items: [(y, text)]."""
    import math
    items = sorted(items)
    placed = []
    for y, text in items:
        ly = math.log10(y)
        if placed and ly - placed[-1][0] < min_gap:
            ly = placed[-1][0] + min_gap
        placed.append((ly, y, text))
    for ly, y, text in placed:
        moved = abs(ly - math.log10(y)) > 0.01
        ax.annotate(text, (x, y), xytext=(x * 1.6, 10 ** ly), textcoords="data", va="center", fontsize=7.5,
                    color=INK_2, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6, "shrinkA": 0, "shrinkB": 2}
                    if moved else None)


def cost_vs_q(lc_rows, rag_rows, lc_cost_at, rag_cost_at, path, q_values=(1, 3, 10, 30, 100, 300, 1000)):
    """Dollars per question as more questions share one corpus: caching makes long context
    cheaper, but above 100K tokens it never gets near RAG's cost."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, dataset in zip(axes, ("musique", "2wiki")):
        labels = []
        for size in SIZES:
            lc = next(x for x in lc_rows if x["dataset"] == dataset and x["size"] == size)
            ys = [lc_cost_at(lc, q) for q in q_values]
            ax.plot(q_values, ys, color=SIZE_RAMP[size], **MARKER, markerfacecolor=SIZE_RAMP[size])
            labels.append((ys[-1], f"Long context, {size_label(size)}"))
        for k, marker in ((80, "o"), (5, "s")):
            rag = next(x for x in rag_rows if x["dataset"] == dataset and x["size"] == 512000 and x["k"] == k
                       and x["reranked"])
            ys = [rag_cost_at(rag, q) for q in q_values]
            ax.plot(q_values, ys, color=RAG_RR, **{**MARKER, "marker": marker}, markerfacecolor=RAG_RR)
            labels.append((ys[-1], f"RAG top {k} reranked, 512K"))
        end_labels(ax, labels, q_values[-1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(dollars))
        ax.set_xlim(0.7, 1500)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("Q: questions per cached corpus")
    axes[0].set_ylabel("Dollars per question (log scale)")
    fig.tight_layout(w_pad=9)
    save(fig, path)
