"""Analyse X5 (prompt count) and X6 (base vs instruct): prints the numbers and draws docs/img/x5_stability.png.

    python scripts/x5_x6_analyze.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.nfn import MODULE_TYPES  # noqa: E402
from plop_explorer.stats import pick, spearman  # noqa: E402

REF = ROOT / "reference/ours"
TYPES = list(MODULE_TYPES)
NS = [5, 10, 25, 50, 100]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#d8d7d2"})


def load(path):
    return json.loads(path.read_text())["result"]


def x5():
    results = {f.stem: load(f) for f in sorted((REF / "x5").glob("*.json"))}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    print("X5: per model and n: mean per-type std over 5 seeds | share of seeds giving the n=100 seed-0 pick")
    for (model, runs), color in zip(results.items(), SERIES):
        spread, agree = [], []
        ref_pick = pick(runs["n100_seed0"])
        for n in NS:
            sets = [runs[f"n{n}_seed{s}"] for s in range(5)]
            spread.append(statistics.mean(statistics.stdev(s[t] for s in sets) for t in TYPES))
            agree.append(sum(pick(s) == ref_pick for s in sets) / len(sets))
        print(f"{model:24}", " ".join(f"n={n}: {sp:.3f}|{a:.0%}" for n, sp, a in zip(NS, spread, agree)))
        axes[0].plot(NS, spread, "o-", color=color, lw=2, label=model)
        axes[1].plot(NS, agree, "o-", color=color, lw=2, label=model)
    axes[0].set_ylabel("mean std of NFN across 5 seeds")
    axes[1].set_ylabel("seeds giving the reference PLoP pick")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(NS, NS)
        ax.set_xlabel("prompts")
        ax.grid(color="#e8e7e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / "docs/img/x5_stability.png", dpi=150)
    plt.close(fig)


def x6():
    pairs = [("gemma-3-1b-pt", "gemma-3-1b-it"), ("Qwen3-0.6B-Base", "Qwen3-0.6B"),
             ("Llama-3.2-1B", "Llama-3.2-1B-Instruct")]
    print("\nX6: base vs instruct, math")
    for base, inst in pairs:
        a, b = load(REF / "x6" / f"{base}.json"), load(REF / "x6" / f"{inst}.json")
        diff = max(abs(a[t] - b[t]) for t in TYPES)
        print(f"{inst:24} rho={spearman([a[t] for t in TYPES], [b[t] for t in TYPES]):.2f} "
              f"max|diff|={diff:.2f} base pick={sorted(pick(a))} instruct pick={sorted(pick(b))}")


if __name__ == "__main__":
    x5()
    x6()
