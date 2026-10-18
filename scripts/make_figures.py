"""Draw the README figures from committed results only (reference/ and reference/ours/).

    python scripts/make_figures.py

Writes docs/img/heatmap_llama1b_math.png and docs/img/bars_vs_paper.png, and
prints the selection-set table used in the README.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "reference" / "ours"
IMG = ROOT / "docs" / "img"
TYPES = ["Query", "Key", "Value", "OutProj", "GateProj", "UpProj", "DownProj"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
# Diverging around the random baseline 1.0: blue = below, gray = no alignment, red = above.
DIVERGING = LinearSegmentedColormap.from_list("nfn", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32626"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, validated (adjacent CVD ΔE >= 9.2)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#d8d7d2", "font.size": 10,
})


def load(path):
    return json.loads(path.read_text())


def heatmap():
    paper = load(ROOT / "reference/plop_fig4_llama1b_math.json")["grid"]
    check = load(OURS / "fig4_layer_check.json")
    ours, err = check["result"]["grid_reference"], check["result"]["mean_rel_err"]["reference"]
    grids = [[paper[t] for t in TYPES], [ours[t] for t in TYPES]]
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=4.1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), constrained_layout=True)
    titles = ["Paper (Fig 4)", f"Ours, reference setup (mean |rel. error| per cell {err:.1%})"]
    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 2.4 or v < 0.65 else INK)
        ax.set_xticks(range(len(grid[0])))
        ax.set_yticks(range(len(TYPES)), TYPES)
        ax.set_xlabel("Layer")
        ax.set_title(title, loc="left", fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("NFN (1.0 = random baseline)")
    cb.ax.axhline(1.0, color=INK, lw=1)
    fig.suptitle("Llama-3.2-1B-Instruct, math (GSM8K): NFN per module", x=0.01, ha="left", fontsize=13)
    fig.savefig(IMG / "heatmap_llama1b_math.png", dpi=150)
    plt.close(fig)


def bars():
    paper = load(ROOT / "reference/plop_reference_by_type.json")["Llama-3.2-1B-Instruct"]["math"]
    ours = load(OURS / "Llama-3.2-1B-Instruct_math.json")["result"]
    series = [("Paper (Sec 5)", paper), ("Ours, reference setup", ours["reference"]["by_type"]),
              ("Ours, padding masked", ours["masked"]["by_type"])]
    order = sorted(TYPES, key=paper.get)
    w = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for k, ((label, scores), color) in enumerate(zip(series, SERIES)):
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(order))]
        ax.bar(xs, [scores[t] for t in order], width=w, color=color, label=label)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.text(-0.45, 1.03, "random baseline", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("NFN (mean over 16 layers)")
    ax.grid(axis="y", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvspan(-0.5, 2.5, color="#f0efec", zorder=0)
    ax.text(1, ax.get_ylim()[1] * 0.97, "PLoP picks (lowest 3)", ha="center", va="top", fontsize=9, color=MUTED)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 1))
    ax.set_title("Llama-3.2-1B-Instruct, math: per-type NFN vs the paper", loc="left", fontsize=12)
    fig.savefig(IMG / "bars_vs_paper.png", dpi=150)
    plt.close(fig)


def picks_table():
    short = {t: t.replace("Proj", "") for t in TYPES}
    rows = []
    for f in sorted(OURS.glob("*_*.json")):
        if f.name.startswith("fig4"):
            continue
        d = load(f)
        r = d["result"]["reference"]
        rows.append((d["meta"]["model"].split("/")[-1], d["meta"]["task"],
                     ", ".join(sorted(short[t] for t in r["plop"])), ", ".join(sorted(short[t] for t in r["plop_inv"]))))
    print("| Model | Task | PLoP (lowest 3) | PLoP^-1 (highest 3) |\n|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    heatmap()
    bars()
    picks_table()
