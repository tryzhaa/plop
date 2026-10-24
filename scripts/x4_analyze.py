"""Analyse X4 results (reference/ours/x4/*.json): task vs model vs control.

    python scripts/x4_analyze.py

Prints the numbers used in the README and plan, writes reference/ours/x4/summary.json,
and draws docs/img/x4_rho.png and docs/img/x4_controls.png.
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.data import CONTROLS, TASKS  # noqa: E402
from plop_explorer.nfn import MODULE_TYPES  # noqa: E402
from plop_explorer.runlog import save_result  # noqa: E402
from plop_explorer.stats import pick, pick_change_rate, spearman  # noqa: E402

X4 = ROOT / "reference/ours/x4"
SETS = [*TASKS, *CONTROLS]
TYPES = list(MODULE_TYPES)
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#d8d7d2"})


def vec(d):
    return [d[t] for t in TYPES]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main():
    data = {f.stem: json.loads(f.read_text())["result"] for f in sorted(X4.glob("*.json")) if f.stem != "summary"}
    models = list(data)

    # Rank agreement of the 7 type scores between every pair of sets, within each model.
    rho = {m: {(a, b): spearman(vec(data[m][a]), vec(data[m][b])) for a, b in itertools.product(SETS, SETS)}
           for m in models}
    mean_rho = [[mean(rho[m][(a, b)] for m in models) for b in SETS] for a in SETS]

    # Rank agreement between models on the same set.
    cross = {s: mean(spearman(vec(data[a][s]), vec(data[b][s])) for a, b in itertools.combinations(models, 2))
             for s in SETS}

    # How often PLoP's top-3 differs across tasks (same model) vs across models (same task).
    across_tasks = {m: pick_change_rate([data[m][t] for t in TASKS]) for m in models}
    across_models = {t: pick_change_rate([data[m][t] for m in models]) for t in TASKS}

    # Distance of each control from the real tasks: mean |score difference| to the mean of the 4 tasks,
    # next to how far the tasks sit from each other.
    def dist(a, b):
        return mean(abs(x - y) for x, y in zip(vec(a), vec(b)))

    task_mean = {m: {t: mean(data[m][k][t] for k in TASKS) for t in TYPES} for m in models}
    ctrl_dist = {m: {c: dist(data[m][c], task_mean[m]) for c in CONTROLS} for m in models}
    task_spread = {m: mean(dist(data[m][a], data[m][b]) for a, b in itertools.combinations(TASKS, 2)) for m in models}
    ctrl_pick = {m: {c: sorted(pick(data[m][c])) for c in CONTROLS} for m in models}
    task_pick = {m: {t: sorted(pick(data[m][t])) for t in TASKS} for m in models}

    print("Mean Spearman rho between sets (mean over models):")
    print(f"{'':16}" + "".join(f"{s[:8]:>10}" for s in SETS))
    for a, row in zip(SETS, mean_rho):
        print(f"{a:16}" + "".join(f"{v:>10.2f}" for v in row))
    print("\nRho between models on the same set:", {s: round(v, 2) for s, v in cross.items()})
    print("\nTop-3 pick differs, pairs of tasks within a model:", {m: round(v, 2) for m, v in across_tasks.items()})
    print("Top-3 pick differs, pairs of models on a task:", {t: round(v, 2) for t, v in across_models.items()})
    print(f"\n{'model':24}{'task spread':>12}" + "".join(f"{c:>16}" for c in CONTROLS))
    for m in models:
        print(f"{m:24}{task_spread[m]:>12.3f}" + "".join(f"{ctrl_dist[m][c]:>16.3f}" for c in CONTROLS))
    print("\nControl picks (lowest 3):")
    for m in models:
        print(f"{m:24}", {c: "/".join(x.replace("Proj", "") for x in ctrl_pick[m][c]) for c in CONTROLS},
              "| tasks:", {"/".join(x.replace("Proj", "") for x in v) for v in task_pick[m].values()})

    save_result(X4 / "summary.json",
                {"mean_rho": dict(zip(SETS, mean_rho)), "rho_between_models": cross,
                 "pick_change_across_tasks": across_tasks, "pick_change_across_models": across_models,
                 "control_distance": ctrl_dist, "task_spread": task_spread,
                 "control_picks": ctrl_pick, "task_picks": task_pick}, models=models)

    IMG = ROOT / "docs/img"
    IMG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    im = ax.imshow(mean_rho, cmap=SEQ, vmin=0.5, vmax=1)
    for i, row in enumerate(mean_rho):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.8 else INK)
    ax.set_xticks(range(len(SETS)), [s.replace("_", "\n") for s in SETS], fontsize=8)
    ax.set_yticks(range(len(SETS)), [s.replace("_", " ") for s in SETS], fontsize=8)
    ax.axhline(3.5, color=SURFACE, lw=3)
    ax.axvline(3.5, color=SURFACE, lw=3)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho of the 7 type scores")
    ax.set_title(f"Do prompt sets rank the module types alike? (mean over {len(models)} models)", loc="left", fontsize=10)
    fig.savefig(IMG / "x4_rho.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    w = 0.2
    for k, c in enumerate(CONTROLS):
        ax.bar([i + (k - 1) * (w + 0.02) for i in range(len(models))], [ctrl_dist[m][c] for m in models],
               width=w, color=SERIES[k], label=c.replace("_", " "))
    ax.scatter(range(len(models)), [task_spread[m] for m in models], marker="_", s=900, color=INK, zorder=3,
               label="tasks vs each other")
    ax.set_xticks(range(len(models)), models, fontsize=8)
    ax.set_ylabel("mean |NFN difference|\nover the 7 types")
    ax.grid(axis="y", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.set_title("How far each control sits from the real tasks", loc="left", fontsize=11)
    fig.savefig(IMG / "x4_controls.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
