"""
Figure of one target's test metric per source: mean with a 95% t-interval over the eval seeds,
and each seed as a dot, in the style of Fig. 2 of the survey (arXiv 2510.00902).

    python -m intuitions.figures                                   # CS-xray, macro AUC
    python -m intuitions.figures --target crc --metric balanced_accuracy

Sources without results.json are left out. Writes figures/<target>_<metric>.{pdf,png} in the repository;
at 3.6 in wide, include it at 0.42\\textwidth to match the text size of the survey's Fig. 2.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from intuitions.paths import PROJECT_ROOT, RUNS_DIR

# The survey's source order (ImageNet-1K, RadImageNet, Ecoset), then the benchmark's additions.
SOURCE_NAMES = {"imagenet": "ImageNet-1K", "radimagenet": "RadImageNet", "ecoset_baseline": "Ecoset",
                "ecoset_dvd_s": "Ecoset DVD-S", "scratch": "Random init."}
METRIC_NAMES = {"macro_auc": "Test macro AUC", "balanced_accuracy": "Test balanced accuracy",
                "macro_f1": "Test macro F1"}

# From the survey's Fig. 2: Okabe-Ito blue and grey, black 0.8 pt frame, #e0e0e0 0.8 pt gridlines,
# 11 pt Times-style serif (Liberation Serif has Times New Roman's metrics).
BLUE, GREY, GRID = "#0072b2", "#999999", "#e0e0e0"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def load_per_seed(target, metric):
    """source -> (eval seeds, per-seed test metric) for every source with finished results."""
    per_seed = {}
    for source in SOURCE_NAMES:
        path = RUNS_DIR / "benchmark" / target / source / "results.json"
        if path.exists():
            r = json.loads(path.read_text())
            per_seed[source] = (r["eval_seeds"], np.array([s["test_metrics"][metric] for s in r["per_seed_results"]]))
    if not per_seed:
        raise SystemExit(f"no results.json under {RUNS_DIR / 'benchmark' / target}")
    return per_seed


def ci95_half_width(values):
    n = len(values)
    return stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)


def plot(per_seed, metric):
    """Call inside plt.rc_context(STYLE), which must also cover saving."""
    sources = list(per_seed)
    x = np.arange(len(sources))
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rng = np.random.default_rng(0)
    for xi, source in zip(x, sources):
        values = per_seed[source][1]
        ax.scatter(xi + rng.uniform(-0.12, 0.12, size=len(values)), values, s=10, color=GREY, linewidths=0,
                   zorder=1)
        ax.errorbar(xi, values.mean(), yerr=ci95_half_width(values), fmt="o", color=BLUE, markersize=6,
                    elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
    ax.set_xticks(x, [SOURCE_NAMES[s] for s in sources], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(sources) - 0.4)
    ax.set_ylabel(METRIC_NAMES.get(metric, metric))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=6, linewidth=1.5, label="Mean, 95% CI"),
               Line2D([], [], color=GREY, marker="o", markersize=3.5, linewidth=0, label="Single seed")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.02), handletextpad=0.4,
               columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="chexpert")
    parser.add_argument("--metric", default="macro_auc")
    args = parser.parse_args()

    per_seed = load_per_seed(args.target, args.metric)
    if len({tuple(seeds) for seeds, _ in per_seed.values()}) > 1:
        print("note: sources were evaluated on different seeds")
    out = PROJECT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):  # fonts are resolved when saving, so the style must cover savefig too
        fig = plot(per_seed, args.metric)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{args.target}_{args.metric}.{ext}", dpi=200, bbox_inches="tight")
    for source, (seeds, values) in per_seed.items():
        mean, half = values.mean(), ci95_half_width(values)
        print(f"{SOURCE_NAMES[source]:14s} {mean:.4f} [{mean - half:.4f}, {mean + half:.4f}]  ({len(values)} seeds)")
    print(f"wrote {out / f'{args.target}_{args.metric}'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
