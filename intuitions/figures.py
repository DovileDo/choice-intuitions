"""
Figure of the test metric per source, one panel per target: mean with a 95% t-interval over the
eval seeds, and each seed as a dot, in the style of Fig. 2 of the survey (arXiv 2510.00902).

    python -m intuitions.figures                                   # (a) CS-tissue, (b) CS-xray, macro AUC
    python -m intuitions.figures --targets crc starc9 --metric balanced_accuracy

Rows follow the survey's source order and are shared by all panels; a source without results.json
for a target is marked "pending". Writes figures/results_<metric>.{pdf,png} in the repository; at
6 in wide, include it at 0.71\\textwidth to match the text size of the survey's Fig. 2.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory
from scipy import stats

from intuitions.paths import PROJECT_ROOT, RUNS_DIR

# The survey's source order (ImageNet-1K, RadImageNet, Ecoset), then the benchmark's additions.
SOURCE_NAMES = {"imagenet": "ImageNet-1K", "radimagenet": "RadImageNet", "ecoset_baseline": "Ecoset",
                "ecoset_dvd_s": "Ecoset DVD-S", "scratch": "Random init."}
TARGET_NAMES = {"crc": "CS-tissue", "starc9": "CS-tissue (STARC-9)", "chexpert": "CS-xray"}
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
    return per_seed


def ci95_half_width(values):
    n = len(values)
    return stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)


def plot(panels, metric):
    """panels: [(target, per_seed)]. Call inside plt.rc_context(STYLE), which must also cover saving."""
    sources = list(SOURCE_NAMES)
    y = np.arange(len(sources))[::-1]  # first source on top
    fig, axes = plt.subplots(1, len(panels), figsize=(1.2 + 2.4 * len(panels), 2.8), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(0)
    for i, (ax, (target, per_seed)) in enumerate(zip(axes[0], panels)):
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        rows = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, source in zip(y, sources):
            if source not in per_seed:
                ax.text(0.5, yi, "pending", transform=rows, ha="center", va="center", color=GREY, style="italic")
                continue
            values = per_seed[source][1]
            ax.scatter(values, yi + rng.uniform(-0.12, 0.12, size=len(values)), s=10, color=GREY, linewidths=0,
                       zorder=1)
            ax.errorbar(values.mean(), yi, xerr=ci95_half_width(values), fmt="o", color=BLUE, markersize=6,
                        elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        span = np.ptp(np.concatenate([v for _, v in per_seed.values()])) if per_seed else 1.0
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if span < 0.05 else "%.2f"))
        ax.set_xlabel(METRIC_NAMES.get(metric, metric))
        ax.text(0.5, -0.27, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="center", va="top")
    axes[0][0].set_yticks(y, [SOURCE_NAMES[s] for s in sources])
    axes[0][0].set_ylim(-0.6, len(sources) - 0.4)
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=6, linewidth=1.5, label="Mean, 95% CI"),
               Line2D([], [], color=GREY, marker="o", markersize=3.5, linewidth=0, label="Single seed")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.04), handletextpad=0.4,
               columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=["crc", "chexpert"])
    parser.add_argument("--metric", default="macro_auc")
    args = parser.parse_args()

    panels = [(target, load_per_seed(target, args.metric)) for target in args.targets]
    if not any(per_seed for _, per_seed in panels):
        raise SystemExit(f"no results.json under {RUNS_DIR / 'benchmark'} for {args.targets}")
    out = PROJECT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):  # fonts are resolved when saving, so the style must cover savefig too
        fig = plot(panels, args.metric)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"results_{args.metric}.{ext}", dpi=200, bbox_inches="tight")
    for target, per_seed in panels:
        print(TARGET_NAMES.get(target, target))
        for source in SOURCE_NAMES:
            if source in per_seed:
                values = per_seed[source][1]
                mean, half = values.mean(), ci95_half_width(values)
                print(f"  {SOURCE_NAMES[source]:14s} {mean:.4f} [{mean - half:.4f}, {mean + half:.4f}]  "
                      f"({len(values)} seeds)")
            else:
                print(f"  {SOURCE_NAMES[source]:14s} pending")
    print(f"wrote {out / f'results_{args.metric}'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
