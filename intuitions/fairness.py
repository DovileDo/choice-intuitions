"""
Fairness of the pretraining sources on CS-xray, compared as in MEDFAIR (Zong et al., ICLR 2023):
every (test set, sensitive attribute) pair is a block, the sources are ranked within each block,
and the ranks are compared with the Friedman test followed by the Nemenyi post-hoc test (Demsar 2006).

Blocks:
  CheXpert       the benchmark test set (saved test predictions): sex, age, race
  ChestX-ray14   the same models on other hospitals' images (saved by intuitions.shift_xray): sex, age
  PadChest       likewise: sex, age
Subgroups as in intuitions.subgroups: female / male, age < 60 / 60+, White / non-White.

Per block and model, over the labels the dataset shares with CheXpert that have positives and
negatives in every subgroup (macro: computed per label, then averaged over labels):
  worst-group macro-AUC   the lowest subgroup macro-AUC (max-min fairness; higher is better)
  macro-AUC gap           highest minus lowest subgroup macro-AUC (group fairness; lower is better)
  equalized odds          MEDFAIR's EqOdd, 1 - (TPR gap + FPR gap) / 2 between the subgroups, per label
                          (1 = equal true and false positive rates; higher is better)
  underdiagnosis gap      gap in the "No Finding" false positive rate, i.e. patients with a finding
                          called healthy (Seyyed-Kalantari et al. 2021; lower is better)
  macro-ECE gap           gap in expected calibration error (15 equal-width bins, Guo et al. 2017;
                          lower is better)
TPR and FPR are taken at the operating point where the dataset as a whole has --specificity (80%, as
in MEDFAIR's TPR@80%TNR and intuitions.subgroups), one threshold per label, model and dataset, so every
subgroup is judged by the same threshold. Gaps are highest minus lowest subgroup. Each metric is
averaged over the evaluation seeds before ranking (rank 1 = best).

The primary metric (--primary, macro-AUC gap by default, chosen before the external-dataset results)
answers whether any source is fairer; the others are secondary and exploratory. As in MEDFAIR, every
Friedman p is uncorrected, and the Nemenyi test is applied where it is below --alpha: sources whose
mean ranks differ by more than the critical difference (CD) differ significantly.

Overall macro-AUC and overall calibration do not depend on the attribute, so the blocks of one dataset
would repeat the same comparison; they are not tested here. Overall macro-AUC is compared in domain by
intuitions.stats and under shift by intuitions.shift_xray; the overall macro-ECE per dataset is
reported alongside, without a test.

    python -m intuitions.fairness
    python -m intuitions.fairness --datasets chexpert       # CheXpert blocks only (no shift_xray run needed)

Writes figures/results_fairness.txt and the critical-difference diagrams
figures/results_fairness_cd.{pdf,png} in the repository; at 4.8 in wide (three sources), include the
diagrams at 0.56\\textwidth for the same text size as figures/results_macro_auc.pdf.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import roc_auc_score

from intuitions.names import SOURCE_NAMES
from intuitions.paths import PROJECT_ROOT, RUNS_DIR
from intuitions.subgroups import ATTRIBUTES, load_predictions, read_demographics
from intuitions.targets.chexpert import DATA_ROOT, PATHOLOGIES

DATASETS = {"chexpert": "CheXpert", "nih": "ChestX-ray14", "padchest": "PadChest"}
METRICS = {  # key -> (name, higher is better)
    "worst": ("Worst-group macro-AUC", True),
    "gap": ("Macro-AUC gap", False),
    "eqodd": ("Equalized odds", True),
    "underdiagnosis": ("Underdiagnosis gap", False),
    "ece_gap": ("Macro-ECE gap", False),
}
SUBGROUP_TABLES = {  # per-subgroup values shown for reference: key -> title
    "auc": "Subgroup macro-AUC",
    "underdiagnosis": "Subgroup underdiagnosis rate (\"No Finding\" false positive rate)",
    "ece": "Subgroup macro-ECE",
}
NO_FINDING = PATHOLOGIES.index("No Finding")
ECE_BINS = 15
SHIFT_PREDICTIONS = RUNS_DIR / "shift" / "chexpert" / "predictions"


def age_group(age):
    return None if np.isnan(age) else ("<60" if age < 60 else "60+")


def chexpert_data(sources, demographics):
    """(seeds, y_true, scored label columns, {attribute: (group per image, groups)}, {source: [y_prob per seed]})."""
    demo = read_demographics(demographics)
    seeds, preds = load_predictions(sources[0])
    patients = list(preds[0]["groups"])
    groups = {attribute: (np.array([group_of(demo[p]) if p in demo else None for p in patients], dtype=object), order)
              for attribute, (group_of, order) in ATTRIBUTES.items()}
    probs = {}
    for source in sources:
        s_seeds, s_preds = load_predictions(source)
        if s_seeds != seeds or any((p["ids"] != preds[0]["ids"]).any() for p in s_preds):
            raise SystemExit(f"{source}: different seeds or test images than {sources[0]}")
        probs[source] = [p["y_prob"] for p in s_preds]
    return seeds, preds[0]["y_true"].astype(int), list(range(len(PATHOLOGIES))), groups, probs


def external_data(name, sources, seeds):
    """As chexpert_data, from the predictions intuitions.shift_xray saved for one external dataset."""
    labels = SHIFT_PREDICTIONS / f"{name}_labels.npz"
    if not labels.exists():
        raise SystemExit(f"{labels} not found; run `python -m intuitions.shift_xray` first")
    meta = np.load(labels)
    groups = {"sex": (np.array([s or None for s in meta["sex"]], dtype=object), ATTRIBUTES["sex"][1]),
              "age": (np.array([age_group(a) for a in meta["age"]], dtype=object), ATTRIBUTES["age"][1])}
    probs = {}
    for source in sources:
        files = [SHIFT_PREDICTIONS / source / f"seed_{seed}_{name}.npy" for seed in seeds]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{source}: no {name} predictions for {missing}; rerun intuitions.shift_xray")
        probs[source] = [np.load(f) for f in files]
    return meta["y_true"].astype(int), [PATHOLOGIES.index(label) for label in meta["scored"]], groups, probs


def threshold(score, negative, specificity):
    """Score above which an image counts as positive, so that `specificity` of the negatives fall at or
    below it (as in intuitions.subgroups.weighted_tpr)."""
    neg = np.sort(score[negative])
    return neg[min(int(np.ceil(specificity * len(neg))) - 1, len(neg) - 1)]


def ece(label, prob):
    """Expected calibration error of binary probabilities: equal-width bins, weighted by their size."""
    b = np.minimum((prob * ECE_BINS).astype(int), ECE_BINS - 1)
    return np.abs(np.bincount(b, label, ECE_BINS) - np.bincount(b, prob, ECE_BINS)).sum() / len(label)


def block_values(y, cols, group, order, probs, specificity):
    """
    Subgroup sizes, the label columns used, and per source the seed means of the per-subgroup values and
    of every metric. Labels without positives or negatives in some subgroup are left out, so all
    subgroups are scored on the same labels; without "No Finding" the underdiagnosis gap is undefined.
    """
    masks = list(group == g for g in order)
    cols = [c for c in cols if all(0 < y[m, c].sum() < m.sum() for m in masks)]
    sizes = {g: int(m.sum()) for g, m in zip(order, masks)}
    per_source = {}
    for source, seed_probs in probs.items():
        auc, tpr, fpr, cal = (np.empty((len(seed_probs), len(order), len(cols))) for _ in range(4))  # seed x group x label
        for i, prob in enumerate(seed_probs):
            for j, c in enumerate(cols):
                positive, score = y[:, c] == 1, prob[:, c]
                called = score > threshold(score, ~positive, specificity)  # one threshold for the whole dataset
                for g, m in enumerate(masks):
                    auc[i, g, j] = roc_auc_score(positive[m], score[m])
                    tpr[i, g, j] = called[m & positive].mean()
                    fpr[i, g, j] = called[m & ~positive].mean()
                    cal[i, g, j] = ece(positive[m], score[m])
        spread = lambda a: a.max(1) - a.min(1)  # highest minus lowest subgroup
        underdiagnosis = fpr[:, :, cols.index(NO_FINDING)] if NO_FINDING in cols else np.full(fpr.shape[:2], np.nan)
        per_source[source] = {
            "groups": {"auc": dict(zip(order, auc.mean(2).mean(0))),
                       "underdiagnosis": dict(zip(order, underdiagnosis.mean(0))),
                       "ece": dict(zip(order, cal.mean(2).mean(0)))},
            "worst": auc.mean(2).min(1).mean(),
            "gap": spread(auc.mean(2)).mean(),
            "eqodd": (1 - (spread(tpr) + spread(fpr)) / 2).mean(),
            "underdiagnosis": spread(underdiagnosis).mean(),
            "ece_gap": spread(cal.mean(2)).mean(),
        }
    return sizes, cols, per_source


def friedman(table, higher_is_better, alpha):
    """Ranks within blocks (1 = best), mean ranks, Friedman statistic and p, and the Nemenyi CD."""
    n, k = table.shape
    ranks = rankdata(-table if higher_is_better else table, axis=1)
    statistic, p = friedmanchisquare(*table.T)
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return ranks, ranks.mean(0), statistic, p, cd


def cliques(mean, cd, significant):
    """Rank intervals of maximal groups of sources that do not differ significantly (all of them if the
    Friedman test did not reject)."""
    r = np.sort(mean)
    if not significant:
        return [(r[0], r[-1])]
    spans = []
    for i in range(len(r)):
        j = max(j for j in range(i, len(r)) if r[j] - r[i] <= cd)
        if j > i and not any(lo <= r[i] and r[j] <= hi for lo, hi in spans):
            spans.append((r[i], r[j]))
    return spans


def cd_diagram(ax, mean, names, cd, spans, title):
    """Critical-difference diagram (Demsar 2006): mean ranks on an axis from 1 (best, left) to k, the CD as
    a scale bar, and bars joining sources that do not differ significantly."""
    k = len(mean)
    ax.set_axis_off()
    ax.plot([1, k], [0, 0], color="black", linewidth=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.08], color="black", linewidth=0.8)
        ax.text(r, 0.14, str(r), ha="center", va="bottom")
    ax.plot([1, 1 + cd], [0.75, 0.75], color="black", linewidth=0.8)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.7, 0.8], color="black", linewidth=0.8)
    ax.text(1 + cd / 2, 0.85, "CD", ha="center", va="bottom")
    order = np.argsort(mean, kind="stable")
    n_left = (k + 1) // 2
    for i, s in enumerate(order):  # best half to the left, nearest the axis first, so no lines cross
        left = i < n_left
        depth = 0.55 + 0.45 * (i if left else k - 1 - i)
        end = 0.85 if left else k + 0.15
        ax.plot([mean[s], mean[s], end], [0, -depth, -depth], color="black", linewidth=0.8)
        ax.text(end - 0.05 if left else end + 0.05, -depth, f"{names[s]} ({mean[s]:.2f})",
                ha="right" if left else "left", va="center")
    for j, (lo, hi) in enumerate(spans):
        ax.plot([lo - 0.04, hi + 0.04], [-0.15 - 0.12 * j, -0.15 - 0.12 * j], color="black", linewidth=3,
                solid_capstyle="butt")
    ax.set_xlim(1 - 1.6, k + 1.6)
    ax.set_ylim(-0.8 - 0.45 * (n_left - 1), 1.2)
    ax.text(0.5, -0.04, title, transform=ax.transAxes, ha="center", va="top")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = DATA_ROOT / "CHEXPERT_DEMO.csv"
    parser.add_argument("--demographics", type=Path,
                        default=default if default.exists() else DATA_ROOT / "CHEXPERT_DEMO.xlsx")
    parser.add_argument("--sources", nargs="+", default=["imagenet", "radimagenet", "ecoset_baseline"])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--specificity", type=float, default=0.8, help="operating point for TPR and FPR")
    parser.add_argument("--primary", choices=list(METRICS), default="gap")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if len(args.sources) < 3:
        raise SystemExit("the Friedman test needs at least three sources")
    metrics = [args.primary] + [m for m in METRICS if m != args.primary]  # primary first

    seeds, *chexpert = chexpert_data(args.sources, args.demographics)
    data = {"chexpert": chexpert}
    for name in args.datasets:
        if name != "chexpert":
            data[name] = external_data(name, args.sources, seeds)

    blocks = {}  # (dataset, attribute) -> (sizes, label columns, per-source values)
    for name in args.datasets:
        y, cols, groups, probs = data[name]
        for attribute, (group, order) in groups.items():
            blocks[(name, attribute)] = block_values(y, cols, group, order, probs, args.specificity)

    names = [SOURCE_NAMES[s] for s in args.sources]
    width = max(len(n) for n in names) + 2
    row = lambda block: f"  {DATASETS[block[0]] + ' ' + block[1]:20s} "
    lines = [f"CS-xray fairness of {', '.join(names)}, compared as in MEDFAIR: {len(blocks)} blocks "
             f"(dataset x attribute); per block the mean over eval seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} models "
             f"per source); TPR and FPR at {args.specificity:.0%} specificity on each dataset; primary metric "
             f"{METRICS[args.primary][0]}, the others secondary; Friedman test (uncorrected, as in MEDFAIR), then "
             f"Nemenyi at alpha = {args.alpha}", "", "Blocks"]
    for block, (sizes, cols, _) in blocks.items():
        lines.append(row(block) + ", ".join(f"{g} {n}" for g, n in sizes.items()) +
                     f" images; {len(cols)} labels: {', '.join(PATHOLOGIES[c] for c in cols)}")

    for key, title in SUBGROUP_TABLES.items():
        lines += ["", f"{title}, mean over seeds", f"  {'':20s} " + "".join(f"{n:>{width + 12}s}" for n in names)]
        for block, (_, _, values) in blocks.items():
            cells = [" / ".join(f"{v:.3f}" for v in values[s]["groups"][key].values()) for s in args.sources]
            lines.append(row(block) + "".join(f"{c:>{width + 12}s}" for c in cells) +
                         f"   ({' / '.join(values[args.sources[0]]['groups'][key])})")

    lines += ["", "Overall macro-ECE per dataset (all images, all shared labels; mean ± std over seeds; not tested)",
              f"  {'':20s} " + "".join(f"{n:>{width + 12}s}" for n in names)]
    for name in args.datasets:
        y, cols, _, probs = data[name]
        cells = []
        for source in args.sources:
            values = np.array([np.mean([ece(y[:, c] == 1, prob[:, c]) for c in cols]) for prob in probs[source]])
            cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        lines.append(f"  {DATASETS[name]:20s} " + "".join(f"{c:>{width + 12}s}" for c in cells))

    results = {}
    for metric in metrics:
        label, higher = METRICS[metric]
        table = np.array([[values[s][metric] for s in args.sources] for _, _, values in blocks.values()])
        defined = ~np.isnan(table).any(1)  # a block without "No Finding" has no underdiagnosis gap
        ranks, mean, statistic, p, cd = friedman(table[defined], higher, args.alpha)
        significant = p < args.alpha
        results[metric] = (mean, cd, significant)
        role = "PRIMARY" if metric == args.primary else "secondary, exploratory"
        lines += ["", f"{label} [{role}] ({'higher' if higher else 'lower'} is better): mean over seeds "
                      f"(rank within block)",
                  f"  {'':20s} " + "".join(f"{n:>{width + 6}s}" for n in names)]
        for block, values, rank in zip([b for b, d in zip(blocks, defined) if d], table[defined], ranks):
            lines.append(row(block) + "".join(f"{f'{v:.4f} ({r:g})':>{width + 6}s}" for v, r in zip(values, rank)))
        lines.append(f"  {'mean rank':20s} " + "".join(f"{m:>{width + 6}.2f}" for m in mean))
        if not defined.all():
            left_out = [row(b).strip() for b, d in zip(blocks, defined) if not d]
            lines.append(f"  left out (no \"No Finding\" positives in some subgroup): {', '.join(left_out)}")
        pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names)) if abs(mean[i] - mean[j]) > cd]
        verdict = (f"Nemenyi CD = {cd:.2f}: " + ("; ".join(f"{names[i]} vs {names[j]} differ (mean ranks "
                                                          f"{mean[i]:.2f} vs {mean[j]:.2f})" for i, j in pairs)
                                                or "no pair differs")
                   if significant else f"no significant difference; Nemenyi not applied (CD would be {cd:.2f})")
        lines.append(f"  Friedman chi2({len(names) - 1}) = {statistic:.2f}, p = {p:.3g}; {verdict}")

    text = "\n".join(lines) + "\n"
    print(text)
    out = PROJECT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_fairness.txt").write_text(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from intuitions.figures import STYLE
    with plt.rc_context(STYLE):  # fonts are resolved when saving, so the style must cover savefig too
        k = len(names)
        fig, axes = plt.subplots(len(metrics), 1, figsize=(1.1 * (k - 1) + 3.5, 1.6 * len(metrics)),
                                 gridspec_kw={"hspace": 0.5})
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            mean, cd, significant = results[metric]
            title = f"({chr(ord('a') + i)}) {METRICS[metric][0]}" + (" (primary)" if metric == args.primary else "")
            cd_diagram(ax, mean, names, cd, cliques(mean, cd, significant), title)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"results_fairness_cd.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'results_fairness.txt'} and {out / 'results_fairness_cd'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
