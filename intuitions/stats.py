"""
Statistical comparison of the sources on each target's test macro-AUC.

Two-level paired bootstrap: every replicate resamples the evaluation seeds and the test units (patients
for CS-xray; patches for CS-tissue, which has no patient identifiers), with the same draws for every
model, and recomputes each model's macro-AUC averaged over the drawn seeds. Per target there are two
families of comparisons, each Holm-corrected: the pretrained models compared with each other, and each
pretrained model against the randomly initialised reference.

    python -m intuitions.stats                           # CS-tissue and CS-xray
    python -m intuitions.stats --targets chexpert --replicates 2000

Prints the results and writes them to figures/results_macro_auc_stats.txt in the repository.
"""

import argparse
import itertools
import json

import numpy as np
from scipy import stats

from intuitions.names import SOURCE_NAMES, TARGET_NAMES
from intuitions.paths import PROJECT_ROOT, RUNS_DIR

REFERENCE = "scratch"


def ci95_half_width(values):
    """Half-width of the 95% t-interval for the mean of values."""
    n = len(values)
    return stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def weighted_auc(score, label, W):
    """AUC for each row of item weights W (replicates x items); ties count one half, as in roc_auc_score."""
    order = np.argsort(score, kind="mergesort")
    s, pos = score[order], label[order].astype(bool)
    Ws = W[:, order]
    new_value = np.r_[True, s[1:] != s[:-1]]
    starts, group = np.flatnonzero(new_value), np.cumsum(new_value) - 1
    neg_per_group = np.add.reduceat(Ws * ~pos, starts, axis=1)
    neg_below = np.cumsum(neg_per_group, axis=1) - neg_per_group
    credit = neg_below[:, group[pos]] + 0.5 * neg_per_group[:, group[pos]]
    w_pos = Ws[:, pos]
    return (w_pos * credit).sum(1) / (w_pos.sum(1) * (Ws * ~pos).sum(1))


def macro_auc(prob, Y, W):
    """Macro-AUC over the columns of Y (labels, or one-vs-rest classes) for each row of weights W."""
    return np.mean([weighted_auc(prob[:, k], Y[:, k], W) for k in range(Y.shape[1])], axis=0)


def load(target):
    """Eval seeds, binary targets (items x outputs), resampling group per item, and source -> seed x item x output."""
    seeds = ids = y = groups = None
    prob = {}
    for source in SOURCE_NAMES:
        res = json.loads((RUNS_DIR / "benchmark" / target / source / "results.json").read_text())
        seeds = seeds or res["eval_seeds"]
        if res["eval_seeds"] != seeds:
            raise SystemExit(f"{target}: {source} was evaluated on seeds {res['eval_seeds']}, not {seeds}")
        runs = [np.load(RUNS_DIR / "benchmark" / target / source / r["predictions"]) for r in res["per_seed_results"]]
        for r in runs:
            if ids is None:
                ids, y, groups = r["ids"], r["y_true"], r["groups"]
            if not ((r["ids"] == ids).all() and (r["y_true"] == y).all()):
                raise SystemExit(f"{target}: test sets differ between runs")
        prob[source] = np.stack([r["y_prob"].astype(np.float64) for r in runs])
    Y = y if y.ndim == 2 else np.eye(prob[source].shape[-1])[y]  # multi-label, or one-vs-rest
    return seeds, Y.astype(bool), groups, prob


def analyse(target, replicates):
    seeds, Y, groups, prob = load(target)
    S = len(seeds)
    units, unit_of = np.unique(groups, return_inverse=True)
    ones = np.ones((1, len(Y)))
    observed = {s: np.array([macro_auc(prob[s][j], Y, ones)[0] for j in range(S)]) for s in prob}

    rng = np.random.default_rng(0)
    W = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=replicates).astype(np.float64)[:, unit_of]
    seed_idx = rng.integers(0, S, size=(replicates, S))
    rows = np.arange(replicates)[:, None]
    boot = {s: np.stack([macro_auc(prob[s][j], Y, W) for j in range(S)], 1)[rows, seed_idx].mean(1) for s in prob}

    fmt_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    lines = [f"{TARGET_NAMES.get(target, target)}: test macro-AUC over {Y.shape[1]} outputs; {len(Y)} test images in "
             f"{len(units)} resampling units; eval seeds {seeds[0]}-{seeds[-1]} ({S}); {replicates} bootstrap replicates",
             "  mean [95% t-interval over seeds]"]
    for s in prob:
        m, h = observed[s].mean(), ci95_half_width(observed[s])
        lines.append(f"    {SOURCE_NAMES[s]:14s} {m:.4f} [{m - h:.4f}, {m + h:.4f}]")
    pretrained = [s for s in prob if s != REFERENCE]
    families = {"pretrained models compared with each other": list(itertools.combinations(pretrained, 2)),
                "each pretrained model vs random init. (reference)": [(s, REFERENCE) for s in pretrained]}
    for family, pairs in families.items():
        diffs = [boot[a] - boot[b] for a, b in pairs]
        p = np.array([min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) for d in diffs])
        lines.append(f"  {family}: difference [95% bootstrap CI], p, Holm-corrected p over {len(pairs)}")
        for (a, b), d, p_raw, p_holm in zip(pairs, diffs, p, holm(p)):
            lo, hi = np.percentile(d, [2.5, 97.5])
            mark = "*" if p_holm < 0.05 else " "
            lines.append(f"   {mark} {SOURCE_NAMES[a]:>12s} - {SOURCE_NAMES[b]:<14s} "
                         f"{observed[a].mean() - observed[b].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                         f"p {fmt_p(p_raw):>7s}  Holm p {fmt_p(p_holm):>7s}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=["crc", "chexpert"])
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()

    report = []
    for target in args.targets:
        report += analyse(target, args.replicates) + [""]
    text = "\n".join(report)
    print(text)
    out = PROJECT_ROOT / "figures" / "results_macro_auc_stats.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
