"""
CS-xray test performance per demographic subgroup: sex (female / male), age at the X-ray (< 60 / 60+)
and race (White / non-White, where non-White pools Asian, Black and other; unknown and declined are
left out), from CheXpert's demographics release (CHEXPERT_DEMO.xlsx, or the same table as CSV).

Metrics, averaged over the eight labels:
  tpr  true positive rate within the subgroup at a fixed operating point: per label, the threshold at
       which the whole test set has the given specificity (default 80%, as in Zhang et al. 2022), so
       every patient is judged by the same threshold;
  auc  macro-AUC within the subgroup.
Per source: the mean over evaluation seeds with a 95% t-interval, and the gap between the two
subgroups with a 95% two-level paired bootstrap interval that resamples evaluation seeds and test
patients (thresholds are recomputed on every resample). Uses the saved test predictions only.

    python -m intuitions.subgroups                    # TPR at 80% specificity
    python -m intuitions.subgroups --metric auc
    python -m intuitions.subgroups --specificity 0.9

Prints the results and writes them to figures/results_subgroups_<metric>.txt in the repository.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from intuitions.names import SOURCE_NAMES
from intuitions.paths import PROJECT_ROOT, RUNS_DIR
from intuitions.stats import ci95_half_width, weighted_auc
from intuitions.targets.chexpert import DATA_ROOT


def race_group(race):
    race = (race or "").lower()
    if not race or any(k in race for k in ("unknown", "declined", "refused")):
        return None
    return "White" if "white" in race else "non-White"


ATTRIBUTES = {  # attribute -> (patient record -> group or None, group order)
    "sex": (lambda d: {"Female": "female", "Male": "male"}.get(d.get("GENDER")), ("female", "male")),
    "age": (lambda d: None if d.get("AGE_AT_CXR") in (None, "") else ("<60" if float(d["AGE_AT_CXR"]) < 60 else "60+"),
            ("<60", "60+")),
    "race": (lambda d: race_group(d.get("PRIMARY_RACE")), ("White", "non-White")),
}


def weighted_tpr(score, label, W, mask, specificity):
    """
    TPR among the positives in `mask`, for each row of item weights W, at the threshold where the
    (weighted) negatives of the whole test set reach `specificity`; scores above it count as positive.
    """
    negatives = ~label
    order = np.argsort(score[negatives], kind="mergesort")
    neg_scores = score[negatives][order]
    cum = np.cumsum(W[:, negatives][:, order], axis=1)
    idx = np.minimum((cum < specificity * cum[:, -1:]).sum(1), len(neg_scores) - 1)
    threshold = neg_scores[idx]
    positives = label & mask
    w_pos = W[:, positives]
    return (w_pos * (score[positives][None] > threshold[:, None])).sum(1) / w_pos.sum(1)


def read_demographics(path):
    """patient id -> record, from the CheXpert demographics table (.xlsx or .csv)."""
    if path.suffix == ".xlsx":
        import openpyxl  # only needed for the spreadsheet; the CSV export needs nothing extra
        rows = list(openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True))
        header, body = rows[0], rows[1:]
        records = [dict(zip(header, r)) for r in body]
    else:
        with open(path) as f:
            records = list(csv.DictReader(f))
    return {r["PATIENT"]: r for r in records}


def load_predictions(source):
    run = RUNS_DIR / "benchmark" / "chexpert" / source
    res = json.loads((run / "results.json").read_text())
    preds = [np.load(run / r["predictions"]) for r in res["per_seed_results"]]
    return res["eval_seeds"], preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = DATA_ROOT / "CHEXPERT_DEMO.csv"
    parser.add_argument("--demographics", type=Path,
                        default=default if default.exists() else DATA_ROOT / "CHEXPERT_DEMO.xlsx")
    parser.add_argument("--metric", choices=["tpr", "auc"], default="tpr")
    parser.add_argument("--specificity", type=float, default=0.8, help="operating point for --metric tpr")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    demo = read_demographics(args.demographics)

    if args.metric == "tpr":
        label = f"TPR at {args.specificity:.0%} specificity (threshold per label on the whole test set)"
        metric = lambda score, y, W, mask: weighted_tpr(score, y, W, mask, args.specificity)
    else:
        label = "macro-AUC"
        metric = lambda score, y, W, mask: weighted_auc(score, y, W * mask)

    seeds, preds = load_predictions("imagenet")
    y, patients = preds[0]["y_true"].astype(bool), list(preds[0]["groups"])
    covered = sum(p in demo for p in patients)
    lines = [f"CS-xray test {label}, averaged over the eight labels, per subgroup; {len(patients)} test images, "
             f"{covered} with demographics; eval seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}); "
             f"{args.replicates} bootstrap replicates"]

    units, unit_of = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=args.replicates).astype(np.float64)[:, unit_of]
    seed_idx = rng.integers(0, len(seeds), size=(args.replicates, len(seeds)))
    rows = np.arange(args.replicates)[:, None]
    ones = np.ones((1, len(patients)))
    np.seterr(invalid="ignore", divide="ignore")  # undefined resampled values are handled below

    dropped = {}
    for attribute, (group_of, order) in ATTRIBUTES.items():
        group = np.array([group_of(demo[p]) if p in demo else None for p in patients], dtype=object)
        masks = {g: group == g for g in order}
        lines.append(f"\n{attribute}: " + ", ".join(f"{g} {m.sum()} images" for g, m in masks.items()) +
                     f"; gap = {order[0]} - {order[1]}")
        for source in SOURCE_NAMES:
            s_seeds, s_preds = load_predictions(source)
            if s_seeds != seeds or any((p["groups"] != preds[0]["groups"]).any() for p in s_preds):
                raise SystemExit(f"{source}: different seeds or test images")
            observed, boot = {}, {}
            for g, m in masks.items():
                obs = np.array([np.mean([metric(p["y_prob"][:, k], y[:, k], ones, m)[0] for k in range(y.shape[1])])
                                for p in s_preds])
                per_seed = np.stack([np.mean([metric(p["y_prob"][:, k], y[:, k], W, m) for k in range(y.shape[1])],
                                             axis=0) for p in s_preds], 1)
                observed[g], boot[g] = obs, per_seed[rows, seed_idx].mean(1)
            # A resample can lack positives (or negatives) of a label within a small subgroup, which leaves
            # its value undefined; such replicates are dropped and counted.
            gap = boot[order[0]] - boot[order[1]]
            valid = np.isfinite(gap)
            lo, hi = np.percentile(gap[valid], [2.5, 97.5])
            dropped[attribute] = max(dropped.get(attribute, 0), int((~valid).sum()))
            cells = [f"{g} {observed[g].mean():.3f} [{observed[g].mean() - ci95_half_width(observed[g]):.3f}, "
                     f"{observed[g].mean() + ci95_half_width(observed[g]):.3f}]" for g in order]
            diff = observed[order[0]].mean() - observed[order[1]].mean()
            lines.append(f"  {SOURCE_NAMES[source]:14s} " + "   ".join(cells) +
                         f"   gap {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    lines += [f"\n{attribute}: {n} of {args.replicates} bootstrap replicates dropped (a label without positives "
              f"or negatives in a subgroup)" for attribute, n in dropped.items() if n]
    text = "\n".join(lines) + "\n"
    print(text)
    out = PROJECT_ROOT / "figures" / f"results_subgroups_{args.metric}.txt"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
