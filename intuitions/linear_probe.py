"""
Linear probe on frozen source features, on the fine-tuning benchmark's own CheXpert splits.

For each evaluation seed it uses the same training, validation and test images as the
fine-tuning runs, extracts frozen features from each source model, and fits one logistic
regression per label. The regularisation strength is chosen on validation macro AUC, the
criterion the fine-tuning search selects on, and test macro AUC is reported next to the
fine-tuned result where runs/benchmark/chexpert/<source>/results.json exists.

    python -m intuitions.linear_probe
    python -m intuitions.linear_probe --sources imagenet radimagenet --seeds 5 6 7

Writes runs/linear_probe/chexpert/{linear_probe.log,results.json}.
"""

import argparse
import json
import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from intuitions.finetune import EVAL_SEEDS, NUM_WORKERS
from intuitions.paths import RUNS_DIR
from intuitions.source_models import SOURCES, load_source_model
from intuitions.targets.chexpert import CheXpert, CheXpertDataset, get_transforms
from intuitions.utils import git_revision, pick_device, setup_logging

log = logging.getLogger("intuitions.linear_probe")

C_GRID = np.logspace(-4, 1, 6)


@torch.no_grad()
def extract_features(source, rows, device, batch_size=64):
    """rel_path -> penultimate-layer features, with evaluation transforms and the source's own normalisation."""
    model = load_source_model(source, num_classes=1).to(device).eval()
    model.fc = torch.nn.Identity()
    loader = DataLoader(CheXpertDataset(rows, get_transforms(train=False)),
                        batch_size=batch_size, num_workers=NUM_WORKERS)
    feats = torch.cat([model(x.to(device)).float().cpu() for x, _ in loader]).numpy()
    return dict(zip((r["rel_path"] for r in rows), feats))


def fit_predict(x_train, y_train, x_eval, C):
    """One logistic regression per label on standardised features; probabilities for each set in x_eval."""
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = [scaler.transform(x) for x in x_eval]
    probs = [np.zeros((len(x), y_train.shape[1])) for x in x_eval]
    for k in range(y_train.shape[1]):
        clf = LogisticRegression(C=C, max_iter=2000).fit(x_train, y_train[:, k])
        for p, x in zip(probs, x_eval):
            p[:, k] = clf.predict_proba(x)[:, 1]
    return probs


def probe_seed(target, features, split):
    """Best C on validation macro AUC, and the test metrics at that C."""
    train, val, test = split
    x = lambda rows: np.stack([features[r["rel_path"]] for r in rows])
    y = lambda rows: np.array([r["_labels"] for r in rows])
    val_auc, best = {}, None
    for C in C_GRID:
        val_prob, test_prob = fit_predict(x(train), y(train), [x(val), x(test)], C)
        val_auc[float(C)] = target.metrics(y(val), val_prob)["macro_auc"]
        if best is None or val_auc[float(C)] > val_auc[best[0]]:
            best = (float(C), target.metrics(y(test), test_prob))
    return {"C": best[0], "val_macro_auc": val_auc[best[0]], "val_macro_auc_by_C": val_auc, "test": best[1]}


def fine_tuned_summary(source):
    path = RUNS_DIR / "benchmark" / "chexpert" / source / "results.json"
    return json.loads(path.read_text())["summary"]["macro_auc"] if path.exists() else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    p.add_argument("--device", default=None, help="default: cuda if available")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = RUNS_DIR / "linear_probe" / "chexpert"
    setup_logging(out_dir / "linear_probe.log")
    device = pick_device(args.device)
    log.info("git %s | torch %s | device %s | %s", git_revision(), torch.__version__, device, vars(args))

    target = CheXpert()
    splits = {seed: target.split(seed) for seed in args.seeds}
    rows = list({r["rel_path"]: r for split in splits.values() for part in split for r in part}.values())
    log.info("%d distinct images across seeds %s", len(rows), args.seeds)

    results = {}
    for source in args.sources:
        features = extract_features(source, rows, device)
        per_seed = {}
        for seed in args.seeds:
            per_seed[seed] = probe_seed(target, features, splits[seed])
            log.info("%s seed %d: C %g | val macro AUC %.4f | test macro AUC %.4f", source, seed,
                     per_seed[seed]["C"], per_seed[seed]["val_macro_auc"], per_seed[seed]["test"]["macro_auc"])
        aucs = [r["test"]["macro_auc"] for r in per_seed.values()]
        results[source] = {"per_seed": per_seed, "test_macro_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
                           "fine_tuned_test_macro_auc": fine_tuned_summary(source)}

    log.info("%-16s %-20s %s", "source", "linear probe", "fine-tuned")
    for source, r in results.items():
        probe, tuned = r["test_macro_auc"], r["fine_tuned_test_macro_auc"]
        log.info("%-16s %.4f +/- %.4f      %s", source, probe["mean"], probe["std"],
                 f"{tuned['mean']:.4f} +/- {tuned['std']:.4f}" if tuned else "not run yet")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
