#!/usr/bin/env python3
"""
Benchmark ResNet50 (ImageNet-pretrained) on CheXpert (8 pathologies).

Pipeline:
  Phase 1 — Optuna HP search on data seeds 0-4 (60 TPE trials)
  Phase 2 — Final evaluation on data seeds 5-9 with best HPs

Data pool:  834 CXRs from 662 patients (val+test, 8 pathologies with >=100 images)
Per seed:   patient-level split -> 250 images for train/val, rest for test

Usage:
    python benchmark_resnet50_chexpert.py                  # run both phases
    python benchmark_resnet50_chexpert.py --phase search   # HP search only
    python benchmark_resnet50_chexpert.py --phase eval     # eval only
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

try:
    import optuna
except ImportError:
    raise SystemExit("optuna is required: pip install optuna")

# ── Constants ──────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("CHEXPERT_DATA_DIR", "/home/doju/data/CheXpert")
BENCHMARK_CSV = os.path.join(DATA_DIR, "chexpert_benchmark.csv")
OUTPUT_DIR = os.environ.get(
    "CHEXPERT_OUTPUT_DIR", "/home/doju/data/CheXpert/benchmark_output/imagenet"
)

PATHOLOGIES = sorted([
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Support Devices",
])
N_PATHOLOGIES = len(PATHOLOGIES)

N_TRAIN_IMAGES = 250
VAL_FRACTION = 0.2

HP_SEEDS = list(range(5))
EVAL_SEEDS = list(range(5, 10))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EARLY_STOP_PATIENCE = 7
WARMUP_EPOCHS = 2
N_OPTUNA_TRIALS = 60


# ── Data utilities ─────────────────────────────────────────────────────


def load_benchmark_csv(csv_path=BENCHMARK_CSV):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_labels"] = [int(row[p]) for p in PATHOLOGIES]
    return rows


def patient_split(rows, seed, n_train=N_TRAIN_IMAGES, val_frac=VAL_FRACTION):
    """Split by patient into train/val/test. Train+val gets n_train images."""
    rng = random.Random(seed)

    patients = defaultdict(list)
    for row in rows:
        patients[row["patient"]].append(row)

    patient_ids = sorted(patients.keys())
    rng.shuffle(patient_ids)

    train_val_rows = []
    test_rows = []
    for pid in patient_ids:
        if len(train_val_rows) < n_train:
            train_val_rows.extend(patients[pid])
        else:
            test_rows.extend(patients[pid])

    rng.shuffle(train_val_rows)
    n_val = int(len(train_val_rows) * val_frac)
    val_rows = train_val_rows[:n_val]
    train_rows = train_val_rows[n_val:]

    return train_rows, val_rows, test_rows


# ── Dataset ────────────────────────────────────────────────────────────


class CheXpertDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row["_labels"], dtype=torch.float32)
        return img, labels


# ── Augmentation ───────────────────────────────────────────────────────


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1),
                                    translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Model ──────────────────────────────────────────────────────────────


def create_model(dropout=0.0):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, N_PATHOLOGIES),
        )
    else:
        model.fc = nn.Linear(in_features, N_PATHOLOGIES)
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def get_optimizer(model, backbone_lr, head_lr, weight_decay):
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)


# ── Training ───────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    n = 0
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
        all_labels.append(labels.cpu())
        all_probs.append(torch.sigmoid(outputs.float()).cpu())
        n += 1

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()

    per_pathology_auc = {}
    valid_aucs = []
    for i, p in enumerate(PATHOLOGIES):
        if y_true[:, i].sum() > 0 and (1 - y_true[:, i]).sum() > 0:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            per_pathology_auc[p] = float(auc)
            valid_aucs.append(auc)
        else:
            per_pathology_auc[p] = None

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    return {
        "loss": total_loss / max(n, 1),
        "macro_auc": macro_auc,
        "per_pathology_auc": per_pathology_auc,
    }


def train_model(train_rows, val_rows, hparams, device, test_rows=None):
    set_seed(hparams.get("training_seed", 0))

    train_tf = get_transforms(train=True)
    eval_tf = get_transforms(train=False)

    bs = hparams["batch_size"]
    train_loader = DataLoader(
        CheXpertDataset(train_rows, train_tf),
        batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        CheXpertDataset(val_rows, eval_tf),
        batch_size=bs, shuffle=False, num_workers=4, pin_memory=True,
    )

    model = create_model(dropout=hparams["dropout"]).to(device)
    optimizer = get_optimizer(
        model, hparams["backbone_lr"], hparams["head_lr"], hparams["weight_decay"]
    )

    max_epochs = hparams["max_epochs"]
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - WARMUP_EPOCHS, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    criterion = nn.BCEWithLogitsLoss()
    if hparams.get("label_smoothing", 0) > 0:
        ls = hparams["label_smoothing"]
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=None,
        )
        # Manual label smoothing for BCE: done in loss computation below
        _raw_criterion = criterion

        def smoothed_bce(logits, targets):
            targets = targets * (1 - ls) + 0.5 * ls
            return _raw_criterion(logits, targets)

        criterion = smoothed_bce

    scaler = torch.amp.GradScaler()

    best_val_auc = 0.0
    best_state = None
    patience = 0

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        val_auc = evaluate(model, val_loader, device)["macro_auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    test_metrics = None
    if test_rows is not None and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        test_loader = DataLoader(
            CheXpertDataset(test_rows, eval_tf),
            batch_size=bs, shuffle=False, num_workers=4, pin_memory=True,
        )
        test_metrics = evaluate(model, test_loader, device)

    return best_val_auc, test_metrics


# ── Optuna HP search ───────────────────────────────────────────────────


def objective(trial, all_rows, device):
    hparams = {
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-5, 1e-3, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.2, 0.5]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": trial.suggest_categorical("max_epochs", [20, 30, 50]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }

    val_aucs = []
    for i, seed in enumerate(HP_SEEDS):
        hparams["training_seed"] = seed
        train_rows, val_rows, _ = patient_split(all_rows, seed)
        val_auc, _ = train_model(train_rows, val_rows, hparams, device)
        val_aucs.append(val_auc)

        trial.report(np.mean(val_aucs), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(val_aucs)


def run_hp_search(all_rows, device, n_trials=N_OPTUNA_TRIALS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(OUTPUT_DIR, "optuna_study.db")

    study = optuna.create_study(
        study_name="resnet50_chexpert_imagenet",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, all_rows, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    with open(os.path.join(OUTPUT_DIR, "best_hparams.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    return study.best_params


# ── Final evaluation ───────────────────────────────────────────────────


def run_final_eval(best_hparams, all_rows, device):
    all_results = []

    for seed in EVAL_SEEDS:
        print(f"\n=== Eval seed {seed} ===")
        hparams = {**best_hparams, "training_seed": seed}
        train_rows, val_rows, test_rows = patient_split(all_rows, seed)

        print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}, "
              f"Test: {len(test_rows)}")

        val_auc, test_metrics = train_model(
            train_rows, val_rows, hparams, device, test_rows=test_rows
        )

        print(f"  Val AUC:   {val_auc:.4f}")
        print(f"  Test AUC:  {test_metrics['macro_auc']:.4f}")
        for p, auc in test_metrics["per_pathology_auc"].items():
            print(f"    {p}: {auc:.4f}" if auc is not None else f"    {p}: N/A")

        all_results.append({"seed": seed, "val_auc": val_auc,
                            "test_metrics": test_metrics})

    aucs = [r["test_metrics"]["macro_auc"] for r in all_results]

    summary = {
        "source": "imagenet",
        "best_hparams": best_hparams,
        "per_seed_results": all_results,
        "summary": {
            "macro_auc": {"mean": float(np.mean(aucs)),
                          "std": float(np.std(aucs))},
        },
    }

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (mean +/- std across eval seeds 5-9)")
    print(f"  Macro AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet50 (ImageNet) on CheXpert"
    )
    parser.add_argument(
        "--phase", choices=["search", "eval", "both"], default="both",
    )
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    if not os.path.exists(BENCHMARK_CSV):
        raise SystemExit(
            f"{BENCHMARK_CSV} not found. Run prepare_chexpert.py first."
        )

    all_rows = load_benchmark_csv()
    print(f"Loaded {len(all_rows)} images from {BENCHMARK_CSV}")

    # Show example split
    train_ex, val_ex, test_ex = patient_split(all_rows, seed=5)
    print(f"Example split (seed 5): train={len(train_ex)}, "
          f"val={len(val_ex)}, test={len(test_ex)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.phase in ("search", "both"):
        best_hparams = run_hp_search(all_rows, device, n_trials=args.n_trials)
    else:
        hp_path = os.path.join(OUTPUT_DIR, "best_hparams.json")
        print(f"Loading best HPs from {hp_path}")
        with open(hp_path) as f:
            best_hparams = json.load(f)

    if args.phase in ("eval", "both"):
        run_final_eval(best_hparams, all_rows, device)


if __name__ == "__main__":
    main()
