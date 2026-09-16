#!/usr/bin/env python3
"""
Benchmark ResNet50 (ImageNet-pretrained) on NCT-CRC-HE / CRC-VAL-HE-7K.

Pipeline:
  Phase 1 — Optuna HP search on data seeds 0-4 (60 TPE trials)
  Phase 2 — Final evaluation on data seeds 5-9 with best HPs

Training data:  250 patches/class sampled from NCT-CRC-HE-100K per seed
Test data:      CRC-VAL-HE-7K minus 250/class (fixed, sampled once)

Usage:
    python benchmark_resnet50_crc.py                     # run both phases
    python benchmark_resnet50_crc.py --phase search      # HP search only
    python benchmark_resnet50_crc.py --phase eval         # eval only (needs prior search)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

try:
    import optuna
except ImportError:
    raise SystemExit("optuna is required: pip install optuna")

try:
    from skimage.color import hed2rgb, rgb2hed

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ── Constants ──────────────────────────────────────────────────────────

DATA_DIR = os.environ.get("PATHOLOGY_DATA_DIR", "/home/doju/data/pathology")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.environ.get(
    "PATHOLOGY_OUTPUT_DIR", "/home/doju/data/pathology/benchmark_output/imagenet"
)

CLASSES = sorted(["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"])
N_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

N_SAMPLES_PER_CLASS = 250
VAL_FRACTION = 0.2

HP_SEEDS = list(range(5))
EVAL_SEEDS = list(range(5, 10))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EARLY_STOP_PATIENCE = 7
WARMUP_EPOCHS = 2
N_OPTUNA_TRIALS = 60


# ── Data utilities ─────────────────────────────────────────────────────


def list_class_files(data_dir):
    class_files = {}
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.endswith(".tif")
        )
        class_files[cls] = files
    return class_files


def create_reduced_test_set(test_dir, seed=0):
    """Remove 250/class from CRC-VAL-HE-7K, return remaining as test set."""
    rng = random.Random(seed)
    class_files = list_class_files(test_dir)
    test_files = []
    removed_files = []

    for cls in CLASSES:
        files = class_files[cls]
        n_remove = min(N_SAMPLES_PER_CLASS, len(files))
        shuffled = files.copy()
        rng.shuffle(shuffled)
        removed = shuffled[:n_remove]
        kept = shuffled[n_remove:]
        test_files.extend([(f, CLASS_TO_IDX[cls]) for f in kept])
        removed_files.extend([(f, CLASS_TO_IDX[cls]) for f in removed])

    return test_files, removed_files


def sample_train_val(train_dir, seed, n_per_class=N_SAMPLES_PER_CLASS,
                     val_frac=VAL_FRACTION):
    """Sample n_per_class from NCT-CRC-HE-100K, split 80/20 train/val."""
    rng = random.Random(seed)
    class_files = list_class_files(train_dir)

    train_files = []
    val_files = []

    for cls in CLASSES:
        files = class_files[cls]
        sampled = rng.sample(files, min(n_per_class, len(files)))
        n_val = int(len(sampled) * val_frac)
        rng.shuffle(sampled)
        val_files.extend([(f, CLASS_TO_IDX[cls]) for f in sampled[:n_val]])
        train_files.extend([(f, CLASS_TO_IDX[cls]) for f in sampled[n_val:]])

    return train_files, val_files


# ── Dataset ────────────────────────────────────────────────────────────


class PatchDataset(Dataset):
    def __init__(self, file_label_pairs, transform=None):
        self.samples = file_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Augmentation ───────────────────────────────────────────────────────


class RandomRotation90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        return transforms.functional.rotate(img, angle)


class HEDStainAugmentation:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, img):
        if not HAS_SKIMAGE:
            return img
        arr = np.array(img).astype(np.float64) / 255.0
        hed = rgb2hed(arr)
        hed += np.random.normal(0, self.sigma, hed.shape)
        rgb = hed2rgb(hed)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)


def get_transforms(stain_aug=False, train=True):
    if train:
        t = []
        if stain_aug:
            t.append(HEDStainAugmentation(sigma=0.05))
        t.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotation90(),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transforms.Compose(t)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Model ──────────────────────────────────────────────────────────────


def create_model(dropout=0.0):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, N_CLASSES),
        )
    else:
        model.fc = nn.Linear(2048, N_CLASSES)
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
    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
        probs = torch.softmax(outputs.float(), dim=1)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(outputs.argmax(dim=1).cpu())
        n += 1

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        macro_auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_auc = 0.0

    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = y_true == i
        per_class_acc[cls] = float((y_pred[mask] == i).mean()) if mask.sum() > 0 else 0.0

    return {
        "loss": total_loss / max(n, 1),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_model(train_files, val_files, hparams, device, test_files=None):
    """Full training run. Returns (best_val_auc, test_metrics_or_None)."""
    set_seed(hparams.get("training_seed", 0))

    train_tf = get_transforms(stain_aug=hparams["stain_augmentation"], train=True)
    eval_tf = get_transforms(train=False)

    bs = hparams["batch_size"]
    train_loader = DataLoader(
        PatchDataset(train_files, train_tf),
        batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        PatchDataset(val_files, eval_tf),
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

    criterion = nn.CrossEntropyLoss(label_smoothing=hparams["label_smoothing"])
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
    if test_files is not None and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        test_loader = DataLoader(
            PatchDataset(test_files, eval_tf),
            batch_size=bs, shuffle=False, num_workers=4, pin_memory=True,
        )
        test_metrics = evaluate(model, test_loader, device)

    return best_val_auc, test_metrics


# ── Optuna HP search ───────────────────────────────────────────────────


def objective(trial, device):
    hparams = {
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-5, 1e-3, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.2, 0.5]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": trial.suggest_categorical("max_epochs", [20, 30, 50]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "stain_augmentation": trial.suggest_categorical(
            "stain_augmentation", [True, False]
        ),
    }

    val_aucs = []
    for i, seed in enumerate(HP_SEEDS):
        hparams["training_seed"] = seed
        train_files, val_files = sample_train_val(TRAIN_DIR, seed)
        val_auc, _ = train_model(train_files, val_files, hparams, device)
        val_aucs.append(val_auc)

        trial.report(np.mean(val_aucs), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(val_aucs)


def run_hp_search(device, n_trials=N_OPTUNA_TRIALS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(OUTPUT_DIR, "optuna_study.db")

    study = optuna.create_study(
        study_name="resnet50_crc_imagenet",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, device),
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


def run_final_eval(best_hparams, device):
    all_results = []

    for seed in EVAL_SEEDS:
        print(f"\n=== Eval seed {seed} ===")
        hparams = {**best_hparams, "training_seed": seed}
        train_files, val_files = sample_train_val(TRAIN_DIR, seed)
        test_files, _ = create_reduced_test_set(TEST_DIR, seed=seed)

        val_auc, test_metrics = train_model(
            train_files, val_files, hparams, device, test_files=test_files
        )

        print(f"  Val AUC:            {val_auc:.4f}")
        print(f"  Test set size:      {len(test_files)}")
        print(f"  Test balanced acc:  {test_metrics['balanced_accuracy']:.4f}")
        print(f"  Test macro F1:      {test_metrics['macro_f1']:.4f}")
        print(f"  Test macro AUC:     {test_metrics['macro_auc']:.4f}")

        all_results.append({"seed": seed, "val_auc": val_auc,
                            "test_metrics": test_metrics})

    bal_accs = [r["test_metrics"]["balanced_accuracy"] for r in all_results]
    f1s = [r["test_metrics"]["macro_f1"] for r in all_results]
    aucs = [r["test_metrics"]["macro_auc"] for r in all_results]

    summary = {
        "source": "imagenet",
        "best_hparams": best_hparams,
        "per_seed_results": all_results,
        "summary": {
            "balanced_accuracy": {"mean": float(np.mean(bal_accs)),
                                  "std": float(np.std(bal_accs))},
            "macro_f1": {"mean": float(np.mean(f1s)),
                         "std": float(np.std(f1s))},
            "macro_auc": {"mean": float(np.mean(aucs)),
                          "std": float(np.std(aucs))},
        },
    }

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (mean +/- std across eval seeds 5-9)")
    print(f"  Balanced accuracy: {np.mean(bal_accs):.4f} +/- {np.std(bal_accs):.4f}")
    print(f"  Macro F1:          {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Macro AUC:         {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet50 (ImageNet) on CRC pathology"
    )
    parser.add_argument(
        "--phase", choices=["search", "eval", "both"], default="both",
        help="search = HP search only, eval = final eval only, both = full pipeline",
    )
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Show test set structure for one example seed
    example_test, example_removed = create_reduced_test_set(TEST_DIR, seed=5)
    per_class = {}
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        per_class[cls] = sum(1 for _, l in example_test if l == idx)
    print(f"Test set (per seed): ~{len(example_test)} patches "
          f"(removing {len(example_removed)} from CRC-VAL-HE-7K)")
    print(f"  Per-class: {per_class}")

    if args.phase in ("search", "both"):
        if HAS_SKIMAGE:
            print("scikit-image available — HED stain augmentation enabled")
        else:
            print("WARNING: scikit-image not installed — stain_augmentation "
                  "will be a no-op. Install: pip install scikit-image")
        best_hparams = run_hp_search(device, n_trials=args.n_trials)
    else:
        hp_path = os.path.join(OUTPUT_DIR, "best_hparams.json")
        print(f"Loading best HPs from {hp_path}")
        with open(hp_path) as f:
            best_hparams = json.load(f)

    if args.phase in ("eval", "both"):
        run_final_eval(best_hparams, device)


if __name__ == "__main__":
    main()
