"""
CS-xray: 8-pathology multi-label chest X-ray classification (CheXpert).

Pool: official CheXpert val + test sets, restricted to images with at least one
positive among the 8 pathologies (834 CXRs, 662 patients), built by
`python -m intuitions.prepare_chexpert`. Per data seed, whole patients are
assigned to train+val until it holds >= 250 images and the rest form the test
set; train+val patients are then split 80/20. All splits are patient-disjoint.
"""

import csv
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from intuitions.paths import DATA_DIR
from intuitions.targets.base import Target

DATA_ROOT = DATA_DIR / "CheXpert"
BENCHMARK_CSV = DATA_ROOT / "chexpert_benchmark.csv"

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

N_TRAIN_IMAGES = 250
VAL_FRACTION = 0.2
IMAGE_SIZE = 224


def load_benchmark_rows(csv_path=BENCHMARK_CSV):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run `python -m intuitions.prepare_chexpert` first")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["rel_path"] = row["image_path"]
        row["image_path"] = str(DATA_ROOT / row["image_path"])
        row["_labels"] = [int(row[p]) for p in PATHOLOGIES]
    return rows


def patient_split(rows, seed, n_train=N_TRAIN_IMAGES, val_frac=VAL_FRACTION):
    rng = random.Random(seed)
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient"]].append(row)

    patient_ids = sorted(patients)
    rng.shuffle(patient_ids)

    train_val_pids, train_val_count, test_rows = [], 0, []
    for pid in patient_ids:
        if train_val_count < n_train:
            train_val_pids.append(pid)
            train_val_count += len(patients[pid])
        else:
            test_rows.extend(patients[pid])

    rng.shuffle(train_val_pids)
    n_val_patients = max(1, int(len(train_val_pids) * val_frac))
    val_rows = [r for pid in train_val_pids[:n_val_patients] for r in patients[pid]]
    train_rows = [r for pid in train_val_pids[n_val_patients:] for r in patients[pid]]
    return train_rows, val_rows, test_rows


class CheXpertDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), torch.tensor(row["_labels"], dtype=torch.float32)


def get_transforms(train=True):
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        ]
    return transforms.Compose(t + [transforms.ToTensor()])


class SmoothedBCEWithLogits(nn.Module):
    """BCE with labels smoothed towards 0.5."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


class CheXpert(Target):
    name = "chexpert"
    label_names = tuple(PATHOLOGIES)
    summary_metrics = ("macro_auc",)
    eval_loss = nn.BCEWithLogitsLoss()

    def __init__(self):
        self.rows = load_benchmark_rows()

    def split(self, seed):
        return patient_split(self.rows, seed)

    def make_dataset(self, items, hparams, train):
        return CheXpertDataset(items, get_transforms(train=train))

    def make_criterion(self, hparams):
        return SmoothedBCEWithLogits(hparams["label_smoothing"])

    def to_probs(self, logits):
        return torch.sigmoid(logits)

    def item_ids(self, items):
        return [r["rel_path"] for r in items], [r["patient"] for r in items]

    def metrics(self, y_true, y_prob):
        per_pathology_auc = {}
        for i, p in enumerate(PATHOLOGIES):
            has_both = 0 < y_true[:, i].sum() < len(y_true)
            per_pathology_auc[p] = float(roc_auc_score(y_true[:, i], y_prob[:, i])) if has_both else None
        valid = [auc for auc in per_pathology_auc.values() if auc is not None]
        return {
            "macro_auc": float(np.mean(valid)) if valid else 0.0,
            "per_pathology_auc": per_pathology_auc,
        }
