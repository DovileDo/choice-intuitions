"""
CS-xray: 8-pathology multi-label chest X-ray classification (CheXpert).

All three partitions keep only images with at least one positive among the 8
labels (the survey's rule; frontal and lateral views), built by
`python -m intuitions.prepare_chexpert`:
- train: per data seed, a label-balanced sample from the official training
  split: for each of the 8 labels, 40 images positive for it that were not
  already drawn (320 distinct images). Images with an uncertain (-1) value in
  any of the 8 labels are excluded and blank labels count as negative.
- val: per data seed, 10 images per label from the official validation split
  (80 images), so train and val together match the case study's 50 per label.
- test: the official test set with whole random patients removed until 434
  images remain (the survey's 834 images minus the 400 training images), fixed
  for every seed and source.
The official splits come from disjoint patients.
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

N_TRAIN_PER_LABEL = 40
N_VAL_PER_LABEL = 10
N_TEST_IMAGES = 434
TEST_HOLDOUT_SEED = 0
IMAGE_SIZE = 320  # the resolution CheXpert models are conventionally trained at (Irvin et al. 2019)


def load_benchmark_rows(csv_path=BENCHMARK_CSV):
    """Eligible images with their official split ("train", "val" or "test")."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run `python -m intuitions.prepare_chexpert` first")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows and "split" not in rows[0]:
        raise ValueError(f"{csv_path} predates the official-split design; rerun `python -m intuitions.prepare_chexpert`")
    for row in rows:
        row["rel_path"] = row["image_path"]
        row["image_path"] = str(DATA_ROOT / row["image_path"])
        row["_labels"] = [int(row[p]) for p in PATHOLOGIES]
    return rows


def group_by_patient(rows):
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient"]].append(row)
    return patients


def reduce_test_set(test_rows, n_images=N_TEST_IMAGES, seed=TEST_HOLDOUT_SEED):
    """Remove whole random patients until exactly n_images remain."""
    patients = group_by_patient(test_rows)
    pids = sorted(patients)
    random.Random(seed).shuffle(pids)
    total, removed = len(test_rows), set()
    for pid in pids:
        if total == n_images:
            break
        if total - len(patients[pid]) >= n_images:
            removed.add(pid)
            total -= len(patients[pid])
    if total != n_images:
        raise ValueError(f"cannot reach exactly {n_images} test images by removing whole patients (got {total})")
    return [r for r in test_rows if r["patient"] not in removed]


def sample_label_balanced(rows_pool, seed, per_label):
    """For each label in turn, per_label random images positive for it and not drawn yet."""
    rng = random.Random(seed)
    rows = sorted(rows_pool, key=lambda r: r["rel_path"])
    chosen, sample = set(), []
    for k, label in enumerate(PATHOLOGIES):
        candidates = [r for r in rows if r["_labels"][k] and r["rel_path"] not in chosen]
        if len(candidates) < per_label:
            raise ValueError(f"only {len(candidates)} undrawn images positive for {label}")
        for r in rng.sample(candidates, per_label):
            chosen.add(r["rel_path"])
            sample.append(r)
    return sample


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
        rows = load_benchmark_rows()
        self.train_pool = [r for r in rows if r["split"] == "train"]
        self.val_pool = [r for r in rows if r["split"] == "val"]
        self.test = reduce_test_set([r for r in rows if r["split"] == "test"])

    def split(self, seed):
        if not self.train_pool:
            raise FileNotFoundError(
                f"no official training images in {BENCHMARK_CSV}; put the official train.csv at "
                f"{DATA_ROOT / 'train.csv'} and rerun `python -m intuitions.prepare_chexpert`")
        return (sample_label_balanced(self.train_pool, seed, N_TRAIN_PER_LABEL),
                sample_label_balanced(self.val_pool, seed, N_VAL_PER_LABEL),
                self.test)

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
