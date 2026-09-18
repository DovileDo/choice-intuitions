"""
CS-tissue: 9-class colorectal cancer tissue classification.

Train/val: 250 patches per class sampled per data seed from NCT-CRC-HE-100K,
split 80/20. Test: a fixed subset of CRC-VAL-HE-7K, the survey's 7,180-patch
target dataset, with 250 random patches per class removed (4,930 patches), so
the training budget plus the test set add up to the target dataset's size. The
test set is identical for every seed and source. CRC-VAL-HE-7K comes from 50
patients not in NCT-CRC-HE-100K, so train and test are patient-disjoint.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from skimage.color import hed2rgb, rgb2hed
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from intuitions.paths import DATA_DIR
from intuitions.targets.base import Target

DATA_ROOT = DATA_DIR / "pathology"
TRAIN_DIR = str(DATA_ROOT / "train")
TEST_DIR = str(DATA_ROOT / "test")

CLASSES = sorted(["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

N_SAMPLES_PER_CLASS = 250
VAL_FRACTION = 0.2
TEST_HOLDOUT_SEED = 0


def list_class_files(split_dir):
    class_files = {}
    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        class_files[cls] = sorted(
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".tif")
        )
    return class_files


def sample_train_val(seed, train_dir=TRAIN_DIR, n_per_class=N_SAMPLES_PER_CLASS, val_frac=VAL_FRACTION):
    """Sample n_per_class patches per class and split them into train/val."""
    rng = random.Random(seed)
    class_files = list_class_files(train_dir)
    train_files, val_files = [], []
    for cls in CLASSES:
        files = class_files[cls]
        sampled = rng.sample(files, min(n_per_class, len(files)))
        n_val = int(len(sampled) * val_frac)
        rng.shuffle(sampled)
        val_files.extend((f, CLASS_TO_IDX[cls]) for f in sampled[:n_val])
        train_files.extend((f, CLASS_TO_IDX[cls]) for f in sampled[n_val:])
    return train_files, val_files


def list_test_files(test_dir=TEST_DIR, n_removed_per_class=N_SAMPLES_PER_CLASS, seed=TEST_HOLDOUT_SEED):
    """CRC-VAL-HE-7K minus n_removed_per_class random patches per class; the same for every data seed."""
    rng = random.Random(seed)
    class_files = list_class_files(test_dir)
    test_files = []
    for cls in CLASSES:
        shuffled = class_files[cls].copy()
        rng.shuffle(shuffled)
        test_files.extend((f, CLASS_TO_IDX[cls]) for f in sorted(shuffled[n_removed_per_class:]))
    return test_files


class PatchDataset(Dataset):
    def __init__(self, file_label_pairs, transform):
        self.samples = file_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class RandomRotation90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle) if angle else img


class HEDStainAugmentation:
    """Tellez et al. HED jitter: one random scale and shift per stain channel per image."""

    def __init__(self, sigma=0.02):  # 0.05 gives strong colour shifts in skimage>=0.19 HED units
        self.sigma = sigma

    def __call__(self, img):
        hed = rgb2hed(np.asarray(img).astype(np.float64) / 255.0)
        alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma, size=3)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3)
        rgb = hed2rgb(hed * alpha + beta)
        return Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))


def get_transforms(stain_aug=False, train=True):
    # Patches are Macenko-normalised 224x224 tiles; no resizing or colour jitter by default.
    if not train:
        return transforms.ToTensor()
    t = [HEDStainAugmentation()] if stain_aug else []
    t += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotation90(),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(t)


class CRC(Target):
    name = "crc"
    label_names = tuple(CLASSES)
    summary_metrics = ("balanced_accuracy", "macro_f1", "macro_auc")
    eval_loss = nn.CrossEntropyLoss()

    def suggest_extra_hparams(self, trial):
        return {"stain_augmentation": trial.suggest_categorical("stain_augmentation", [True, False])}

    def split(self, seed):
        train, val = sample_train_val(seed)
        return train, val, list_test_files()

    def make_dataset(self, items, hparams, train):
        return PatchDataset(items, get_transforms(stain_aug=train and hparams["stain_augmentation"], train=train))

    def make_criterion(self, hparams):
        return nn.CrossEntropyLoss(label_smoothing=hparams["label_smoothing"])

    def to_probs(self, logits):
        return torch.softmax(logits, dim=1)

    def item_ids(self, items):
        # NCT/CRC patches carry no patient identifiers, so each patch is its own resampling unit.
        ids = [os.path.relpath(path, DATA_ROOT) for path, _ in items]
        return ids, ids

    def metrics(self, y_true, y_prob):
        y_pred = y_prob.argmax(axis=1)
        try:
            macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            macro_auc = 0.0
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "macro_auc": float(macro_auc),
            "per_class_accuracy": {
                cls: float((y_pred[y_true == i] == i).mean()) if (y_true == i).any() else 0.0
                for i, cls in enumerate(CLASSES)
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
        }
