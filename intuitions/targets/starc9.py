"""
CS-tissue on STARC-9: 9-class colorectal H&E tissue classification (Subramanian et al. 2025),
with the same budget and training recipe as the CRC target.

Per data seed: 200 tiles per class from the STARC-9 training set for training and 50 per class
from STANFORD-CRC-HE-VAL-SMALL for validation, i.e. CRC's 250 per class split 80/20. Test: a
fixed class-balanced 4,930 tiles from STANFORD-CRC-HE-VAL-LARGE (50 slides), identical for
every seed and source, so the training budget plus the test set add up to the survey's
7,180-image target dataset. The training set and both Stanford validation sets come from
different patients (Subramanian et al. 2025), so all three partitions are patient-disjoint,
unlike CRC's patch-level train/val split.

Tiles are Macenko-normalised 256x256 PNGs at 0.25 um/px, resized to the 224x224 input used for
CRC. Expected layout, from the Hugging Face release at revision 06e8090 (see README):
<data>/STARC-9/{train,val_small,val_large}/<CLASS>/*.png.
"""

import os
import random

from torchvision import transforms

from intuitions.paths import DATA_DIR
from intuitions.targets.crc import CRC, HEDStainAugmentation, PatchDataset, RandomRotation90

DATA_ROOT = DATA_DIR / "STARC-9"
TRAIN_DIR = str(DATA_ROOT / "train")
VAL_DIR = str(DATA_ROOT / "val_small")
TEST_DIR = str(DATA_ROOT / "val_large")

CLASSES = sorted(["ADI", "BLD", "FCT", "LYM", "MUC", "MUS", "NCS", "NOR", "TUM"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

N_TRAIN_PER_CLASS = 200  # CRC's 250 per class, split 80/20 between training and validation
N_VAL_PER_CLASS = 50
N_TEST = 4930  # 7,180 minus the 2,250-tile training budget, as for CRC
TEST_HOLDOUT_SEED = 0
IMAGE_SIZE = 224


def list_class_files(split_dir):
    return {cls: sorted(os.path.join(split_dir, cls, f) for f in os.listdir(os.path.join(split_dir, cls))
                        if f.endswith(".png"))
            for cls in CLASSES}


def sample_per_class(split_dir, n_per_class, rng):
    class_files = list_class_files(split_dir)
    return [(f, CLASS_TO_IDX[cls]) for cls in CLASSES for f in rng.sample(class_files[cls], n_per_class)]


def sample_train_val(seed, train_dir=TRAIN_DIR, val_dir=VAL_DIR):
    """Training tiles from the STARC-9 training set, validation tiles from VAL-SMALL."""
    rng = random.Random(seed)
    return sample_per_class(train_dir, N_TRAIN_PER_CLASS, rng), sample_per_class(val_dir, N_VAL_PER_CLASS, rng)


def list_test_files(test_dir=TEST_DIR, n_test=N_TEST, seed=TEST_HOLDOUT_SEED):
    """A fixed class-balanced subset of VAL-LARGE (547 or 548 per class); the same for every data seed."""
    rng = random.Random(seed)
    class_files = list_class_files(test_dir)
    per_class, extra = divmod(n_test, len(CLASSES))
    test_files = []
    for i, cls in enumerate(CLASSES):
        chosen = rng.sample(class_files[cls], per_class + (i < extra))
        test_files.extend((f, CLASS_TO_IDX[cls]) for f in sorted(chosen))
    return test_files


def get_transforms(stain_aug=False, train=True):
    # As for CRC, after resizing the 256x256 tiles to 224x224.
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        t += [HEDStainAugmentation()] if stain_aug else []
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotation90(),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        ]
    return transforms.Compose(t + [transforms.ToTensor()])


class STARC9(CRC):
    """Same search space, loss and metrics as CRC; only the data differ."""

    name = "starc9"
    label_names = tuple(CLASSES)

    def split(self, seed):
        train, val = sample_train_val(seed)
        return train, val, list_test_files()

    def make_dataset(self, items, hparams, train):
        return PatchDataset(items, get_transforms(stain_aug=train and hparams["stain_augmentation"], train=train))

    def item_ids(self, items):
        # STARC-9 tiles carry no patient identifiers, so each tile is its own resampling unit.
        ids = [os.path.relpath(path, DATA_ROOT) for path, _ in items]
        return ids, ids
