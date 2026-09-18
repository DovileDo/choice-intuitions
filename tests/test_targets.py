import unittest

import numpy as np
import torch
from PIL import Image

from intuitions.targets import crc
from intuitions.targets.chexpert import BENCHMARK_CSV, CheXpert, get_transforms as chexpert_transforms
from intuitions.targets.crc import CRC, CLASSES, HEDStainAugmentation

HAS_CRC = (crc.DATA_ROOT / "train").is_dir() and (crc.DATA_ROOT / "test").is_dir()
HAS_CHEXPERT = BENCHMARK_CSV.exists()


class TestStainAugmentation(unittest.TestCase):
    def test_same_colour_maps_to_same_colour(self):
        """Stain jitter is one colour transform per image, not per-pixel noise."""
        np.random.seed(0)
        img = Image.new("RGB", (32, 32), (180, 90, 160))
        for _ in range(5):
            out = np.asarray(HEDStainAugmentation()(img)).reshape(-1, 3)
            self.assertEqual(len(np.unique(out, axis=0)), 1)

    def test_changes_colour_mildly(self):
        np.random.seed(0)
        img = Image.new("RGB", (8, 8), (180, 90, 160))
        diffs = [np.abs(np.asarray(HEDStainAugmentation()(img), dtype=float) - np.asarray(img)).mean()
                 for _ in range(20)]
        self.assertGreater(max(diffs), 0)
        self.assertLess(np.mean(diffs), 40)


@unittest.skipUnless(HAS_CRC, "CRC data not found")
class TestCRC(unittest.TestCase):
    def test_split_sizes_and_disjointness(self):
        train, val, test = CRC().split(seed=5)
        self.assertEqual(len(train), 200 * len(CLASSES))
        self.assertEqual(len(val), 50 * len(CLASSES))
        self.assertEqual(len(test), 7180 - 250 * len(CLASSES))
        self.assertFalse({f for f, _ in train} & {f for f, _ in val})
        full_test = crc.list_class_files(crc.TEST_DIR)
        for cls, idx in crc.CLASS_TO_IDX.items():
            self.assertEqual(sum(label == idx for _, label in train), 200, cls)
            self.assertEqual(sum(label == idx for _, label in test), len(full_test[cls]) - 250, cls)

    def test_split_is_deterministic_per_seed_with_a_fixed_test_set(self):
        target = CRC()
        self.assertEqual(target.split(3)[:2], target.split(3)[:2])
        self.assertNotEqual(target.split(3)[0], target.split(4)[0])
        self.assertEqual(target.split(3)[2], target.split(4)[2])

    def test_transforms_return_unit_range_tensors(self):
        path, _ = CRC().split(seed=0)[0][0]
        img = Image.open(path).convert("RGB")
        for train in (True, False):
            x = crc.get_transforms(stain_aug=True, train=train)(img)
            self.assertEqual(tuple(x.shape), (3, 224, 224))
            self.assertGreaterEqual(x.min().item(), 0.0)
            self.assertLessEqual(x.max().item(), 1.0)


@unittest.skipUnless(HAS_CHEXPERT, "CheXpert benchmark CSV not found")
class TestCheXpert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = CheXpert()

    def test_pool(self):
        self.assertEqual(len(self.target.rows), 834)
        self.assertEqual(len({r["patient"] for r in self.target.rows}), 662)

    def test_splits_are_patient_disjoint_and_complete(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                splits = self.target.split(seed)
                patients = [{r["patient"] for r in rows} for rows in splits]
                self.assertFalse(patients[0] & patients[1])
                self.assertFalse(patients[0] & patients[2])
                self.assertFalse(patients[1] & patients[2])
                self.assertEqual(sum(len(rows) for rows in splits), len(self.target.rows))
                self.assertGreaterEqual(len(splits[0]) + len(splits[1]), 250)

    def test_split_is_deterministic_per_seed(self):
        self.assertEqual(self.target.split(7), self.target.split(7))

    def test_item_ids_are_unique_relative_paths_grouped_by_patient(self):
        _, _, test = self.target.split(5)
        ids, groups = self.target.item_ids(test)
        self.assertEqual(len(set(ids)), len(test))
        self.assertTrue(all((BENCHMARK_CSV.parent / i).exists() for i in ids[:20]))
        self.assertEqual(groups, [r["patient"] for r in test])

    def test_transforms_return_unit_range_tensors(self):
        img = Image.open(self.target.rows[0]["image_path"]).convert("RGB")
        for train in (True, False):
            x = chexpert_transforms(train=train)(img)
            self.assertEqual(tuple(x.shape), (3, 224, 224))
            self.assertTrue(0.0 <= x.min().item() and x.max().item() <= 1.0)

    def test_label_smoothing_loss(self):
        crit = self.target.make_criterion({"label_smoothing": 0.0})
        logits, labels = torch.randn(4, 8), torch.randint(0, 2, (4, 8)).float()
        torch.testing.assert_close(crit(logits, labels), torch.nn.BCEWithLogitsLoss()(logits, labels))


if __name__ == "__main__":
    unittest.main()
