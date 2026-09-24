import os
import unittest

import numpy as np
import torch
from PIL import Image

from intuitions.prepare_chexpert import relative_image_path, target_labels
from intuitions.targets import crc
from intuitions.targets.chexpert import BENCHMARK_CSV, IMAGE_SIZE, PATHOLOGIES, CheXpert, sample_label_balanced
from intuitions.targets.chexpert import get_transforms as chexpert_transforms
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


class TestCheXpertLabelRules(unittest.TestCase):
    def row(self, **values):
        return {p: values.get(p, "") for p in PATHOLOGIES}

    def test_uncertain_target_label_excludes_the_image(self):
        self.assertIsNone(target_labels(self.row(Edema="1.0", Cardiomegaly="-1.0")))

    def test_blank_counts_as_negative_and_a_positive_is_required(self):
        labels = target_labels(self.row(Edema="1.0", Cardiomegaly="0.0"))
        self.assertEqual(labels, [int(p == "Edema") for p in PATHOLOGIES])
        self.assertIsNone(target_labels(self.row(Cardiomegaly="0.0")))

    def test_official_paths_map_to_local_split_folders(self):
        self.assertEqual(relative_image_path("CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg"),
                         "val/patient64541/study1/view1_frontal.jpg")
        self.assertEqual(relative_image_path("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"),
                         "train/patient00001/study1/view1_frontal.jpg")
        self.assertEqual(relative_image_path("test/patient64741/study1/view1_frontal.jpg"),
                         "test/patient64741/study1/view1_frontal.jpg")

    def test_train_sampling_is_label_balanced_without_repeats(self):
        rng = np.random.default_rng(0)
        pool = [{"patient": f"p{i}", "rel_path": f"img{i:05d}", "_labels": list(rng.integers(0, 2, 8))}
                for i in range(5000)]
        sample = sample_label_balanced(pool, seed=0, per_label=40)
        self.assertEqual(len(sample), 8 * 40)
        self.assertEqual(len({r["rel_path"] for r in sample}), 8 * 40)
        for k in range(8):
            self.assertGreaterEqual(sum(r["_labels"][k] for r in sample), 40)
            self.assertTrue(all(r["_labels"][k] for r in sample[40 * k:40 * (k + 1)]))
        self.assertEqual(sample, sample_label_balanced(pool, seed=0, per_label=40))
        self.assertNotEqual(sample, sample_label_balanced(pool, seed=1, per_label=40))


@unittest.skipUnless(HAS_CHEXPERT, "CheXpert benchmark CSV not found")
class TestCheXpert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = CheXpert()

    def test_pools_and_fixed_test_set(self):
        self.assertEqual((len(self.target.val_pool), len({r["patient"] for r in self.target.val_pool})), (225, 196))
        self.assertEqual(len(self.target.test), 434)
        self.assertEqual(self.target.test, CheXpert().test)

    def test_partitions_are_patient_disjoint(self):
        partitions = [self.target.train_pool, self.target.val_pool, self.target.test]
        patients = [{r["patient"] for r in rows} for rows in partitions]
        self.assertFalse(patients[0] & patients[1])
        self.assertFalse(patients[0] & patients[2])
        self.assertFalse(patients[1] & patients[2])

    @unittest.skipUnless(os.path.exists(BENCHMARK_CSV.parent / "train.csv"), "official train.csv not found")
    def test_split_sizes_and_resampling(self):
        train3, val3, test3 = self.target.split(3)
        train4, val4, test4 = self.target.split(4)
        self.assertEqual((len(train3), len(val3), len(test3)), (320, 80, 434))
        self.assertEqual(len(train3) + len(val3) + len(test3), 834)  # the case study's subset size
        self.assertNotEqual(train3, train4)
        self.assertNotEqual(val3, val4)
        self.assertEqual(test3, test4)
        self.assertFalse({r["patient"] for r in train3} & {r["patient"] for r in val3})

    def test_item_ids_are_unique_relative_paths_grouped_by_patient(self):
        ids, groups = self.target.item_ids(self.target.test)
        self.assertEqual(len(set(ids)), len(self.target.test))
        self.assertTrue(all((BENCHMARK_CSV.parent / i).exists() for i in ids[:20]))
        self.assertEqual(groups, [r["patient"] for r in self.target.test])

    def test_transforms_return_unit_range_tensors(self):
        img = Image.open(self.target.test[0]["image_path"]).convert("RGB")
        for train in (True, False):
            x = chexpert_transforms(train=train)(img)
            self.assertEqual(tuple(x.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
            self.assertTrue(0.0 <= x.min().item() and x.max().item() <= 1.0)

    def test_label_smoothing_loss(self):
        crit = self.target.make_criterion({"label_smoothing": 0.0})
        logits, labels = torch.randn(4, 8), torch.randint(0, 2, (4, 8)).float()
        torch.testing.assert_close(crit(logits, labels), torch.nn.BCEWithLogitsLoss()(logits, labels))


if __name__ == "__main__":
    unittest.main()
