import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import torch

from intuitions.finetune import regime, run_final_eval, suggest_hparams
from intuitions.targets.crc import CLASSES, CRC, DATA_ROOT
from tests.test_targets import HAS_CRC

HPARAMS = {"backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.2, "batch_size": 16,
           "max_epochs": 2, "label_smoothing": 0.1, "stain_augmentation": True}


class TinyCRC(CRC):
    def split(self, seed):
        train, val, test = super().split(seed)
        return train[::50], val[::10], test[::100]


@unittest.skipUnless(HAS_CRC, "CRC data not found")
class TestFinalEval(unittest.TestCase):
    def test_tiny_end_to_end_run_saves_consistent_predictions(self):
        """Train/validate/test on a few patches; saved predictions must reproduce the reported test metrics."""
        target = TinyCRC()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results = run_final_eval(target, "ecoset_dvd_s", out_dir, HPARAMS, device, seeds=(5,))
            seed_result = results["per_seed_results"][0]
            self.assertEqual(json.loads((out_dir / "results.json").read_text())["eval_seeds"], [5])
            self.assertEqual([h["epoch"] for h in seed_result["history"]], [1, 2])

            pred = np.load(out_dir / seed_result["predictions"])
            _, _, test = target.split(5)
            self.assertEqual(pred["y_prob"].shape, (len(test), len(CLASSES)))
            np.testing.assert_allclose(pred["y_prob"].sum(axis=1), 1.0, rtol=1e-5)
            self.assertEqual(pred["y_true"].tolist(), [label for _, label in test])
            self.assertEqual(pred["ids"].tolist(), [str(Path(p).relative_to(DATA_ROOT)) for p, _ in test])
            self.assertEqual(pred["label_names"].tolist(), CLASSES)

            recomputed = target.metrics(pred["y_true"], pred["y_prob"])
            for metric in target.summary_metrics:
                self.assertAlmostEqual(recomputed[metric], seed_result["test_metrics"][metric], places=10)

    def test_eval_seeds_can_be_added_later_with_the_same_hparams(self):
        target = TinyCRC()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = run_final_eval(target, "ecoset_dvd_s", out_dir, HPARAMS, device, seeds=(5,))
            second = run_final_eval(target, "ecoset_dvd_s", out_dir, HPARAMS, device, seeds=(5, 6))
            self.assertEqual(second["eval_seeds"], [5, 6])
            self.assertEqual(second["per_seed_results"][0], first["per_seed_results"][0])  # seed 5 not retrained
            self.assertEqual(json.loads((out_dir / "results.json").read_text())["eval_seeds"], [5, 6])
            with self.assertRaises(RuntimeError):
                run_final_eval(target, "ecoset_dvd_s", out_dir, {**HPARAMS, "head_lr": 1e-2}, device, seeds=(7,))
            self.assertFalse((out_dir / "seeds" / "seed_7.json").exists())


class TestSearchSpace(unittest.TestCase):
    def distributions(self, source):
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
        suggest_hparams(trial, TinyCRC(), source)
        return trial.distributions

    def test_pretrained_sources_keep_the_fine_tuning_space(self):
        for source in ("imagenet", "radimagenet", "ecoset_baseline", "ecoset_dvd_s"):
            d = self.distributions(source)
            self.assertEqual((d["backbone_lr"].low, d["backbone_lr"].high), (1e-5, 1e-3))
            self.assertEqual((d["weight_decay"].low, d["weight_decay"].high), (1e-5, 1e-2))
            self.assertEqual(list(d["max_epochs"].choices), [20, 30, 50])
            self.assertEqual((regime(source)["early_stop_patience"], regime(source)["warmup_epochs"]), (7, 2))

    def test_scratch_gets_higher_learning_rates_and_longer_schedules(self):
        d = self.distributions("scratch")
        self.assertEqual((d["backbone_lr"].low, d["backbone_lr"].high), (1e-4, 1e-2))
        self.assertEqual((d["weight_decay"].low, d["weight_decay"].high), (1e-4, 1e-1))
        self.assertEqual(list(d["max_epochs"].choices), [50, 100, 150])
        self.assertEqual((regime("scratch")["early_stop_patience"], regime("scratch")["warmup_epochs"]), (15, 5))


class TestSingleFileResultsImport(unittest.TestCase):
    def test_results_json_without_seed_files_is_split_into_seed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            record = {"seed": 5, "test_metrics": {"balanced_accuracy": 0.5, "macro_f1": 0.4, "macro_auc": 0.9}}
            (out_dir / "results.json").write_text(json.dumps({"best_hparams": HPARAMS, "per_seed_results": [record]}))
            results = run_final_eval(TinyCRC(), "ecoset_dvd_s", out_dir, HPARAMS, torch.device("cpu"), seeds=())
            self.assertEqual(results["eval_seeds"], [5])
            self.assertAlmostEqual(results["summary"]["macro_auc"]["mean"], 0.9)
            self.assertTrue((out_dir / "seeds" / "seed_5.json").exists())


if __name__ == "__main__":
    unittest.main()
