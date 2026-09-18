import unittest

import torch
from torchvision.models import resnet50

from intuitions.source_models import SOURCES, is_pretrained, load_backbone_state, load_source_model

# Verified input conventions of each source's pretraining (see intuitions.verify_sources).
EXPECTED_INPUT = {
    "imagenet": (False, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "radimagenet": (True, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "ecoset_baseline": (False, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "ecoset_dvd_s": (False, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "scratch": (False, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def available_sources():
    return [s for s, spec in SOURCES.items() if spec.checkpoint is None or spec.checkpoint.exists()]


class TestSourceModels(unittest.TestCase):
    def test_all_sources_available(self):
        self.assertEqual(sorted(available_sources()), sorted(SOURCES))
        self.assertEqual(sorted(EXPECTED_INPUT), sorted(SOURCES))

    def test_scratch_is_random_and_seeded(self):
        torch.manual_seed(0)
        a = load_source_model("scratch", num_classes=3).state_dict()
        torch.manual_seed(0)
        b = load_source_model("scratch", num_classes=3).state_dict()
        imagenet = load_backbone_state("imagenet")
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(torch.equal(a["conv1.weight"], imagenet["conv1.weight"]))

    def test_backbone_matches_checkpoint(self):
        for source in filter(is_pretrained, available_sources()):
            with self.subTest(source=source):
                state = load_backbone_state(source)
                model_state = load_source_model(source, num_classes=3).state_dict()
                self.assertEqual(len(state), 318)
                for key, value in state.items():
                    self.assertTrue(torch.equal(model_state[key], value), key)

    def test_forward_equals_manual_normalization(self):
        """Model on [0, 1] input == plain torchvision ResNet-50 on manually normalized input."""
        torch.manual_seed(0)
        x = torch.rand(2, 3, 64, 64)
        for source in available_sources():
            with self.subTest(source=source):
                bgr, mean, std = EXPECTED_INPUT[source]
                model = load_source_model(source, num_classes=5).eval()
                reference = resnet50(num_classes=5).eval()
                reference.load_state_dict(model.state_dict())
                manual = (x.flip(1) if bgr else x) - torch.tensor(mean).view(1, 3, 1, 1)
                manual = manual / torch.tensor(std).view(1, 3, 1, 1)
                with torch.no_grad():
                    torch.testing.assert_close(model(x), reference(manual))

    def test_head(self):
        model = load_source_model("imagenet", num_classes=7, dropout=0.5)
        dropout, linear = model.fc
        self.assertEqual(dropout.p, 0.5)
        self.assertEqual(linear.out_features, 7)
        self.assertTrue(torch.all(linear.bias == 0))
        self.assertIsInstance(load_source_model("imagenet", num_classes=7).fc, torch.nn.Linear)

    def test_unknown_source(self):
        with self.assertRaises(ValueError):
            load_source_model("cifar", num_classes=2)


if __name__ == "__main__":
    unittest.main()
