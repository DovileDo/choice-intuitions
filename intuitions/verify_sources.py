"""
Check that every source model is loaded and preprocessed correctly.

1. Input normalization: on source-domain images, conv1 output statistics should
   be closest to bn1's running statistics under the normalization the loader
   uses (lower mismatch = better).
2. Source-task accuracy through load_source_model:
   - Ecoset DVD models on the Ecoset test split (reference 63.10% / 50.34% top-1)
   - ImageNet logits identical to torchvision's reference model
   - RadImageNet test split with the head from `python -m intuitions.radimagenet_probe`
     (reference 74.32% top-1)

    python -m intuitions.verify_sources [--skip-accuracy]

Writes runs/source_checks/verify_sources.{log,json}.
"""

import argparse
import json
import logging
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50

from intuitions.paths import DATA_DIR, RUNS_DIR
from intuitions.radimagenet_probe import HEAD_PATH, RadImageNetDataset, radimagenet_labels, topk_accuracy
from intuitions.source_models import SOURCES, load_source_model, read_checkpoint
from intuitions.utils import git_revision, pick_device, setup_logging

log = logging.getLogger("intuitions.verify_sources")

OUT_DIR = RUNS_DIR / "source_checks"
ECOSET_TEST_DIR = DATA_DIR / "ecoset" / "test_data"
REFERENCE_TOP1 = {"ecoset_baseline": 0.6310, "ecoset_dvd_s": 0.5034, "radimagenet": 0.7432}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CAFFE_BGR_MEAN = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
CANDIDATE_NORMALIZATIONS = {
    "RGB, ImageNet mean/std": lambda x: (x - IMAGENET_MEAN) / IMAGENET_STD,
    "RGB, [0, 1]": lambda x: x,
    "RGB, [-1, 1]": lambda x: (x - 0.5) / 0.5,
    "BGR, [-1, 1]": lambda x: (x.flip(1) - 0.5) / 0.5,
    "BGR, 0-255 minus caffe mean": lambda x: x.flip(1) * 255 - CAFFE_BGR_MEAN,
}


class CenterCropSquare:
    def __call__(self, img):
        side = min(img.size)
        return transforms.functional.center_crop(img, [side, side])


def ecoset_test_set():
    # DVD models were trained on square 256px Ecoset images; ImageFolder order matches their labels.
    tf = transforms.Compose([CenterCropSquare(), transforms.Resize((256, 256)), transforms.ToTensor()])
    return ImageFolder(ECOSET_TEST_DIR, transform=tf)


def sample_images(dataset, n=512, seed=0):
    idx = random.Random(seed).sample(range(len(dataset)), n)
    return torch.cat([x for x, _ in DataLoader(Subset(dataset, idx), batch_size=128, num_workers=8)])


@torch.no_grad()
def bn1_mismatch(model, x, device):
    """Mean |z-score| of conv1 channel means plus mean |log variance ratio| vs bn1 running stats."""
    y = model.conv1(x.to(device))
    mean, var = y.mean(dim=(0, 2, 3)), y.var(dim=(0, 2, 3))
    run_mean, run_var = model.bn1.running_mean, model.bn1.running_var + model.bn1.eps
    return ((mean - run_mean).abs() / run_var.sqrt()).mean().item() + (var / run_var).log().abs().mean().item()


def loader_normalization(source):
    spec = SOURCES[source]
    order = "BGR" if spec.bgr else "RGB"
    return f"{order}, mean={spec.mean}, std={spec.std}"


def check_normalization(device, samples):
    results = {}
    for source, domain in [("imagenet", "ecoset"), ("ecoset_baseline", "ecoset"),
                           ("ecoset_dvd_s", "ecoset"), ("radimagenet", "radimagenet")]:
        model = load_source_model(source, num_classes=2).to(device).eval()
        scores = {name: bn1_mismatch(model, fn(samples[domain]), device)
                  for name, fn in CANDIDATE_NORMALIZATIONS.items()}
        # Ties are expected for grayscale images (RadImageNet), where RGB and BGR are indistinguishable.
        best = " = ".join(name for name, score in scores.items() if score - min(scores.values()) < 1e-6)
        log.info("%s (loader: %s; %s images) -> best match: %s", source, loader_normalization(source), domain, best)
        for name, score in scores.items():
            log.info("    %-30s mismatch %7.3f", name, score)
        results[source] = {"loader": loader_normalization(source), "best_match": best, "mismatch": scores}
    return results


def check_accuracy(device, ecoset_test):
    results = {}
    for source in ("ecoset_baseline", "ecoset_dvd_s"):
        model = load_source_model(source, num_classes=len(ecoset_test.classes))
        state = read_checkpoint(source)
        model.fc.load_state_dict({"weight": state["fc.weight"], "bias": state["fc.bias"]})
        loader = DataLoader(ecoset_test, batch_size=128, num_workers=8, pin_memory=True)
        top1, top5 = topk_accuracy(model.to(device), loader, device)
        results[source] = {"top1": top1, "top5": top5, "reference_top1": REFERENCE_TOP1[source]}
        log.info("%s: Ecoset test top-1 %.2f%% top-5 %.2f%% (reference %.2f%%)",
                 source, top1 * 100, top5 * 100, REFERENCE_TOP1[source] * 100)

    x = sample_images(ecoset_test, n=256)
    ours = load_source_model("imagenet", num_classes=1000)
    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    ours.fc.load_state_dict(reference.fc.state_dict())
    with torch.no_grad():
        diff = (ours.to(device).eval()(x.to(device))
                - reference.to(device).eval()(((x - IMAGENET_MEAN) / IMAGENET_STD).to(device))).abs().max().item()
    results["imagenet"] = {"max_abs_logit_diff_vs_torchvision": diff}
    log.info("imagenet: max |logit difference| vs torchvision reference %.2e", diff)

    if HEAD_PATH.exists():
        labels = radimagenet_labels()
        model = load_source_model("radimagenet", num_classes=len(labels))
        model.fc.load_state_dict(torch.load(HEAD_PATH, map_location="cpu", weights_only=True))
        test = RadImageNetDataset("test", {l: i for i, l in enumerate(labels)})
        top1, top5 = topk_accuracy(model.to(device), DataLoader(test, batch_size=256, num_workers=8, pin_memory=True), device)
        results["radimagenet"] = {"top1": top1, "top5": top5, "reference_top1": REFERENCE_TOP1["radimagenet"]}
        log.info("radimagenet: RadImageNet test top-1 %.2f%% top-5 %.2f%% (reference %.2f%%)",
                 top1 * 100, top5 * 100, REFERENCE_TOP1["radimagenet"] * 100)
    else:
        log.warning("radimagenet: %s not found; run `python -m intuitions.radimagenet_probe` first", HEAD_PATH)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-accuracy", action="store_true", help="only run the (fast) normalization check")
    args = parser.parse_args()

    setup_logging(OUT_DIR / "verify_sources.log")
    device = pick_device()
    log.info("git %s | device %s", git_revision(), device)

    ecoset_test = ecoset_test_set()
    labels = radimagenet_labels()
    samples = {"ecoset": sample_images(ecoset_test),
               "radimagenet": sample_images(RadImageNetDataset("test", {l: i for i, l in enumerate(labels)}))}
    results = {"normalization": check_normalization(device, samples)}
    if not args.skip_accuracy:
        results["accuracy"] = check_accuracy(device, ecoset_test)
    (OUT_DIR / "verify_sources.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
