"""
Evaluate the downloaded Ecoset ResNet50 checkpoint on the ecoset test split
and compare against the published Top-1 accuracy (66.7%, per the OSF wiki
project thingsvision points to for this checkpoint: "trained for 121 epochs,
66.7% top-1 on the ecoset test split").

Preprocessing matches thingsvision's default for this model (verified from
its source, thingsvision/core/extraction/base.py get_default_transformation):
    Resize(256) -> CenterCrop(224) -> ToTensor() ->
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

IMPORTANT class-index caveat:
The test folders are named "<numeric_id>_<name>" (e.g. "0009_car"), and the
numeric ids are NOT contiguous 0..564, so they are not directly usable as
class indices. There are (at least) two plausible orderings for what index
the checkpoint's fc layer actually uses for each class, and they disagree:
  (A) ImageFolder's default: classes sorted alphabetically by folder name,
      which (since ids are zero-padded to 4 digits) is equivalent to sorting
      by ascending numeric id.
  (B) The order of the `labels` list embedded in the official HuggingFace
      `kietzmannlab/ecoset` loader script (ecoset.py), saved here as
      ecoset_hf_labels.json - a different, non-numeric order.
Nothing in the checkpoint or thingsvision's code documents which one (if
either) matches training. So this script scores BOTH hypotheses and reports
which one (if any) lands near the published 66.7% - that agreement is the
actual evidence for correctness, not an assumption.

Run after `download_ecoset_test.py` has populated $ECOSET_DATA_DIR/test_data/<class>/<img>.

Usage:
    python3 eval_ecoset_resnet50.py
"""

import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm

DATA_DIR = os.environ.get("ECOSET_DATA_DIR", "/home/doju/data/ecoset")
MODEL_DIR = os.environ.get("ECOSET_MODEL_DIR", "/home/doju/pretrained_models/ecoset")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "resnet50_ecoset.pth")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test_data")
HF_LABELS_PATH = os.path.join(DATA_DIR, "ecoset_hf_labels.json")
PUBLISHED_TOP1 = 0.667

NUM_CLASSES = 565


def build_model(device):
    model = resnet50(weights=None, num_classes=NUM_CLASSES)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)  # strict by default - will raise if it doesn't match cleanly
    model.to(device)
    model.eval()
    return model


def build_dataloader(batch_size=64, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
    if len(dataset.classes) != NUM_CLASSES:
        print(
            f"WARNING: found {len(dataset.classes)} classes in {TEST_DATA_DIR}, "
            f"expected {NUM_CLASSES}."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset


def build_label_remap(dataset):
    """Map ImageFolder's target index -> hypothesis-B (HF labels list) index,
    via the shared human-readable class suffix (folder name minus numeric id)."""
    with open(HF_LABELS_PATH) as f:
        hf_labels = json.load(f)
    name_to_hf_idx = {name: i for i, name in enumerate(hf_labels)}

    remap = torch.empty(len(dataset.classes), dtype=torch.long)
    for imgfolder_idx, folder_name in enumerate(dataset.classes):
        suffix = folder_name.split("_", 1)[1]
        remap[imgfolder_idx] = name_to_hf_idx[suffix]
    return remap


@torch.no_grad()
def evaluate(model, loader, device, label_remap):
    correct_top1_a = correct_top1_b = 0
    correct_top5_a = correct_top5_b = 0
    total = 0
    label_remap = label_remap.to(device)
    for images, labels_a in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        labels_a = labels_a.to(device)
        labels_b = label_remap[labels_a]

        logits = model(images)
        top5 = logits.topk(5, dim=1).indices

        correct_top1_a += (top5[:, 0] == labels_a).sum().item()
        correct_top1_b += (top5[:, 0] == labels_b).sum().item()
        correct_top5_a += (top5 == labels_a.unsqueeze(1)).any(dim=1).sum().item()
        correct_top5_b += (top5 == labels_b.unsqueeze(1)).any(dim=1).sum().item()
        total += labels_a.size(0)

    return {
        "A_imagefolder_order": (correct_top1_a / total, correct_top5_a / total),
        "B_hf_labels_order": (correct_top1_b / total, correct_top5_b / total),
        "total": total,
    }


def main():
    if not os.path.isdir(TEST_DATA_DIR) or not os.listdir(TEST_DATA_DIR):
        raise SystemExit(f"{TEST_DATA_DIR} is empty. Run download_ecoset_test.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device)
    loader, dataset = build_dataloader()
    label_remap = build_label_remap(dataset)
    print(f"Loaded {len(dataset)} test images across {len(dataset.classes)} classes")

    results = evaluate(model, loader, device, label_remap)
    total = results["total"]
    print(f"\nEvaluated on {total} images (published top-1 reference: {PUBLISHED_TOP1 * 100:.1f}%)\n")

    for hyp_name, (top1, top5) in (
        ("A) ImageFolder alphabetical/numeric-id order", results["A_imagefolder_order"]),
        ("B) HF ecoset.py `labels` list order", results["B_hf_labels_order"]),
    ):
        delta = (top1 - PUBLISHED_TOP1) * 100
        print(f"{hyp_name}:")
        print(f"    top-1 = {top1 * 100:.2f}%  top-5 = {top5 * 100:.2f}%  (delta vs published: {delta:+.2f} pp)")

    best_hyp, (best_top1, _) = max(
        [("A", results["A_imagefolder_order"]), ("B", results["B_hf_labels_order"])],
        key=lambda kv: kv[1][0],
    )
    print(
        f"\n=> Hypothesis {best_hyp} scores closer to the published number "
        f"({best_top1*100:.2f}% top-1). If neither hypothesis lands within a "
        f"few points of 66.7%, the class-index mapping is still wrong and "
        f"needs further investigation before trusting this checkpoint's outputs."
    )


if __name__ == "__main__":
    main()
