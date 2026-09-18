"""
Linear probe of the RadImageNet backbone on the RadImageNet (RadiologyAI)
splits: frozen backbone, 165-way linear head trained for 10 epochs, best
validation epoch kept. Published reference: top-1 72.3%, top-5 94.1%.

    python -m intuitions.radimagenet_probe

Writes runs/source_checks/radimagenet_probe/: probe.log, linear_head_best.pt, results.json.
The head is reused by `python -m intuitions.verify_sources`.
"""

import csv
import json
import logging
import time

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from intuitions.paths import DATA_DIR, RUNS_DIR
from intuitions.source_models import load_source_model
from intuitions.utils import pick_device, setup_logging

log = logging.getLogger("intuitions.radimagenet_probe")

DATA_ROOT = DATA_DIR / "radiology_ai"
OUT_DIR = RUNS_DIR / "source_checks" / "radimagenet_probe"
HEAD_PATH = OUT_DIR / "linear_head_best.pt"
TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
EPOCHS = 10
PUBLISHED = {"top1": 0.723, "top5": 0.941}


def radimagenet_labels():
    with open(DATA_ROOT / "RadiologyAI_train.csv") as f:
        return sorted({row["label"] for row in csv.DictReader(f)})


class RadImageNetDataset(Dataset):
    def __init__(self, split, label_to_idx, transform=TRANSFORM):
        # CSV filenames are relative to the data dir, e.g. "radiology_ai/MR/knee/...".
        with open(DATA_ROOT / f"RadiologyAI_{split}.csv") as f:
            self.samples = [(row["filename"], label_to_idx[row["label"]]) for row in csv.DictReader(f)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(DATA_DIR / path).convert("RGB")), label


@torch.no_grad()
def topk_accuracy(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        top5 = model(images).topk(5, dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total += labels.numel()
    return correct1 / total, correct5 / total


def main():
    setup_logging(OUT_DIR / "probe.log")
    device = pick_device()
    labels = radimagenet_labels()
    label_to_idx = {l: i for i, l in enumerate(labels)}
    loaders = {
        split: DataLoader(RadImageNetDataset(split, label_to_idx), batch_size=256, shuffle=(split == "train"),
                          num_workers=4, pin_memory=True)
        for split in ("train", "val", "test")
    }
    log.info("%d classes; %s", len(labels), {s: len(l.dataset) for s, l in loaders.items()})

    model = load_source_model("radimagenet", num_classes=len(labels)).to(device)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.eval()  # frozen BatchNorm statistics; only the linear head trains
        t0, running_loss, n = time.time(), 0.0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.numel()
            n += targets.numel()
        scheduler.step()
        val_top1, _ = topk_accuracy(model, loaders["val"], device)
        log.info("epoch %d: loss %.4f val top-1 %.2f%% (%.0fs)", epoch, running_loss / n, val_top1 * 100,
                 time.time() - t0)
        if val_top1 > best_val:
            best_val, best_epoch = val_top1, epoch
            torch.save(model.fc.state_dict(), HEAD_PATH)

    model.fc.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    results = {"best_epoch": best_epoch, "published": PUBLISHED}
    for split in ("val", "test"):
        top1, top5 = topk_accuracy(model, loaders[split], device)
        results[split] = {"top1": top1, "top5": top5, "n": len(loaders[split].dataset)}
        log.info("%s: top-1 %.2f%% top-5 %.2f%%", split, top1 * 100, top5 * 100)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
