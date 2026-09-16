"""
Evaluate RadImageNet ResNet50: freeze backbone, train FC (2048->165), report accuracy.
Published reference: top-1=72.3%, top-5=94.1%.

Usage:
    python3 eval_radimagenet_resnet50.py
"""

import csv
import json
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

DATA_DIR = os.environ.get("RADIMAGENET_DATA_DIR", "/home/doju/data/radiology_ai")
MODEL_DIR = os.environ.get("RADIMAGENET_MODEL_DIR", "/home/doju/pretrained_models/radimagenet")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "RadImageNet_ResNet50_pytorch.pt")
RESULTS_DIR = os.path.join(DATA_DIR, "eval_results")


class CSVDataset(Dataset):
    def __init__(self, csv_path, root, transform, label_to_idx):
        self.root = root
        self.transform = transform
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.samples.append((row["filename"], label_to_idx[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        tensor = self.transform(img)
        # model was trained with cv2.imread (BGR); flip R<->B
        return tensor[[2, 1, 0]], label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # label map
    labels_set = set()
    with open(os.path.join(DATA_DIR, "RadiologyAI_train.csv")) as f:
        for row in csv.DictReader(f):
            labels_set.add(row["label"])
    sorted_labels = sorted(labels_set)
    label_to_idx = {l: i for i, l in enumerate(sorted_labels)}
    num_classes = len(sorted_labels)
    print(f"classes: {num_classes}")

    # model
    model = resnet50(weights=None)
    backbone = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
    )
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items()}, strict=True)
    del sd

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    fc = nn.Linear(2048, num_classes)
    net = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc).to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    root = os.path.dirname(DATA_DIR)
    train_ds = CSVDataset(os.path.join(DATA_DIR, "RadiologyAI_train.csv"), root, transform, label_to_idx)
    val_ds = CSVDataset(os.path.join(DATA_DIR, "RadiologyAI_val.csv"), root, transform, label_to_idx)
    test_ds = CSVDataset(os.path.join(DATA_DIR, "RadiologyAI_test.csv"), root, transform, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.SGD(fc.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()

    # train with per-epoch val tracking and best-checkpoint saving
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(10):
        fc.train()
        backbone.eval()
        correct = total = 0
        running_loss = 0.0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feat = backbone(images)
            logits = fc(nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        train_acc = correct / total

        # val accuracy
        net.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = net(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"epoch {epoch+1:2d}: loss={running_loss/total:.4f} "
              f"train={train_acc*100:.2f}% val={val_acc*100:.2f}% time={time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(fc.state_dict(), os.path.join(RESULTS_DIR, "linear_head_best.pt"))

    print(f"\nbest val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
    fc.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "linear_head_best.pt"), map_location=device))

    # eval with best checkpoint
    results = {}
    net.eval()
    for name, loader in [("val", val_loader), ("test", test_loader)]:
        correct = correct5 = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = net(images)
                correct += (logits.argmax(1) == labels).sum().item()
                top5 = logits.topk(5, dim=1).indices
                correct5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
                total += labels.size(0)
        top1 = correct / total
        top5_acc = correct5 / total
        print(f"\n{name}: top-1={top1*100:.2f}%  top-5={top5_acc*100:.2f}%")
        results[name] = {"top1": round(top1, 5), "top5": round(top5_acc, 5), "n": total}

    results["published_top1"] = 0.723
    results["published_top5"] = 0.941
    results["best_epoch"] = best_epoch
    results["best_val_acc"] = round(best_val_acc, 5)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {os.path.join(RESULTS_DIR, 'results.json')}")
    print(f"best checkpoint (epoch {best_epoch}) -> {os.path.join(RESULTS_DIR, 'linear_head_best.pt')}")
    print(f"published reference: top-1=72.3%, top-5=94.1%")


if __name__ == "__main__":
    main()
