"""
Pilot training run to (a) validate the ResNet50-on-ecoset training loop works
end-to-end on this GB10 GPU, and (b) measure real images/sec throughput so we
can estimate how long a full run (torchvision V1-style recipe: ~90 epochs
over the full ~1.45M-image train split) would actually take on this hardware.

Uses the existing local test_data/ (28,250 images, 565 classes) as a stand-in
dataset purely for timing/mechanics - this is NOT a real training run and the
resulting weights are meaningless. Once $ECOSET_DATA_DIR/full_dataset/ finishes downloading,
the real run will point at the actual train/ split instead.

Recipe follows torchvision's classic "V1" ImageNet ResNet50 training recipe:
  SGD, momentum=0.9, weight_decay=1e-4, lr=0.1 (for batch size 256, scaled
  linearly for the batch size actually used), step LR decay /10 every 30
  epochs, standard augmentation (RandomResizedCrop(224) + RandomHorizontalFlip).
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

ECOSET_DATA_DIR = os.environ.get("ECOSET_DATA_DIR", os.path.expanduser("~/data/ecoset"))
DATA_DIR = os.path.join(ECOSET_DATA_DIR, "test_data")
NUM_CLASSES = 565
BATCH_SIZE = 256
NUM_WORKERS = 16
PILOT_STEPS = 50  # just enough to warm up + get a stable throughput estimate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(DATA_DIR, transform=train_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    print(f"pilot dataset: {len(dataset)} images, {len(dataset.classes)} classes, "
          f"{len(loader)} batches/epoch at batch size {BATCH_SIZE}")

    model = resnet50(weights=None, num_classes=NUM_CLASSES).to(device)
    model.train()

    base_lr = 0.1 * BATCH_SIZE / 256
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    step_times = []
    images_seen = 0
    t_start = time.time()

    for step, (images, labels) in enumerate(loader):
        if step >= PILOT_STEPS:
            break
        t0 = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        # skip the first couple of steps (cudnn autotune / warmup) for the estimate
        if step >= 3:
            step_times.append(dt)
        images_seen += images.size(0)
        if step % 10 == 0:
            print(f"step {step:3d}  loss={loss.item():.3f}  step_time={dt*1000:.0f}ms")

    total_time = time.time() - t_start
    avg_step = sum(step_times) / len(step_times)
    imgs_per_sec = BATCH_SIZE / avg_step

    print(f"\n=== Pilot summary ===")
    print(f"steady-state step time: {avg_step*1000:.1f} ms/step (batch={BATCH_SIZE})")
    print(f"throughput: {imgs_per_sec:.1f} images/sec")

    n_train_full = 1_445_477  # from the remote zip's central directory listing
    epoch_time_sec = n_train_full / imgs_per_sec
    print(f"\nEstimated time per epoch on full train split ({n_train_full} images): "
          f"{epoch_time_sec/60:.1f} min")
    for n_epochs in (30, 90):
        total_h = epoch_time_sec * n_epochs / 3600
        print(f"Estimated total time for {n_epochs} epochs: {total_h:.1f} hours (~{total_h/24:.1f} days)")


if __name__ == "__main__":
    main()
