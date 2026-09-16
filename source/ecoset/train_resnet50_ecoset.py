"""
Train ResNet50 on ecoset from scratch using the torchvision v2 recipe.
Reference: pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

Recipe: SGD lr=0.5 (auto-scaled by batch size), cosine LR + warmup,
TrivialAugmentWide, MixUp+CutMix, label smoothing, random erasing,
EMA, repeated augmentation, FixRes (train@176, eval@232->224).

Saves checkpoint every epoch for resumable HPC jobs.

Usage:
    python3 train_resnet50_ecoset.py --train-dir /data/ecoset/train --val-dir /data/ecoset/val
    python3 train_resnet50_ecoset.py --train-dir ... --val-dir ... --resume
    python3 train_resnet50_ecoset.py --resume --eval-only --test-dir /data/ecoset/test
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REF_BATCH = 1024
REF_LR = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./ecoset_resnet50_v2")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-GPU batch size. LR auto-scales from 0.5 @ batch 1024")
    p.add_argument("--lr", type=float, default=None,
                   help="Override auto-scaled LR")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=2e-5)
    p.add_argument("--norm-weight-decay", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--lr-warmup-epochs", type=int, default=5)
    p.add_argument("--lr-warmup-decay", type=float, default=0.01)
    p.add_argument("--train-crop-size", type=int, default=176)
    p.add_argument("--val-resize-size", type=int, default=232)
    p.add_argument("--val-crop-size", type=int, default=224)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--random-erase", type=float, default=0.1)
    p.add_argument("--ra-reps", type=int, default=1,
                   help="Repeated augmentation reps (1=off, 4=full recipe)")
    p.add_argument("--ema-decay", type=float, default=0.99998)
    p.add_argument("--ema-steps", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class RepeatAugSampler(Sampler):
    def __init__(self, dataset, num_repeats=4, seed=0):
        self.n = len(dataset)
        self.num_repeats = num_repeats
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.n, generator=g)
        return iter(order.repeat_interleave(self.num_repeats).tolist())

    def __len__(self):
        return self.n * self.num_repeats


class ModelEMA:
    def __init__(self, model, decay=0.99998):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.copy_(mb.data)

    def copy_from(self, model):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.copy_(mp.data)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.data.copy_(mb.data)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, device):
        self.module.load_state_dict(sd)
        self.module.to(device)


def apply_mixup_cutmix(images, targets, num_classes, mixup_alpha, cutmix_alpha):
    targets_oh = F.one_hot(targets, num_classes).float()
    idx = torch.randperm(images.size(0), device=images.device)

    if random.random() < 0.5:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[idx]
        targets_oh = lam * targets_oh + (1 - lam) * targets_oh[idx]
    else:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, H, W = images.shape
        r = math.sqrt(1 - lam)
        rh, rw = int(H * r), int(W * r)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
        x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_adj = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        targets_oh = lam_adj * targets_oh + (1 - lam_adj) * targets_oh[idx]

    return images, targets_oh


def smooth_cross_entropy(logits, targets, smoothing=0.1):
    C = logits.size(1)
    if targets.dim() == 1:
        targets = F.one_hot(targets, C).float()
    targets = targets * (1 - smoothing) + smoothing / C
    return -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()


def get_param_groups(model, wd, norm_wd):
    norm_ids = set()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for p in m.parameters():
                norm_ids.add(id(p))
    regular = [p for p in model.parameters() if id(p) not in norm_ids]
    norms = [p for p in model.parameters() if id(p) in norm_ids]
    groups = []
    if regular:
        groups.append({"params": regular, "weight_decay": wd})
    if norms:
        groups.append({"params": norms, "weight_decay": norm_wd})
    return groups


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == targets).sum().item()
        correct5 += (top5 == targets.unsqueeze(1)).any(1).sum().item()
        total += targets.size(0)
    return correct1 / total, correct5 / total


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, best_acc):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }, path)


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = REF_LR * args.batch_size / REF_BATCH
    use_amp = not args.no_amp

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=args.random_erase),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.val_resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.val_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # --- Data ---
    train_ds = ImageFolder(args.train_dir, transform=train_transform)
    val_ds = ImageFolder(args.val_dir, transform=val_transform)
    num_classes = len(train_ds.classes)

    if args.ra_reps > 1:
        sampler = RepeatAugSampler(train_ds, num_repeats=args.ra_reps, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    test_loader = None
    if args.test_dir:
        test_ds = ImageFolder(args.test_dir, transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # --- Model ---
    model = resnet50(weights=None, num_classes=num_classes).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)

    # --- Optimizer ---
    param_groups = get_param_groups(model, args.weight_decay, args.norm_weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[args.lr_warmup_epochs])

    scaler = GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Resume ---
    start_epoch = 0
    best_acc = 0.0
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"], device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        log(f"resumed from epoch {start_epoch}, best_acc={best_acc*100:.2f}%")
        del ckpt

    # --- Eval only ---
    if args.eval_only:
        best_path = os.path.join(args.output_dir, "best.pt")
        if os.path.exists(best_path):
            sd = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(sd["model"])
            ema.load_state_dict(sd["ema"], device)
        for name, m in [("raw", model), ("ema", ema.module)]:
            v1, v5 = evaluate(m, val_loader, device)
            log(f"[{name}] val: top1={v1*100:.2f}% top5={v5*100:.2f}%")
            if test_loader:
                t1, t5 = evaluate(m, test_loader, device)
                log(f"[{name}] test: top1={t1*100:.2f}% top5={t5*100:.2f}%")
        return

    # --- Log config ---
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    log(f"classes: {num_classes}, train: {len(train_ds)}, val: {len(val_ds)}")
    log(f"batch_size: {args.batch_size}, lr: {args.lr:.4f}, epochs: {args.epochs}")
    log(f"ra_reps: {args.ra_reps}, ema_decay: {args.ema_decay}, amp: {use_amp}")
    log(f"train_crop: {args.train_crop_size}, val_resize: {args.val_resize_size}")
    log(f"steps/epoch: {len(train_loader)}")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)

        running_loss = 0.0
        correct = total = 0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            orig_targets = targets

            images, targets_mixed = apply_mixup_cutmix(
                images, targets, num_classes, args.mixup_alpha, args.cutmix_alpha)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = smooth_cross_entropy(logits, targets_mixed, args.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % args.ema_steps == 0:
                if epoch < args.lr_warmup_epochs:
                    ema.copy_from(model)
                else:
                    ema.update(model)

            running_loss += loss.item() * images.size(0)
            with torch.no_grad():
                correct += (logits.argmax(1) == orig_targets).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        ema_val1, ema_val5 = evaluate(ema.module, val_loader, device)

        log(f"epoch {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
            f"train={train_acc*100:.1f}%  ema_val={ema_val1*100:.2f}%  "
            f"lr={lr_now:.6f}  time={elapsed:.0f}s")

        is_best = ema_val1 > best_acc
        if is_best:
            best_acc = ema_val1

        save_checkpoint(ckpt_path, model, ema, optimizer, scheduler, scaler,
                        epoch, best_acc)
        if is_best:
            save_checkpoint(os.path.join(args.output_dir, "best.pt"),
                            model, ema, optimizer, scheduler, scaler,
                            epoch, best_acc)

    log(f"\ntraining complete. best ema val top-1: {best_acc*100:.2f}%")

    # --- Final eval with best model ---
    best_path = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        ema.load_state_dict(sd["ema"], device)

    v1, v5 = evaluate(ema.module, val_loader, device)
    log(f"[best ema] val: top1={v1*100:.2f}% top5={v5*100:.2f}%")
    if test_loader:
        t1, t5 = evaluate(ema.module, test_loader, device)
        log(f"[best ema] test: top1={t1*100:.2f}% top5={t5*100:.2f}%")

    torch.save(ema.module.state_dict(), os.path.join(args.output_dir, "resnet50_ecoset.pth"))
    log(f"saved final weights -> {args.output_dir}/resnet50_ecoset.pth")


if __name__ == "__main__":
    main()
