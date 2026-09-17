#!/usr/bin/env python3
"""
Build the CheXpert benchmark subset from the official val and test splits.

Combines val + test, keeps 8 pathologies with >=100 images, and filters to
images with at least one positive label among the 8. Produces a single CSV
(834 images, 662 patients) used by the benchmark script.

Usage:
    python prepare_chexpert.py
"""

import csv
import os

DATA_DIR = os.environ.get("CHEXPERT_DATA_DIR", "/home/doju/data/CheXpert")

TARGET_PATHOLOGIES = sorted([
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Support Devices",
])


def read_labels(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_patient(orig_path):
    for part in orig_path.split("/"):
        if part.startswith("patient"):
            return part
    return None


def resolve_image_path(orig_path, data_dir):
    if "valid" in orig_path:
        parts = orig_path.split("/")
        rel = os.path.join("val", *parts[2:])
    else:
        rel = orig_path
    return os.path.join(data_dir, rel)


def main():
    val = read_labels(os.path.join(DATA_DIR, "val_labels.csv"))
    test = read_labels(os.path.join(DATA_DIR, "test_labels.csv"))
    all_rows = val + test

    filtered = [
        row for row in all_rows
        if any(float(row[p]) == 1.0 for p in TARGET_PATHOLOGIES)
    ]

    patients = set(get_patient(row["Path"]) for row in filtered)
    print(f"Total: {len(all_rows)} -> filtered: {len(filtered)} images, "
          f"{len(patients)} patients")

    missing = 0
    out_rows = []
    for row in filtered:
        img_path = resolve_image_path(row["Path"], DATA_DIR)
        if not os.path.exists(img_path):
            missing += 1
            continue
        patient = get_patient(row["Path"])
        out_row = {"image_path": img_path, "patient": patient}
        for p in TARGET_PATHOLOGIES:
            out_row[p] = int(float(row[p]))
        out_rows.append(out_row)

    if missing:
        print(f"WARNING: {missing} images not found on disk")

    out_path = os.path.join(DATA_DIR, "chexpert_benchmark.csv")
    fieldnames = ["image_path", "patient"] + TARGET_PATHOLOGIES
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"\nPathology prevalence:")
    for p in TARGET_PATHOLOGIES:
        pos = sum(r[p] for r in out_rows)
        print(f"  {p}: {pos}/{len(out_rows)} ({pos / len(out_rows):.1%})")


if __name__ == "__main__":
    main()
