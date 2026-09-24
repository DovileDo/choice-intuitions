"""
Build the CheXpert benchmark pool from the official train, validation and test splits.

Keeps images with at least one positive among the 8 target labels (the survey's
rule; frontal and lateral views). Official training labels come from an
automatic report labeler: images with an uncertain (-1) value in any of the 8
labels are excluded and blank labels count as negative. Writes
<data>/CheXpert/chexpert_benchmark.csv with a `split` column and image paths
relative to <data>/CheXpert (train/, val/, test/).

    python -m intuitions.prepare_chexpert
    python -m intuitions.prepare_chexpert --needed-train-images needed.txt   # images the seeds sample
    python -m intuitions.prepare_chexpert --image-roots "<...>/CheXpert-v1.0 batch 2 (train 1)" ...

Expects val_labels.csv, test_labels.csv and (for training) train.csv in <data>/CheXpert.
Images are looked up under <data>/CheXpert; --image-roots adds folders to search for
them (e.g. the CheXpert batch folders on a cluster), and images found there are stored
with their absolute path, so the data need not be copied.
"""

import argparse
import csv
from pathlib import Path

from intuitions.targets.chexpert import BENCHMARK_CSV, DATA_ROOT, PATHOLOGIES

LABEL_FILES = {"train": "train.csv", "val": "val_labels.csv", "test": "test_labels.csv"}


def read_labels(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_patient(orig_path):
    return next(part for part in orig_path.split("/") if part.startswith("patient"))


def relative_image_path(orig_path):
    """Official CSV path -> path under <data>/CheXpert, e.g. "CheXpert-v1.0/valid/..." -> "val/..."."""
    parts = orig_path.split("/")
    if parts[0].startswith("CheXpert-v1.0"):
        parts = parts[1:]
    if parts[0] == "valid":
        parts[0] = "val"
    return "/".join(parts)


def target_labels(row):
    """0/1 labels for the 8 targets, or None if the image is not eligible."""
    labels = []
    for p in PATHOLOGIES:
        value = float(row[p]) if row[p].strip() else 0.0
        if value == -1.0:
            return None
        labels.append(int(value == 1.0))
    return labels if any(labels) else None


def index_patients(image_roots):
    """patient id -> directory holding its studies, searched a few levels down in each root."""
    index = {}
    for root in image_roots:
        # Stop at the first depth that holds patients: descending further would walk every
        # study and image below them, which costs minutes per root on a shared filesystem.
        for depth in ("patient*", "*/patient*", "*/*/patient*", "*/*/*/patient*"):
            found = [d for d in root.glob(depth) if d.is_dir()]
            for patient_dir in found:
                index.setdefault(patient_dir.name, patient_dir)
            if found:
                break
    if image_roots:
        print(f"indexed {len(index)} patients under {len(image_roots)} image root(s)")
    return index


def resolve_image(rel, patient_index):
    """Path under <data>/CheXpert if present, else an absolute path from the image roots."""
    if (DATA_ROOT / rel).exists():
        return rel
    parts = Path(rel).parts
    patient = next((p for p in parts if p.startswith("patient")), None)
    patient_dir = patient_index.get(patient)
    if patient_dir is None:
        return None
    path = patient_dir.joinpath(*parts[parts.index(patient) + 1:])
    return str(path) if path.exists() else None


def build_rows(patient_index):
    out_rows = []
    for split, filename in LABEL_FILES.items():
        path = DATA_ROOT / filename
        if not path.exists():
            print(f"{split}: {path} not found, skipped")
            continue
        rows = read_labels(path)
        kept, missing = [], 0
        for row in rows:
            labels = target_labels(row)
            if labels is None:
                continue
            rel = relative_image_path(row["Path"])
            found = resolve_image(rel, patient_index)
            # Official val/test images must be on disk; training images may be fetched selectively later.
            if split != "train" and found is None:
                missing += 1
                continue
            kept.append({"image_path": found or rel, "patient": get_patient(row["Path"]), "split": split,
                         **dict(zip(PATHOLOGIES, labels))})
        on_disk = sum((DATA_ROOT / r["image_path"]).exists() for r in kept)
        print(f"{split}: {len(rows)} -> {len(kept)} eligible images, {len({r['patient'] for r in kept})} patients, "
              f"{on_disk} on disk" + (f" (WARNING: {missing} listed images missing, dropped)" if missing else ""))
        out_rows.extend(kept)
    return out_rows


def write_needed_train_images(out_path, seeds):
    from intuitions.targets.chexpert import CheXpert
    target = CheXpert()
    needed = sorted({r["rel_path"] for seed in seeds for r in target.split(seed)[0]})
    out_path.write_text("\n".join(needed) + "\n")
    missing = sum(not (DATA_ROOT / p).exists() for p in needed)
    print(f"{len(needed)} training images needed for seeds {seeds[0]}-{seeds[-1]} -> {out_path} ({missing} not on disk)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--needed-train-images", type=Path,
                        help="also write the training image paths sampled by --seeds to this file")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    parser.add_argument("--image-roots", type=Path, nargs="+", default=[],
                        help="extra folders to search for images, e.g. the CheXpert batch folders")
    args = parser.parse_args()

    out_rows = build_rows(index_patients(args.image_roots))
    with open(BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "patient", "split", *PATHOLOGIES])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {BENCHMARK_CSV}")

    for split in LABEL_FILES:
        rows = [r for r in out_rows if r["split"] == split]
        if rows:
            print(f"  {split} prevalence: " + ", ".join(f"{p} {sum(r[p] for r in rows) / len(rows):.1%}" for p in PATHOLOGIES))

    if args.needed_train_images:
        write_needed_train_images(args.needed_train_images, args.seeds)


if __name__ == "__main__":
    main()
