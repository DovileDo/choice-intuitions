"""
Build the CheXpert benchmark pool from the official val and test splits.

Combines val + test and keeps images with at least one positive label among the
8 target pathologies (834 images, 662 patients). Writes
<data>/CheXpert/chexpert_benchmark.csv with image paths relative to <data>/CheXpert.

    python -m intuitions.prepare_chexpert
"""

import csv

from intuitions.targets.chexpert import BENCHMARK_CSV, DATA_ROOT, PATHOLOGIES


def read_labels(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_patient(orig_path):
    return next(part for part in orig_path.split("/") if part.startswith("patient"))


def relative_image_path(orig_path):
    # Val paths look like "CheXpert-v1.0/valid/patientX/...", stored locally under "val/".
    if "valid" in orig_path:
        return "/".join(["val", *orig_path.split("/")[2:]])
    return orig_path


def main():
    all_rows = read_labels(DATA_ROOT / "val_labels.csv") + read_labels(DATA_ROOT / "test_labels.csv")
    filtered = [row for row in all_rows if any(float(row[p]) == 1.0 for p in PATHOLOGIES)]
    print(f"Total: {len(all_rows)} -> filtered: {len(filtered)} images, "
          f"{len({get_patient(r['Path']) for r in filtered})} patients")

    out_rows, missing = [], 0
    for row in filtered:
        rel = relative_image_path(row["Path"])
        if not (DATA_ROOT / rel).exists():
            missing += 1
            continue
        out_rows.append({"image_path": rel, "patient": get_patient(row["Path"]),
                         **{p: int(float(row[p])) for p in PATHOLOGIES}})
    if missing:
        print(f"WARNING: {missing} images not found on disk")

    with open(BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "patient", *PATHOLOGIES])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {BENCHMARK_CSV}\nPathology prevalence:")
    for p in PATHOLOGIES:
        pos = sum(r[p] for r in out_rows)
        print(f"  {p}: {pos}/{len(out_rows)} ({pos / len(out_rows):.1%})")


if __name__ == "__main__":
    main()
