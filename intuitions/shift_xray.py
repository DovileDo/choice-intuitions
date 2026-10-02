"""
Robustness of the fine-tuned CS-xray (CheXpert) models to distribution shift: every saved evaluation
model (runs/benchmark/chexpert/<source>/models/seed_<N>.pt) is tested on chest X-rays from other
hospitals, scored by macro-AUC over the labels each dataset shares with our eight CheXpert labels.

  - ChestX-ray14 (NIH): the 810 PA images of the official test split with Google's expert labels
    ("all findings", Nabulsi et al. 2021: five radiologists, majority of three; --nih-labels) rather
    than NIH's labels text-mined from the reports. Shared labels: No Finding (not "Abnormal"),
    Cardiomegaly, Edema, Atelectasis, Pleural Effusion ("Effusion").
  - PadChest: physician-labelled, PA/AP/lateral views, all ages. The same five labels plus Support
    Devices, mapped from PadChest's terms by keyword (the matched terms are printed for auditing).
    The 16-bit images are rescaled per image to 8 bits (0.5-99.5th percentile) and inverted when
    stored as MONOCHROME1.

As in the benchmark, only images with at least one positive shared label are used. Images are
prepared as in training (RGB, resized to 320x320). The in-distribution reference is the CheXpert
test set restricted to the same labels, from the saved test predictions. A sheet of example images
is written for checking the conversion by eye.

Every image's sex and age are kept for the fairness comparison (intuitions.fairness): ChestX-ray14's
Patient Gender and Patient Age (the few impossible ages, above 120, count as unknown), PadChest's
PatientSex_DICOM and the study year minus PatientBirth (PadChest gives only the year of birth, so
ages are accurate to a year). For PadChest the scanner is kept too, its Manufacturer_DICOM and
Modality_DICOM (CR or DX); ChestX-ray14 records neither.

    python -m intuitions.shift_xray
    python -m intuitions.shift_xray --limit 200 --sources imagenet      # quick check

Writes runs/shift/chexpert/
  per_model.json                         macro-AUC and per-label AUCs per model
  examples.png                           example images per dataset
  predictions/<dataset>_labels.npz       y_true (-1 for labels the dataset lacks), scored (the shared
                                         labels), label_names, ids (file names), sex, age, manufacturer,
                                         modality ("" where unknown)
  predictions/<source>/seed_<N>_<dataset>.npy   y_prob of eval seed N's model, rows as in the labels file
--limit runs save only the examples.
"""

import argparse
import ast
import csv
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from intuitions.names import SOURCE_NAMES
from intuitions.paths import DATA_DIR, RUNS_DIR
from intuitions.source_models import load_source_model
from intuitions.targets.chexpert import IMAGE_SIZE, PATHOLOGIES

# CheXpert label -> predicate on a row of the expert label file (YES / NO per finding).
NIH_LABELS = {
    "No Finding": lambda r: r["Abnormal"] == "NO",
    "Cardiomegaly": lambda r: r["Cardiomegaly"] == "YES",
    "Edema": lambda r: r["Edema"] == "YES",
    "Atelectasis": lambda r: r["Atelectasis"] == "YES",
    "Pleural Effusion": lambda r: r["Effusion"] == "YES",
}
# CheXpert label -> predicate on a (lower-case) PadChest term.
PADCHEST_LABELS = {
    "No Finding": lambda t: t == "normal",
    "Cardiomegaly": lambda t: t == "cardiomegaly",
    "Edema": lambda t: "edema" in t,
    "Atelectasis": lambda t: "atelectasis" in t,
    "Pleural Effusion": lambda t: "pleural effusion" in t,
    "Support Devices": lambda t: any(k in t for k in ("catheter", "tube", "pacemaker", "electrical device",
                                                        "chamber device", "drain", "prosthesis valve")),
}
PADCHEST_VIEWS = {"PA", "AP", "AP_horizontal", "L"}
SEX = {"F": "female", "M": "male"}  # anything else is unknown ("")


def years(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def known(text):
    text = (text or "").strip()
    return "" if text in ("None", "nan") else text


def nih_attributes(row):
    age = years(row["Patient Age"])
    return {"sex": SEX.get(row["Patient Gender"], ""), "age": age if age <= 120 else np.nan,
            "manufacturer": "", "modality": ""}


def padchest_attributes(row):
    return {"sex": SEX.get(row["PatientSex_DICOM"], ""),
            "age": years(row["StudyDate_DICOM"][:4]) - years(row["PatientBirth"]),
            "manufacturer": known(row["Manufacturer_DICOM"]), "modality": known(row["Modality_DICOM"])}


def nih_items(root, expert_labels):
    """(path, {CheXpert label: 0/1}, conversion info, attributes) for the expert-labelled ChestX-ray14
    images with at least one positive shared label; sex and age from Data_Entry_2017.csv."""
    with open(root / "Data_Entry_2017.csv") as f:
        entries = {row["Image Index"]: row for row in csv.DictReader(f)}
    paths = {p.name: p for p in root.glob("images*/**/*.png")}
    with open(expert_labels) as f:
        rows = list(csv.DictReader(f))
    missing = [r["Image ID"] for r in rows if r["Image ID"] not in paths or r["Image ID"] not in entries]
    if missing:
        raise SystemExit(f"{len(missing)} expert-labelled images have no file or Data_Entry_2017.csv row, "
                         f"e.g. {missing[:3]}")
    items = []
    for row in rows:
        labels = {label: int(match(row)) for label, match in NIH_LABELS.items()}
        if any(labels.values()):
            items.append((str(paths[row["Image ID"]]), labels, {}, nih_attributes(entries[row["Image ID"]])))
    print(f"ChestX-ray14 ({expert_labels.name}): {len(items)} of {len(rows)} expert-labelled images have a "
          f"positive shared label")
    return items


def padchest_items(root):
    """(path, labels, conversion info, attributes) for physician-labelled PadChest images (any age) of a
    chest view with a positive shared label."""
    csv_path = next(root.glob("*.csv"))
    items, matched, other_views = [], {label: set() for label in PADCHEST_LABELS}, Counter()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["MethodLabel"] != "Physician":
                continue
            if row["Projection"] not in PADCHEST_VIEWS:
                other_views[row["Projection"]] += 1
                continue
            try:
                terms = [t.strip().lower() for t in ast.literal_eval(row["Labels"]) if isinstance(t, str)]
            except (ValueError, SyntaxError):
                continue
            labels = {}
            for label, match in PADCHEST_LABELS.items():
                hits = {t for t in terms if match(t)}
                matched[label] |= hits
                labels[label] = int(bool(hits))
            if any(labels.values()):
                info = {"invert": row["PhotometricInterpretation_DICOM"] == "MONOCHROME1"}
                items.append((str(root / row["ImageDir"] / row["ImageID"]), labels, info, padchest_attributes(row)))
    print(f"PadChest ({csv_path.name}): {len(items)} images; physician-labelled images of other projections "
          f"left out: {dict(other_views.most_common())}")
    for label, terms in matched.items():
        print(f"  {label}: {sorted(terms)}")
    return items


def load_gray(job):
    """Image -> uint8 grayscale at IMAGE_SIZE, prepared like the CheXpert training images; None if unreadable."""
    path, info = job
    try:
        img = Image.open(path)
        a = np.asarray(img, dtype=np.float64)
    except (OSError, ValueError):
        return None
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F") or a.max() > 255:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    if info.get("invert"):
        a = 255 - a
    gray = Image.fromarray(np.round(a).astype(np.uint8), mode="L")
    return np.asarray(gray.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))[..., 0]


def load_set(items, workers):
    """(images, labels, meta): meta holds each image's file name, sex, age, manufacturer and modality."""
    with Pool(workers) as pool:
        images = pool.map(load_gray, [(p, info) for p, _, info, _ in items], chunksize=16)
    ok = [i for i, im in enumerate(images) if im is not None]
    if len(ok) < len(items):
        print(f"  {len(items) - len(ok)} unreadable images skipped")
    labels = np.array([[items[i][1].get(p, -1) for p in PATHOLOGIES] for i in ok])  # -1: not a shared label
    meta = {"ids": np.array([Path(items[i][0]).name for i in ok]),
            "sex": np.array([items[i][3]["sex"] for i in ok]),
            "age": np.array([items[i][3]["age"] for i in ok], dtype=np.float64),
            "manufacturer": np.array([items[i][3]["manufacturer"] for i in ok]),
            "modality": np.array([items[i][3]["modality"] for i in ok])}
    return np.stack([images[i] for i in ok]), labels, meta


@torch.no_grad()
def predict(model, gray, device, batch_size):
    probs = []
    for i in range(0, len(gray), batch_size):
        x = torch.from_numpy(gray[i:i + batch_size]).to(device).float().div_(255)[:, None].expand(-1, 3, -1, -1)
        with torch.amp.autocast(device_type=device.type):
            probs.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(probs).numpy()


def label_aucs(y, prob, labels):
    """AUC per shared label, among images with at least one positive shared label."""
    cols = [PATHOLOGIES.index(label) for label in labels]
    keep = (y[:, cols] == 1).any(1)
    return {label: float(roc_auc_score(y[keep, c], prob[keep, c])) for label, c in zip(labels, cols)}


def write_examples(sets, path):
    rows = [(name, gray[np.linspace(0, len(gray) - 1, 8).astype(int)]) for name, (gray, *_) in sets.items()]
    sheet = np.concatenate([np.concatenate(list(imgs), axis=1) for _, imgs in rows], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(path)
    print(f"example sheet ({', '.join(name for name, _ in rows)}, one row each): {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nih-dir", type=Path, default=Path("/home/data_shares/purrlab/ChestX-ray14"))
    parser.add_argument("--nih-labels", type=Path,
                        default=DATA_DIR / "ChestX-ray14" / "all_findings_expert_labels_test_labels.csv",
                        help="Google's expert labels for the ChestX-ray14 test images")
    parser.add_argument("--padchest-dir", type=Path, default=Path("/home/data_shares/purrlab/padchest"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_NAMES))
    parser.add_argument("--limit", type=int, default=None, help="use a random N images per dataset (quick check)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS_DIR / "shift" / "chexpert"

    external = {"nih": (nih_items(args.nih_dir, args.nih_labels), [lbl for lbl in NIH_LABELS]),
                "padchest": (padchest_items(args.padchest_dir), [lbl for lbl in PADCHEST_LABELS])}
    sets = {}
    for name, (items, _) in external.items():
        if args.limit:
            items = [items[i] for i in np.random.default_rng(0).permutation(len(items))[:args.limit]]
        sets[name] = load_set(items, args.workers)
        sex, age = sets[name][2]["sex"], sets[name][2]["age"]
        print(f"{name}: {len(sets[name][0])} images loaded; female {(sex == 'female').sum()}, male "
              f"{(sex == 'male').sum()}, sex unknown {(sex == '').sum()}; age < 60 {(age < 60).sum()}, "
              f"60+ {(age >= 60).sum()}, unknown {np.isnan(age).sum()} (median {np.nanmedian(age):.0f}, "
              f"under 18 {(age < 18).sum()})")
        y = sets[name][1]
        print("  positives per label: " + ", ".join(f"{label} {(y[:, PATHOLOGIES.index(label)] == 1).sum()}"
                                                    for label in external[name][1]))
        scanners =Counter(zip(sets[name][2]["manufacturer"], sets[name][2]["modality"]))
        if any(manufacturer for manufacturer, _ in scanners):
            print("  images per scanner (manufacturer, modality): " +
                  ", ".join(f"{m or 'unknown'} {mod or 'unknown'} {n}" for (m, mod), n in scanners.most_common()))
    write_examples(sets, out / "examples.png")
    predictions = out / "predictions"
    if args.limit is None:  # labels and attributes once per dataset; probabilities per model below
        predictions.mkdir(parents=True, exist_ok=True)
        for name, (_, y, meta) in sets.items():
            np.savez(predictions / f"{name}_labels.npz", y_true=y, scored=np.array(external[name][1]),
                     label_names=np.array(PATHOLOGIES), **meta)

    results = {}
    for source in args.sources:
        run = RUNS_DIR / "benchmark" / "chexpert" / source
        for record in sorted((run / "seeds").glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            rec = json.loads(record.read_text())
            if "model" not in rec:
                continue
            model = load_source_model(source, len(PATHOLOGIES), dropout=rec["hparams"]["dropout"])
            model.load_state_dict(torch.load(run / rec["model"], map_location="cpu", weights_only=True))
            model.to(device).eval()
            entry = {"per_class": {}}
            saved = np.load(run / rec["predictions"])  # in-distribution reference: the CheXpert test set
            for name, (_, labels) in external.items():
                ref = label_aucs(saved["y_true"].astype(int), saved["y_prob"], labels)
                entry[f"chexpert_{name}_labels"] = float(np.mean(list(ref.values())))
                entry["per_class"][f"chexpert_{name}_labels"] = ref
                prob = predict(model, sets[name][0], device, args.batch_size)
                if args.limit is None:
                    (predictions / source).mkdir(exist_ok=True)
                    np.save(predictions / source / f"seed_{rec['seed']}_{name}.npy", prob.astype(np.float32))
                aucs = label_aucs(sets[name][1], prob, labels)
                entry[name] = float(np.mean(list(aucs.values())))
                entry["per_class"][name] = aucs
            results[f"{source}/seed_{rec['seed']}"] = entry
            print(f"  {source} seed {rec['seed']}: " + ", ".join(f"{k} {v:.4f}" for k, v in entry.items() if k != "per_class"),
                  flush=True)
    if not results:
        raise SystemExit("no saved models under runs/benchmark/chexpert/*/models")

    columns = [c for name in external for c in (f"chexpert_{name}_labels", name)]
    print(f"\n{'':14s}" + "".join(f"{c:>26s}" for c in columns))
    for source in args.sources:
        runs = [r for key, r in results.items() if key.startswith(source + "/")]
        if runs:
            cells = [f"{np.mean([r[c] for r in runs]):.4f} ± {np.std([r[c] for r in runs], ddof=1) if len(runs) > 1 else 0:.4f}"
                     for c in columns]
            print(f"{SOURCE_NAMES[source]:14s}" + "".join(f"{c:>26s}" for c in cells))
    if args.limit is None:
        (out / "per_model.json").write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out / 'per_model.json'}")


if __name__ == "__main__":
    main()
