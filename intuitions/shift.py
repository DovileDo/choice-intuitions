"""
Robustness of the fine-tuned CS-tissue (CRC) models to distribution shift.

Every saved evaluation model (runs/benchmark/crc/<source>/models/seed_<N>.pt) is tested on
  - synthetic corruptions of the CRC test set (stain shift, blur, JPEG compression, Gaussian noise,
    reduced contrast), each at three severities: macro-AUC over all nine classes;
  - external H&E tiles from other hospitals and scanners, scored by macro-AUC over the classes shared
    with NCT, with the CRC test set restricted to the same classes as the in-distribution reference:
    STARC-9's STANFORD-CRC-HE-VAL-LARGE and CURATED-TCGA-CRC-HE-VAL-20K (colorectal; six shared
    classes; downsampled 2x from 0.25 to NCT's 0.5 um/px), and HMU-GC-HE-30K (gastric; eight shared
    classes, two of them organ-specific: normal mucosa and tumour; native 224 px at 20x).
Corrupted images are generated once with fixed seeds, so every model sees identical inputs.

    python -m intuitions.shift
    python -m intuitions.shift --sources imagenet --limit 300      # quick check
    python -m intuitions.shift --sets clean_8class hmu_8class      # (re)compute only these

Writes runs/shift/crc/per_model.json (with per-class AUCs for the external sets) after every set and
prints the mean +/- std over seeds per source. A rerun skips sets already done for every model, so an
interrupted run resumes; --sets recomputes the named sets. --limit runs are checks and save nothing.
"""

import argparse
import io
import json
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.color import hed2rgb, rgb2hed
from sklearn.metrics import roc_auc_score

from intuitions.names import SOURCE_NAMES
from intuitions.paths import DATA_DIR, RUNS_DIR
from intuitions.source_models import load_source_model
from intuitions.targets.crc import CLASS_TO_IDX, CLASSES, list_test_files

STARC_DIR = DATA_DIR / "STARC-9"
# NCT class -> external class, for the classes the datasets share.
STARC_SHARED = {"ADI": "ADI", "LYM": "LYM", "MUC": "MUC", "MUS": "MUS", "NORM": "NOR", "TUM": "TUM"}
HMU_SHARED = {"ADI": "ADI", "DEB": "DEB", "LYM": "LYM", "MUC": "MUC", "MUS": "MUS", "NORM": "NOR", "STR": "STR",
              "TUM": "TUM"}
# name -> (tile folder with one subfolder per class, shared classes, downsample 2x to 0.5 um/px)
EXTERNAL = {"stanford": (STARC_DIR / "val_large", STARC_SHARED, True),
            "tcga": (STARC_DIR / "tcga", STARC_SHARED, True),
            "hmu": (DATA_DIR / "HMU-GC-HE-30K" / "tiles", HMU_SHARED, False)}
CORRUPTIONS = {
    "stain": (0.05, 0.1, 0.2),     # HED jitter strength (the training augmentation used 0.02)
    "blur": (1, 2, 3),             # Gaussian blur radius in pixels
    "jpeg": (50, 20, 10),          # JPEG quality
    "noise": (0.04, 0.08, 0.16),   # Gaussian noise std on [0, 1] intensities
    "contrast": (0.7, 0.5, 0.3),   # contrast factor
}


def load_rgb(path, halve=False):
    img = Image.open(path).convert("RGB")
    return np.asarray(img.reduce(2) if halve else img)


def corrupt(job):
    """Apply one corruption at one severity to one image, with a seed fixed by (kind, severity, image)."""
    img, kind, level, seed = job
    rng = np.random.default_rng(seed)
    if kind == "stain":
        hed = rgb2hed(img / 255.0)
        rgb = hed2rgb(hed * rng.uniform(1 - level, 1 + level, 3) + rng.uniform(-level, level, 3))
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
    if kind == "blur":
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=level)))
    if kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=level)
        return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "noise":
        return np.clip(img + rng.normal(0, level * 255, img.shape), 0, 255).astype(np.uint8)
    if kind == "contrast":
        return np.asarray(ImageEnhance.Contrast(Image.fromarray(img)).enhance(level))
    raise ValueError(kind)


def class_aucs(y, prob, classes):
    """One-vs-rest AUC per class in `classes`, among the images whose label is one of them."""
    keep = np.isin(y, classes)
    return {CLASSES[k]: float(roc_auc_score(y[keep] == k, prob[keep, k])) for k in classes}


@torch.no_grad()
def predict(model, images, device, batch_size):
    probs = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2).float().div_(255)
        with torch.amp.autocast(device_type=device.type):
            probs.append(torch.softmax(model(x).float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def saved_models(sources):
    """(source, seed, hparams, model path) for every evaluation model on disk."""
    found = []
    for source in sources:
        run = RUNS_DIR / "benchmark" / "crc" / source
        for record in sorted((run / "seeds").glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            rec = json.loads(record.read_text())
            if "model" in rec:
                found.append((source, rec["seed"], rec["hparams"], run / rec["model"], rec["test_metrics"]["macro_auc"]))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_NAMES))
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images per set (quick check)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sets", nargs="+", default=None,
                        help="(re)compute only these sets, keeping the others' results (default: all not yet done)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = saved_models(args.sources)
    if not models:
        raise SystemExit("no saved models under runs/benchmark/crc/*/models")
    print(f"{len(models)} models: " + ", ".join(f"{s} ({sum(m[0] == s for m in models)})" for s in args.sources))

    test = list_test_files()
    if args.limit:  # a fixed random subset; the list itself is ordered by class
        test = [test[i] for i in sorted(np.random.default_rng(0).permutation(len(test))[:args.limit])]
    crc_images = np.stack([load_rgb(path) for path, _ in test])
    crc_y = np.array([label for _, label in test])
    all_classes = list(range(len(CLASSES)))
    # Evaluation sets: (name, images, labels, classes scored). External and corrupted sets load lazily.
    sets = [("clean", crc_images, crc_y, all_classes)]
    for n_shared in sorted({len(shared) for _, shared, _ in EXTERNAL.values()}):
        idx = next([CLASS_TO_IDX[c] for c in shared] for _, shared, _ in EXTERNAL.values() if len(shared) == n_shared)
        sets.append((f"clean_{n_shared}class", crc_images, crc_y, idx))
    sets += [(f"{name}_{len(shared)}class", ("external", name), None, [CLASS_TO_IDX[c] for c in shared])
             for name, (_, shared, _) in EXTERNAL.items()]
    sets += [(f"{kind}_{s}", ("corrupt", kind, s, level), crc_y, all_classes)  # s = severity 1-3
             for kind, levels in CORRUPTIONS.items() for s, level in enumerate(levels, 1)]
    all_names = [name for name, *_ in sets]
    if args.sets:
        unknown = set(args.sets) - set(all_names)
        if unknown:
            raise SystemExit(f"unknown sets {sorted(unknown)}; choose from {all_names}")
        sets = [s for s in sets if s[0] in args.sets]

    out = RUNS_DIR / "shift" / "crc"
    path_json = out / "per_model.json"
    save = args.limit is None
    results = json.loads(path_json.read_text()) if save and path_json.exists() else {}
    for source, seed, *_ in models:
        results.setdefault(f"{source}/seed_{seed}", {}).setdefault("per_class", {})
    if not args.sets:  # resume: skip sets every model already has
        sets = [s for s in sets if not all(s[0] in results[f"{src}/seed_{sd}"] for src, sd, *_ in models)]
    print(f"to evaluate: {[name for name, *_ in sets]}", flush=True)
    for name, images, y, classes in sets:
        if isinstance(images, tuple) and images[0] == "external":  # tiles of the shared classes, all of them
            root, shared, halve = EXTERNAL[images[1]]
            files = [(p, CLASS_TO_IDX[nct]) for nct, ext in shared.items() for p in sorted((root / ext).glob("*.png"))]
            files = [files[i] for i in np.random.default_rng(0).permutation(len(files))][:args.limit]
            with Pool(args.workers) as pool:
                images = np.stack(pool.starmap(load_rgb, [(p, halve) for p, _ in files], chunksize=64))
            y = np.array([label for _, label in files])
            print(f"{name}: {len(files)} tiles, {images.shape[1]} px", flush=True)
        elif isinstance(images, tuple):  # corrupted CRC test set, generated once for all models
            _, kind, s, level = images
            jobs = [(img, kind, level, [list(CORRUPTIONS).index(kind), s, i]) for i, img in enumerate(crc_images)]
            with Pool(args.workers) as pool:
                images = np.stack(pool.map(corrupt, jobs, chunksize=32))
        for source, seed, hparams, path, recorded in models:
            model = load_source_model(source, len(CLASSES), dropout=hparams["dropout"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            aucs = class_aucs(y, predict(model.to(device).eval(), images, device, args.batch_size), classes)
            entry = results[f"{source}/seed_{seed}"]
            entry[name] = float(np.mean(list(aucs.values())))
            if "class" in name:  # restricted and external sets: keep the per-class AUCs too
                entry["per_class"][name] = aucs
            if name == "clean" and args.limit is None and abs(entry[name] - recorded) > 1e-3:
                raise SystemExit(f"{source} seed {seed}: clean macro-AUC {entry[name]:.4f} != recorded {recorded:.4f}")
        if save:  # after every set, so an interrupted run keeps what it finished
            out.mkdir(parents=True, exist_ok=True)
            tmp = path_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(results, indent=2))
            tmp.replace(path_json)
        print(f"  done: {name}", flush=True)

    columns = ["clean"] + list(CORRUPTIONS) + [name for name in all_names if "class" in name]
    print(f"\n{'':14s}" + "".join(f"{c:>17s}" for c in columns))
    for source in args.sources:
        runs = [r for key, r in results.items() if key.startswith(source + "/")]
        cells = []
        for c in columns:
            keys = [f"{c}_{i}" for i in range(1, 4)] if c in CORRUPTIONS else [c]
            if not all(k in r for r in runs for k in keys):
                cells.append("-")
                continue
            values = np.array([np.mean([r[k] for k in keys]) for r in runs])
            cells.append(f"{values.mean():.4f} ± {values.std(ddof=1) if len(values) > 1 else 0:.4f}")
        print(f"{SOURCE_NAMES[source]:14s}" + "".join(f"{c:>17s}" for c in cells))
    if save:
        print(f"\nwrote {path_json}")


if __name__ == "__main__":
    main()
