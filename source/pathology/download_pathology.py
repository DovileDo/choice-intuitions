"""
Download the "pathology" dataset: colorectal cancer histology image patches
from Kather et al., hosted on Zenodo (DOI 10.5281/zenodo.1214456).

  train -> NCT-CRC-HE-100K (100,000 H&E patches, color-normalized, 9 tissue classes)
  test  -> CRC-VAL-HE-7K   (7,180 H&E patches from 50 held-out patients)

Each zip is downloaded with a streaming GET, verified against its published
MD5 (from the Zenodo API), then extracted so DATA_DIR ends up as:
    DATA_DIR/train/<class>/*.tif
    DATA_DIR/test/<class>/*.tif
The original zips are kept under DATA_DIR/raw/ for provenance.

Usage:
    python3 download_pathology.py
"""

import hashlib
import os
import shutil
import zipfile

import requests
from tqdm import tqdm

DATA_DIR = os.environ.get("PATHOLOGY_DATA_DIR", "/home/doju/data/pathology")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# key -> (zenodo download url, expected md5, split subdir name)
FILES = {
    "NCT-CRC-HE-100K.zip": (
        "https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content",
        "6fd702d11df6292bc054397ae038a464",
        "train",
    ),
    "CRC-VAL-HE-7K.zip": (
        "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content",
        "2fd1651b4f94ebd818ebf90ad2b6ce06",
        "test",
    ),
}


def download(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download")
        return
    tmp = dest + ".part"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))
    os.rename(tmp, dest)


def verify_md5(path, expected):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise SystemExit(f"MD5 mismatch for {path}: expected {expected}, got {actual}")
    print(f"{os.path.basename(path)}: MD5 OK ({actual})")


def extract(zip_path, split_dir):
    if os.path.isdir(split_dir) and os.listdir(split_dir):
        print(f"{split_dir} already populated, skipping extraction")
        return
    os.makedirs(split_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # zip contains a single top-level folder (e.g. "NCT-CRC-HE-100K/<class>/*.tif");
        # strip it so DATA_DIR/<split>/<class>/*.tif is the final layout.
        top_level = names[0].split("/")[0]
        for name in tqdm(names, desc=f"extracting {os.path.basename(zip_path)}"):
            if name == top_level + "/" or not name.startswith(top_level + "/"):
                continue
            rel = name[len(top_level) + 1:]
            if not rel:
                continue
            target = os.path.join(split_dir, rel)
            if name.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename, (url, md5, split) in FILES.items():
        zip_path = os.path.join(RAW_DIR, filename)
        print(f"\n=== {filename} -> {split}/ ===")
        download(url, zip_path)
        verify_md5(zip_path, md5)
        extract(zip_path, os.path.join(DATA_DIR, split))
    print("\nDone.")
    for split in ("train", "test"):
        split_dir = os.path.join(DATA_DIR, split)
        classes = sorted(os.listdir(split_dir))
        n_images = sum(len(os.listdir(os.path.join(split_dir, c))) for c in classes)
        print(f"{split}: {len(classes)} classes, {n_images} images -> {split_dir}")


if __name__ == "__main__":
    main()
