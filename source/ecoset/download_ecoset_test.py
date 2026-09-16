"""
Download only the `test` split of the ecoset dataset from the official archive,
using HTTP range requests so we don't have to pull the full ~165GB zip.

The remote zip is a plain (non-solid) archive with classic PKWARE/ZipCrypto
per-entry encryption (confirmed via its central directory: flag bit 0 set,
compress_type 0/8, no AES extra field) - so Python's stdlib `zipfile` can
decrypt entries directly given the password, no extra crypto library needed.

Usage:
    ECOSET_PASSWORD=... python3 download_ecoset_test.py
    (or leave the env var unset and it will prompt interactively)

The password is obtained by reading the license/README at
https://codeocean.com/capsule/9570390 and agreeing to the terms - get it
yourself, this script never stores or guesses it.
"""

import io
import os
import zipfile
from getpass import getpass

import remotezip
from tqdm import tqdm

ZIP_URL = "https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip"
DATA_DIR = os.environ.get("ECOSET_DATA_DIR", "/home/doju/data/ecoset")
OUT_DIR = os.path.join(DATA_DIR, "test_data")


def main():
    password = os.environ.get("ECOSET_PASSWORD") or getpass(
        "Enter ecoset password (from https://codeocean.com/capsule/9570390): "
    )
    pwd_bytes = password.encode("ascii")

    print("Listing remote archive contents (central directory only, no data download yet)...")
    rz = remotezip.RemoteZip(ZIP_URL)
    test_infos = [i for i in rz.infolist() if i.filename.startswith("test/") and not i.is_dir()]
    print(f"Found {len(test_infos)} test images "
          f"({sum(i.compress_size for i in test_infos) / 1e9:.2f} GB compressed)")

    os.makedirs(OUT_DIR, exist_ok=True)

    n_ok, n_fail = 0, 0
    for info in tqdm(test_infos, desc="Downloading+decrypting test split"):
        # filename looks like "test/0070_bag/n03958227_682.JPEG"
        rel_path = info.filename[len("test/"):]
        dest_path = os.path.join(OUT_DIR, rel_path)
        if os.path.exists(dest_path):
            n_ok += 1
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # remotezip fetches only the byte range for this entry, then
        # zipfile's normal decryption path (pwd=...) decrypts+inflates it
        try:
            data = rz.read(info.filename, pwd=pwd_bytes)
        except RuntimeError as e:
            n_fail += 1
            tqdm.write(f"FAILED {info.filename}: {e}")
            continue

        with open(dest_path, "wb") as f:
            f.write(data)
        n_ok += 1

    print(f"Done. {n_ok} images written to {OUT_DIR}, {n_fail} failed.")


if __name__ == "__main__":
    main()
