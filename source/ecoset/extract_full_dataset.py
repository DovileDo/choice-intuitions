"""
Extract train/ and val/ from the downloaded ecoset.zip ($ECOSET_DATA_DIR/full_dataset/ecoset.zip).
Skips test/ since we already have that extracted separately in $ECOSET_DATA_DIR/test_data.

Usage:
    ECOSET_PASSWORD=... python3 extract_full_dataset.py
"""

import os
import subprocess
from getpass import getpass

DATA_DIR = os.environ.get("ECOSET_DATA_DIR", "/home/doju/data/ecoset")
ZIP_PATH = os.path.join(DATA_DIR, "full_dataset", "ecoset.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "full_dataset", "extracted")


def main():
    if not os.path.exists(ZIP_PATH):
        raise SystemExit(f"{ZIP_PATH} not found - download not finished yet?")

    password = os.environ.get("ECOSET_PASSWORD") or getpass(
        "Enter ecoset password (from https://codeocean.com/capsule/9570390): "
    )
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # -n: never overwrite existing files (safe to re-run/resume)
    # only extract train/* and val/* - test/* we already have in test_data/
    cmd = ["unzip", "-n", "-P", password, ZIP_PATH, "train/*", "val/*", "-d", EXTRACT_DIR]
    print("Running:", " ".join(c if c != password else "***" for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode not in (0, 1):  # 1 = some warnings (e.g. skipped existing), still ok
        raise SystemExit(f"unzip failed with exit code {result.returncode}")
    print(f"Done. Extracted to {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
