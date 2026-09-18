import logging
import random
import subprocess
import sys

import numpy as np
import torch

from intuitions.paths import PROJECT_ROOT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path):
    """INFO to the console, DEBUG (incl. per-epoch metrics) appended to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(logfile)
    for noisy in ("PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def git_revision():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}{'-dirty' if dirty else ''}"


def pick_device(name=None):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
