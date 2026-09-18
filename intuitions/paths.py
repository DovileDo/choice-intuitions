"""Filesystem locations, overridable through environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("INTUITIONS_DATA_DIR", Path.home() / "data"))
MODELS_DIR = Path(os.environ.get("INTUITIONS_MODELS_DIR", Path.home() / "pretrained_models"))
RUNS_DIR = Path(os.environ.get("INTUITIONS_RUNS_DIR", PROJECT_ROOT / "runs"))
