"""
Fine-tune one source model on one target task (HP search + final evaluation).

    python -m intuitions.benchmark --target crc --source imagenet
    python -m intuitions.benchmark --target chexpert --source radimagenet --phase search
    python -m intuitions.benchmark --target crc --source ecoset_dvd_s --phase eval
    python -m intuitions.benchmark --target crc --source ecoset_dvd_s --phase eval --eval-seeds 10 11 12 13 14

Outputs go to runs/benchmark/<target>/<source>/: benchmark.log (per-epoch
detail), optuna_study.db, best_hparams.json, seeds/, predictions/ and
results.json. Finished eval seeds are skipped, so evals resume and can be extended.
"""

import argparse
import json
import logging

import optuna
import torch

from intuitions.finetune import EVAL_SEEDS, N_TRIALS, SEARCH_SEEDS, run_final_eval, run_search
from intuitions.paths import RUNS_DIR
from intuitions.source_models import SOURCES
from intuitions.targets import TARGETS
from intuitions.utils import git_revision, pick_device, setup_logging

log = logging.getLogger("intuitions.benchmark")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=list(TARGETS), required=True)
    p.add_argument("--source", choices=list(SOURCES), required=True)
    p.add_argument("--phase", choices=["search", "eval", "both"], default="both")
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--search-seeds", type=int, nargs="+", default=list(SEARCH_SEEDS))
    p.add_argument("--eval-seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    p.add_argument("--device", default=None, help="default: cuda if available")
    args = p.parse_args()
    if set(args.search_seeds) & set(args.eval_seeds):
        p.error("search and eval seeds must be disjoint")
    return args


def main():
    args = parse_args()
    out_dir = RUNS_DIR / "benchmark" / args.target / args.source
    setup_logging(out_dir / "benchmark.log")
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()

    device = pick_device(args.device)
    log.info("git %s | torch %s | device %s | %s", git_revision(), torch.__version__, device, vars(args))
    target = TARGETS[args.target]()

    if args.phase in ("search", "both"):
        best_hparams = run_search(target, args.source, out_dir, device, args.n_trials, args.search_seeds)
    else:
        hp_path = out_dir / "best_hparams.json"
        if not hp_path.exists():
            raise SystemExit(f"{hp_path} not found; run the search phase first")
        best_hparams = json.loads(hp_path.read_text())
        log.info("Loaded best hyperparameters from %s: %s", hp_path, best_hparams)

    if args.phase in ("eval", "both"):
        run_final_eval(target, args.source, out_dir, best_hparams, device, args.eval_seeds)


if __name__ == "__main__":
    main()
