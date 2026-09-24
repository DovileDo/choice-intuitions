"""
Estimate how far a running benchmark is and how long it still needs.

    python -m intuitions.progress runs/benchmark/chexpert            # every source
    python -m intuitions.progress runs/benchmark/chexpert/imagenet   # one source

Reads the timestamps in benchmark.log, so it works while the run is in progress.
Estimates assume the remaining trials cost what the finished ones did; they are
rough while few trials have run and get sharper as the search proceeds.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ")
TRIAL_START = re.compile(r"trial (\d+): \{")
TRIAL_SEED = re.compile(r"trial (\d+) seed (\d+):")
EVAL_SEED = re.compile(r"Eval seed (\d+): val macro AUC")
N_TRIALS = re.compile(r"'n_trials': (\d+)")
N_EVAL_SEEDS = re.compile(r"'eval_seeds': \[([^\]]*)\]")


def parse(log_path):
    """Counts and timestamps extracted from one benchmark.log."""
    state = {"n_trials": 60, "n_eval_seeds": 5, "trials": 0, "search_seeds": 0, "eval_seeds": 0,
             "start": None, "last": None, "search_done": False, "finished": False}
    for line in log_path.read_text(errors="replace").splitlines():
        stamp = TIMESTAMP.match(line)
        if not stamp:
            continue
        when = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S")
        state["start"] = state["start"] or when
        state["last"] = when
        if match := N_TRIALS.search(line):
            state["n_trials"] = int(match.group(1))
        if match := N_EVAL_SEEDS.search(line):
            state["n_eval_seeds"] = len(match.group(1).split(","))
        state["trials"] += bool(TRIAL_START.search(line))
        state["search_seeds"] += bool(TRIAL_SEED.search(line))
        state["eval_seeds"] += bool(EVAL_SEED.search(line))
        state["search_done"] |= "Best trial" in line
        state["finished"] |= "FINAL macro_auc" in line
    return state


def estimate(state):
    """(elapsed seconds, remaining seconds or None, one-line status)."""
    if state["start"] is None:
        return 0.0, None, "no timestamps yet"
    elapsed = (state["last"] - state["start"]).total_seconds()
    if state["finished"]:
        return elapsed, 0.0, "finished"
    done_seeds = state["search_seeds"] + state["eval_seeds"]
    if done_seeds == 0:
        return elapsed, None, "started, no seed finished yet"
    per_seed = elapsed / done_seeds
    # Pruning stops trials early, so use the observed seeds-per-trial rather than all five.
    seeds_per_trial = state["search_seeds"] / max(state["trials"], 1)
    remaining_search = 0.0 if state["search_done"] else \
        max(state["n_trials"] - state["trials"], 0) * seeds_per_trial * per_seed
    # Pruning drops whole seeds, not epochs, so a finished seed costs the same in either phase.
    remaining_eval = max(state["n_eval_seeds"] - state["eval_seeds"], 0) * per_seed
    phase = "eval" if state["search_done"] else "search"
    status = (f"{phase}: trial {state['trials']}/{state['n_trials']}, "
              f"{state['eval_seeds']}/{state['n_eval_seeds']} eval seeds")
    return elapsed, remaining_search + remaining_eval, status


def hm(seconds):
    return "?" if seconds is None else f"{int(seconds // 3600)}h{int(seconds % 3600) // 60:02d}m"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a run directory, or a directory of run directories")
    args = parser.parse_args()

    logs = sorted(args.path.glob("benchmark.log")) or sorted(args.path.glob("*/benchmark.log"))
    if not logs:
        raise SystemExit(f"no benchmark.log under {args.path}")

    total_remaining = 0.0
    for log_path in logs:
        elapsed, remaining, status = estimate(parse(log_path))
        total_remaining += remaining or 0.0
        print(f"{log_path.parent.name:<18} elapsed {hm(elapsed):>7}  left {hm(remaining):>7}  {status}")
    if len(logs) > 1:
        print(f"{'TOTAL (started)':<18} {'':>15} left {hm(total_remaining):>7}")


if __name__ == "__main__":
    main()
