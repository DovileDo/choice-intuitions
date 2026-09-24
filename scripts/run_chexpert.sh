#!/bin/bash
# Fine-tune every source on CS-xray, one after another, in a single job.
#
#     sbatch scripts/run_chexpert.sh            # all five sources
#     bash   scripts/run_chexpert.sh imagenet   # or a subset, in the given order
#
# Each source writes runs/benchmark/chexpert/<source>/, and this script reports the
# time it took plus an estimate for what is left. While a source is running:
#     python -m intuitions.progress runs/benchmark/chexpert
#
#SBATCH --job-name=chexpert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=chexpert_%j.out

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

SOURCES=("$@")
if [ ${#SOURCES[@]} -eq 0 ]; then
    SOURCES=(imagenet radimagenet ecoset_baseline ecoset_dvd_s scratch)
fi

# Rough relative cost, used to project the remaining time: training from random
# initialisation searches over three times as many epochs as fine-tuning does.
weight_of() { [ "$1" = scratch ] && echo 3 || echo 1; }

hm() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }

total_weight=0
for source in "${SOURCES[@]}"; do total_weight=$((total_weight + $(weight_of "$source"))); done

echo "host $(hostname) | $(date '+%F %T') | sources: ${SOURCES[*]}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

job_start=$(date +%s)
done_weight=0
failed=()

for source in "${SOURCES[@]}"; do
    echo "=== $source starting at $(date '+%F %T')"
    start=$(date +%s)
    python -m intuitions.benchmark --target chexpert --source "$source" || failed+=("$source")
    took=$(($(date +%s) - start))

    done_weight=$((done_weight + $(weight_of "$source")))
    elapsed=$(($(date +%s) - job_start))
    left=$((elapsed * (total_weight - done_weight) / done_weight))
    echo "=== $source took $(hm $took) | elapsed $(hm $elapsed) | estimated remaining $(hm $left)"
done

echo "all done in $(hm $(($(date +%s) - job_start)))"
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
