#!/bin/bash
#SBATCH --job-name=chexpert
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=job.%j.out

#   sbatch --gres=gpu:v100:1 scripts/run_chexpert.sh
#   sbatch scripts/run_chexpert.sh imagenet radimagenet     # a subset, in that order
#
# Every source on CS-xray, one after another in a single job. Results land in
# runs/benchmark/chexpert/<source>/. Two to three days for all five, extrapolated from the
# CRC runs: 5-10 h per pretrained source and about three times that for the randomly
# initialised one, which searches over 50-150 epochs against 20-50.
#
# The first job also builds the benchmark pool -- the label CSVs and the per-seed sampling
# index -- under $INTUITIONS_DATA_DIR/CheXpert, reading the images in place from the shared
# copy at $CHEXPERT_DIR. Later jobs reuse it.
#
# An interrupted job resumes: submit the same command again. Finished trials are in
# optuna_study.db and finished evaluation seeds in seeds/seed_<N>.json, and both are
# skipped on a second run.
#
# Progress while it runs, from the timestamps in the logs:
#
#   python -m intuitions.progress runs/benchmark/chexpert
#
# The estimate reads about 20% high in the first hours -- pruning only starts after five
# completed trials -- and converges as the search proceeds.

SOURCES=("$@")
if [ ${#SOURCES[@]} -eq 0 ]; then
    SOURCES=(imagenet radimagenet ecoset_baseline ecoset_dvd_s scratch)
fi

echo "Running on $(hostname): ${SOURCES[*]}"
export TMPDIR=$HOME/tmp

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
# the conda shell function is not defined in a batch job; conda activate needs it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/intuitions" || exit 1
export PYTHONNOUSERSITE=1

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

# Where the images and the source checkpoints are; the defaults are what
# intuitions/paths.py assumes. Set them at submission if they live elsewhere.
export INTUITIONS_DATA_DIR="${INTUITIONS_DATA_DIR:-$HOME/data}"
export INTUITIONS_MODELS_DIR="${INTUITIONS_MODELS_DIR:-$HOME/pretrained_models}"

# The benchmark pool: the official label CSVs, and every listed image resolved to the
# batch folder that holds it, so the images are read in place and nothing is copied.
# Built once and reused by every later job; delete the CSV to rebuild it.
CHEXPERT="${CHEXPERT_DIR:-/home/data_shares/purrlab/CheXpert}"
BENCHMARK_CSV="$INTUITIONS_DATA_DIR/CheXpert/chexpert_benchmark.csv"
if [ ! -f "$BENCHMARK_CSV" ]; then
    echo "building $BENCHMARK_CSV"
    mkdir -p "$INTUITIONS_DATA_DIR/CheXpert" || exit 1
    # train.csv ships with the first batch, the expert-labelled val and test sets with CheXlocalize.
    for csv in "$CHEXPERT/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv" \
               "$CHEXPERT/chexlocalize/CheXpert/val_labels.csv" \
               "$CHEXPERT/chexlocalize/CheXpert/test_labels.csv"; do
        [ -f "$INTUITIONS_DATA_DIR/CheXpert/$(basename "$csv")" ] && continue
        cp "$csv" "$INTUITIONS_DATA_DIR/CheXpert/" || { echo "missing $csv" >&2; exit 1; }
    done
    python -m intuitions.prepare_chexpert --image-roots \
        "$CHEXPERT/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)" \
        "$CHEXPERT/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)" \
        "$CHEXPERT/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)" \
        "$CHEXPERT/chexlocalize/CheXpert" || exit 1
fi

# The Optuna study is a SQLite file written after every trial, so it runs on node-local
# disk and is copied back after each source rather than living on shared storage.
SHARED="$REPO/runs/benchmark/chexpert"
LOCAL="/scratch/$USER/chexpert-$SLURM_JOB_ID"
mkdir -p "$LOCAL" 2>/dev/null || LOCAL="/tmp/chexpert-$SLURM_JOB_ID"
mkdir -p "$LOCAL/benchmark/chexpert" "$SHARED" || exit 1
export INTUITIONS_RUNS_DIR="$LOCAL"
echo "working in $LOCAL"

# Resume: bring any previous run over before starting.
rsync -a "$SHARED/" "$LOCAL/benchmark/chexpert/" || exit 1

copy_back() { rsync -a "$LOCAL/benchmark/chexpert/" "$SHARED/"; }
trap copy_back EXIT

# Rough relative cost, used to project the remaining time: training from random
# initialisation searches over three times as many epochs as fine-tuning does.
weight_of() { [ "$1" = scratch ] && echo 3 || echo 1; }
hm() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }

total_weight=0
for source in "${SOURCES[@]}"; do total_weight=$((total_weight + $(weight_of "$source"))); done

job_start=$(date +%s)
done_weight=0
failed=()

for source in "${SOURCES[@]}"; do
    echo "=== $source starting at $(date '+%F %T')"
    start=$(date +%s)
    python -m intuitions.benchmark --target chexpert --source "$source" || failed+=("$source")
    took=$(($(date +%s) - start))
    copy_back

    done_weight=$((done_weight + $(weight_of "$source")))
    elapsed=$(($(date +%s) - job_start))
    left=$((elapsed * (total_weight - done_weight) / done_weight))
    echo "=== $source took $(hm $took) | elapsed $(hm $elapsed) | estimated remaining $(hm $left)"
done

echo "all done in $(hm $(($(date +%s) - job_start)))"

trap - EXIT
if copy_back; then
    rm -rf "$LOCAL"
else
    echo "final copy back failed; leaving $LOCAL on $(hostname)" >&2
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
