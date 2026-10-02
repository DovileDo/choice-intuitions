#!/bin/bash
#SBATCH --job-name=shift-xray
#SBATCH --partition=acltr
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=job.%j.out

#   sbatch scripts/run_shift_xray.sh --limit 200 --sources imagenet     # quick check, saves only examples.png
#   sbatch scripts/run_shift_xray.sh                                    # every saved CS-xray model
#
# Arguments are passed to intuitions.shift_xray. Every saved CS-xray evaluation model
# (runs/benchmark/chexpert/<source>/models/seed_<N>.pt) predicts ChestX-ray14 and PadChest, read in
# place from the shared copies (--nih-dir, --padchest-dir). ChestX-ray14 uses Google's expert labels,
# expected at $INTUITIONS_DATA_DIR/ChestX-ray14/all_findings_expert_labels_test_labels.csv (or pass
# --nih-labels). Writes runs/shift/chexpert/:
# per_model.json, examples.png, and predictions/ with each image's labels, sex and age and every
# model's probabilities, which intuitions.fairness reads.
#
# Look at runs/shift/chexpert/examples.png after the quick check: the X-rays should look like
# CheXpert's (white bones, black air), not inverted or washed out. The full run is a few hours,
# mostly reading PadChest's 16-bit PNGs.

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0
# the conda shell function is not defined in a batch job; conda activate needs it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/intuitions" || exit 1
export PYTHONNOUSERSITE=1

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1

# The source checkpoints rebuild each model before its fine-tuned weights are loaded; the defaults
# are what intuitions/paths.py assumes.
export INTUITIONS_DATA_DIR="${INTUITIONS_DATA_DIR:-$HOME/data}"
export INTUITIONS_MODELS_DIR="${INTUITIONS_MODELS_DIR:-$HOME/pretrained_models}"

echo "Running on $(hostname) at $(date '+%F %T'): intuitions.shift_xray $*"
python -m intuitions.shift_xray "$@" || exit 1
echo "done at $(date '+%F %T')"
