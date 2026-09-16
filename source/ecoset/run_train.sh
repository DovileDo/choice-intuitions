#!/bin/bash
#SBATCH --job-name=ecoset_r50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=ecoset_train_%j.log

# === Edit these paths ===
DATA_DIR="${ECOSET_DATA_DIR:-/path/to/ecoset}"
OUTPUT_DIR="${ECOSET_OUTPUT_DIR:-./ecoset_resnet50_v2}"

# === Activate env ===
source activate ecoset 2>/dev/null || conda activate ecoset

# === Extract data if needed ===
if [ ! -d "$DATA_DIR/train" ] && [ -f "$DATA_DIR/ecoset.zip" ]; then
    echo "Extracting ecoset.zip..."
    read -sp "Ecoset password: " ECOSET_PW
    echo
    unzip -n -P "$ECOSET_PW" "$DATA_DIR/ecoset.zip" -d "$DATA_DIR"
fi

# === Train (resumes automatically if checkpoint exists) ===
python3 "$(dirname "$0")/train_resnet50_ecoset.py" \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/val" \
    --test-dir "$DATA_DIR/test" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 256 \
    --workers 8 \
    --resume
