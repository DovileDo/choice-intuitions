#!/bin/bash
# Create conda environment for ecoset ResNet50 training.
# Usage: bash setup_env.sh

set -e

ENV_NAME="ecoset"

conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME"

# PyTorch with CUDA (adjust cu version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install numpy tqdm requests remotezip

echo ""
echo "Environment '$ENV_NAME' ready."
echo "Activate with: conda activate $ENV_NAME"
