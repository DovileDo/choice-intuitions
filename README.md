# Choice Intuitions

Train ResNet50 on ecoset from scratch using the [torchvision v2 recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) (80.9% on ImageNet). Checkpoints every epoch for resumable HPC jobs.

## Data

1.5M images, 565 categories. Password-protected zip from [Kietzmann Lab](https://codeocean.com/capsule/9570390) (agree to license terms to get the password).

```bash
wget https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip -P /path/to/data/
unzip -P YOUR_PASSWORD ecoset.zip -d /path/to/data/ecoset/
```

Expected layout: `ecoset/{train,val,test}/<class>/*.jpg` (565 class folders each).

## Setup

```bash
bash source/ecoset/setup_env.sh
```

Or manually:
```bash
conda create -n ecoset python=3.11 -y
conda activate ecoset
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm
```

## Training

```bash
python3 source/ecoset/train_resnet50_ecoset.py \
    --train-dir /path/to/ecoset/train \
    --val-dir /path/to/ecoset/val \
    --test-dir /path/to/ecoset/test \
    --output-dir ./output \
    --batch-size 256 \
    --resume
```

SLURM (edit paths in the script first):
```bash
sbatch source/ecoset/run_train.sh
```

Monitor progress:
```bash
tail -f output/log.txt
```

## Evaluation

```bash
python3 source/ecoset/train_resnet50_ecoset.py \
    --train-dir /path/to/ecoset/train \
    --val-dir /path/to/ecoset/val \
    --test-dir /path/to/ecoset/test \
    --output-dir ./output \
    --resume --eval-only
```
