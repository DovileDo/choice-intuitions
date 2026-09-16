# Choice Intuitions

Comparing pretrained source datasets (ImageNet, RadImageNet, ecoset) for CRC pathology classification with ResNet50.

## Data

### Ecoset

1.5M images, 565 categories. Password-protected zip from [Kietzmann Lab](https://codeocean.com/capsule/9570390) (agree to license terms to get the password).

```bash
wget https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip -P /path/to/data/
unzip -P YOUR_PASSWORD ecoset.zip -d /path/to/data/ecoset/
```

Expected layout: `ecoset/{train,val,test}/<class>/*.jpg` (565 class folders each).

### RadImageNet

1.35M medical images (CT/MR/US), 165 classes. Download from [RadImageNet](https://www.radimagenet.com/) — requires registration and access approval.

Pretrained ResNet50 checkpoint: [RadImageNet_ResNet50_pytorch.pt](https://zenodo.org/records/10528450)

Expected layout: CSV splits (`RadiologyAI_{train,val,test}.csv`) pointing to `CT/`, `MR/`, `US/` image directories.

### CRC Pathology (benchmark target)

100K + 7K colorectal cancer histology patches from [Kather et al.](https://zenodo.org/records/1214456) (DOI: 10.5281/zenodo.1214456).

```bash
# Train set (NCT-CRC-HE-100K)
wget https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content -O NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip -d /path/to/data/pathology/train/

# Test set (CRC-VAL-HE-7K)
wget https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content -O CRC-VAL-HE-7K.zip
unzip CRC-VAL-HE-7K.zip -d /path/to/data/pathology/test/
```

Expected layout: `pathology/{train,test}/<class>/*.tif` (9 tissue classes).

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

## Training ecoset ResNet50

Trains from scratch using the [torchvision v2 recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) (80.9% ResNet50 on ImageNet). Checkpoints every epoch for resumable HPC jobs.

```bash
python3 source/ecoset/train_resnet50_ecoset.py \
    --train-dir /path/to/ecoset/train \
    --val-dir /path/to/ecoset/val \
    --test-dir /path/to/ecoset/test \
    --output-dir ./output \
    --batch-size 256 \
    --resume

# SLURM:
sbatch source/ecoset/run_train.sh
```

## Evaluation

```bash
# Ecoset checkpoint (tests class ordering hypotheses)
python3 source/ecoset/eval_ecoset_resnet50.py

# RadImageNet linear probe (freeze backbone, train FC)
python3 source/radimagenet/eval_radimagenet_resnet50.py

# CRC benchmark (Optuna HP search + held-out eval)
python3 source/pathology/benchmark_resnet50_crc.py
```
