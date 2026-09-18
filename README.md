# Choice Intuitions

Benchmark of how ResNet-50 source models pretrained on **ImageNet**, **RadImageNet** and
**Ecoset** (DVD baseline and DVD-S, Lu et al. 2026) transfer to two small medical imaging
targets, against training from scratch, used to compare practitioners' intuitions about
source-dataset choice with empirical outcomes:

- **CS-tissue (`crc`)**: 9-class colorectal tissue classification. 250 patches/class from
  NCT-CRC-HE-100K for train/val (80/20); test is CRC-VAL-HE-7K minus 250 random patches
  per class (4,930 patches), fixed for every seed.
- **CS-xray (`chexpert`)**: 8-pathology multi-label chest X-ray classification on 834 CheXpert
  val+test images (662 patients), patient-level splits.

Protocol (`intuitions/finetune.py`, written up in `paper/`): Optuna TPE search (60 trials),
each trial scored by mean validation macro AUC over data seeds 0-4; the best configuration is
retrained on held-out seeds 5-9 and test metrics are reported as mean ± std.

## Layout

```
intuitions/                 benchmark package; run modules with `python -m` from the repo root
  paths.py                  data, model and output locations
  source_models.py          the four source models, each with its own input normalization
  targets/                  target tasks: crc.py, chexpert.py
  finetune.py               shared protocol: Optuna search + final evaluation
  benchmark.py              CLI: fine-tune one source on one target
  prepare_chexpert.py       builds the CheXpert benchmark pool
  verify_sources.py         checks loading, input normalization and source-task accuracy
  radimagenet_probe.py      RadImageNet linear probe (head used by verify_sources)
ecoset_pretraining/         standalone from-scratch Ecoset ResNet-50 training (HPC)
paper/                      LaTeX: benchmark protocol and hyperparameter appendix
tests/                      unit tests
runs/                       all outputs and logs (git-ignored)
```

## Data, weights and outputs

Locations are set in `intuitions/paths.py` and can be overridden with environment variables:

| Variable | Default | Contents |
|---|---|---|
| `INTUITIONS_DATA_DIR` | `~/data` | datasets (read-only) |
| `INTUITIONS_MODELS_DIR` | `~/pretrained_models` | pretrained weights (read-only) |
| `INTUITIONS_RUNS_DIR` | `<repo>/runs` | logs, Optuna studies, results |

Expected data layout:

```
~/data/pathology/train/<CLASS>/*.tif     NCT-CRC-HE-100K (ADI BACK DEB LYM MUC MUS NORM STR TUM)
~/data/pathology/test/<CLASS>/*.tif      CRC-VAL-HE-7K
~/data/CheXpert/{val,test}/, {val,test}_labels.csv
~/data/ecoset/test_data/<id>_<class>/    Ecoset test split (verify_sources only)
~/data/radiology_ai/                     RadImageNet + RadiologyAI_{train,val,test}.csv (probe only)
```

Source models (`--source`); every model takes RGB tensors in [0, 1] and applies its own
pretraining normalization internally, so all targets use identical transforms:

| `--source` | Weights | Input convention | Pretrained at |
|---|---|---|---|
| `imagenet` | torchvision `IMAGENET1K_V2` (downloaded automatically) | RGB, ImageNet mean/std | 224 px |
| `radimagenet` | `radimagenet/RadImageNet_ResNet50_pytorch.pt` ([BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)) | BGR, scaled to [-1, 1] | 224 px |
| `ecoset_baseline` | `ecoset/dvd_baseline_checkpoint_best.pth` ([OSF 7mkuq](https://osf.io/7mkuq/)) | RGB, [0, 1] | 256 px |
| `ecoset_dvd_s` | `ecoset/dvd_s_checkpoint_best.pth` ([OSF 7mkuq](https://osf.io/7mkuq/)) | RGB, [0, 1] | 256 px |
| `scratch` | none: random initialisation (baseline) | RGB, ImageNet mean/std | – |

```bash
curl -L -o ~/pretrained_models/ecoset/dvd_baseline_checkpoint_best.pth https://osf.io/download/45zhv/
curl -L -o ~/pretrained_models/ecoset/dvd_s_checkpoint_best.pth https://osf.io/download/z7msp/
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m intuitions.prepare_chexpert        # once: writes ~/data/CheXpert/chexpert_benchmark.csv
python -m intuitions.verify_sources          # optional: confirm all source models load and preprocess correctly

python -m intuitions.benchmark --target crc --source imagenet                 # search + final eval
python -m intuitions.benchmark --target crc --source imagenet --phase search   # search only
python -m intuitions.benchmark --target crc --source imagenet --phase eval     # final eval from best_hparams.json

# all sources x targets (~4 days on one GB10 GPU before pruning)
for target in crc chexpert; do
  for source in imagenet radimagenet ecoset_baseline ecoset_dvd_s scratch; do
    python -m intuitions.benchmark --target "$target" --source "$source"
  done
done
```

Re-running an interrupted search resumes its Optuna study and tops it up to `--n-trials`.
The final evaluation skips eval seeds that are already finished, so an interrupted eval resumes
and more seeds can be added later; `results.json` always summarises every finished seed:

```bash
python -m intuitions.benchmark --target crc --source imagenet --phase eval --eval-seeds 10 11 12 13 14
```

Finished seeds must have been trained with the current `best_hparams.json`; if the search is
rerun and picks different hyperparameters, move the old `seeds/` and `predictions/` aside first.
For a quick end-to-end check use `--n-trials 1 --search-seeds 0 --eval-seeds 5` with
`INTUITIONS_RUNS_DIR` pointing somewhere temporary.

## Outputs

```
runs/benchmark/<target>/<source>/
  benchmark.log          console output plus per-epoch train/val metrics
  optuna_study.db        Optuna study (SQLite)
  best_hparams.json      best search configuration
  seeds/seed_<N>.json    eval seed N: hyperparameters, split sizes, test metrics, learning curve
  results.json           all finished eval seeds and their mean ± std summary
  predictions/seed_<N>.npz  per-image test predictions for eval seed N: y_true, y_prob,
                         ids (image path relative to the dataset dir), groups (patient
                         for CheXpert; the patch itself for CRC), label_names
runs/source_checks/      verify_sources.{log,json}, radimagenet_probe/
runs/ecoset_pretraining/ checkpoints and log.txt from ecoset_pretraining/
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

Tests that need a dataset or checkpoint are skipped when it is missing.

## Ecoset pretraining from scratch

`ecoset_pretraining/train_resnet50_ecoset.py` trains a ResNet-50 on the full Ecoset with the
[torchvision v2 recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/),
checkpointing every epoch for resumable HPC jobs. It is independent of the benchmark package.

Data: 1.5M images, 565 categories. Password-protected zip from
[Kietzmann Lab](https://codeocean.com/capsule/9570390) (agree to the license terms to get the password):

```bash
wget https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip -P ~/data/ecoset/full_dataset/
unzip -P YOUR_PASSWORD ~/data/ecoset/full_dataset/ecoset.zip -d ~/data/ecoset/full_dataset/
```

```bash
bash ecoset_pretraining/setup_env.sh              # conda env "ecoset"
sbatch ecoset_pretraining/run_train.sh            # SLURM, submit from the repo root
python3 ecoset_pretraining/train_resnet50_ecoset.py \
    --train-dir ~/data/ecoset/full_dataset/train --val-dir ~/data/ecoset/full_dataset/val \
    --test-dir ~/data/ecoset/full_dataset/test --resume [--eval-only]
tail -f runs/ecoset_pretraining/log.txt
```

`ecoset_pretraining/pilot_train.py` measures training throughput on the Ecoset test split to
estimate full-run duration.
