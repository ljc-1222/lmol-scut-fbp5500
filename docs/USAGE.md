# Usage

Run commands from the repository root. This guide covers data preparation, balanced three-class training, released-checkpoint evaluation, and code entry points.

## Environment

Use **Python 3.11** on Linux or WSL2 with an NVIDIA CUDA GPU. Install the pinned dependencies in a separate environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

The environment check verifies runtime imports and CUDA visibility. For platform-specific PyTorch wheels, use the [PyTorch 2.4.1 installation options](https://docs.pytorch.org/get-started/previous-versions/).

## Prepare the dataset

Review the [SCUT-FBP5500 non-commercial research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), then download and validate the dataset:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
```

Expected files are `data/raw/SCUT-FBP5500/Images/` and `data/raw/SCUT-FBP5500/labels.txt`. The annotation file accepts an image filename and mean rating per row, separated by whitespace or commas.

For manual setup:

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
```

## Build pairs

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

The command writes `train_images.csv`, `test_images.csv`, `train_pairs.csv`, `references.csv`, and `test_reference_pairs.csv` under `artifacts/pairs/full/fold_0/`.

The script generates deterministic seeded folds and requests 30,000 training pairs per class. Omitting `--fold-index` builds all configured folds. References come from training images only, with a default cap of 10 per 0.1-wide rating bin.

## Train

Validate the data paths without loading model weights:

```bash
python scripts/train.py --fold-index 0 --dry-run
```

Start training:

```bash
python scripts/train.py --fold-index 0 --no-wandb
```

The [YAML](../configs/lmol_scut_fbp5500.yaml) controls the training schedule, batch size, learning rates, logging, and checkpoint frequency. The example uses one epoch, per-device batch size 1, and gradient accumulation 16. See [experiment settings and defaults](TRAINING_ALIGNMENT.md#experiment-settings-and-public-defaults) before comparing a run with the ablation tables.

W&B is enabled in the YAML. `--no-wandb` disables it; `--wandb-mode offline` retains local logging. `--stop-after-steps N` stops a diagnostic run while preserving the configured scheduler horizon.

Training-time evaluation is opt-in via `--eval-pairs`. Use a validation pair file held out from the training data; `test_reference_pairs.csv` is for final evaluation.

Checkpoints are saved under `outputs/lmol_scut_fbp5500/fold_0/`. The final `last/` checkpoint contains the adapter, `projector.pt`, and processor files.

## Evaluate a trained model

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

The evaluator processes all target/reference pairs and writes predictions, metrics, confusion counts, and progress under `outputs/evaluation/fold_0/`. `--limit 256` processes only the first 256 pairs for a pipeline check; it does not produce a full-test result.

Pairwise inference and score estimation can also run separately:

```bash
python scripts/pairwise_infer.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last

python scripts/estimate_scores.py \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

## Released checkpoint

The [Hugging Face release](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500) contains the fold-0 LoRA adapter, full projector, processor files, and evaluation metadata. Download the checkpoint with `huggingface_hub`, which is installed with the model dependencies:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ljc-1222/LMOL-fbp5500",
    repo_type="dataset",
    revision="fbbc31831db7b3a5a3866c7a1e616930099350e1",
    allow_patterns=["checkpoints/fold0/*"],
    local_dir=".",
)
PY

python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```

Use the dataset configuration with split seed 42. The published evaluation covers 1,100 test images and 374,000 comparisons. [Released results](EXPERIMENTS.md#released-checkpoint) links the metrics, completed progress record, and confusion matrix. Loading the adapter alone is incomplete: `projector.pt` is also required.

## Tests and code guide

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic data and do not require the dataset or 7B model weights.

| Component | Entry points |
| --- | --- |
| Splits, pair sampling, references | [`data/`](../data/), [`scripts/build_pairs.py`](../scripts/build_pairs.py) |
| Model loading and checkpoint handling | [`model/lmol.py`](../model/lmol.py) |
| Prompt construction and first-token CE | [`data/instruction_dataset.py`](../data/instruction_dataset.py), [`training/train.py`](../training/train.py) |
| Pairwise inference and score recovery | [`evaluation/`](../evaluation/), [`scripts/evaluate_fold.py`](../scripts/evaluate_fold.py) |

Input data, generated pair CSVs, checkpoints, predictions, and logs are excluded from Git.
