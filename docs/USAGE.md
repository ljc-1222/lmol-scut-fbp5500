# Running the public LMOL pipeline

This document covers the **balanced three-class pipeline** in this repository. Start with the [README](../README.md) for the project and results, [Experiments](EXPERIMENTS.md) for historical settings and evaluation scope, and [implementation notes](TRAINING_ALIGNMENT.md) for the objective and score estimator.

## Environment

Use **Python 3.11**, preferably on Linux or WSL2. Training and pairwise inference use an NVIDIA CUDA GPU with enough VRAM for LLaVA-1.5-7B. Install the pinned dependencies in [`requirements.txt`](../requirements.txt) in a separate environment. Choose a PyTorch wheel compatible with your CUDA device; the environment check below verifies imports and device visibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

`check_environment.py` checks installed modules, runtime imports, and whether a CUDA device is visible; it is not a full training test. For platform-specific PyTorch wheels, consult the [PyTorch 2.4.1 installation instructions](https://docs.pytorch.org/get-started/previous-versions/).

## Prepare SCUT-FBP5500

Review the [dataset's non-commercial research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), then install and validate the dataset locally:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
```

Expected layout:

```text
data/raw/SCUT-FBP5500/
├── Images/
└── labels.txt
```

The annotation file supports image filename and mean score per row, separated by whitespace or commas. For manual setup instructions:

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
```

## Generate training and reference pairs

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

This creates the following files beneath `artifacts/pairs/full/fold_0/`: `train_images.csv`, `test_images.csv`, `train_pairs.csv`, `references.csv`, and `test_reference_pairs.csv`.

The public script **generates its own deterministic seeded folds**, rather than reading an original paper split. The default balanced training set uses up to 30,000 pairs per class. Test images are never used as known-score references. The default reference set contains up to 10 training images in each non-empty score bin of width 0.1; set `evaluation.reference_max_per_bin: null` to remove the cap.

## Fine-tune a comparator

Use a dry run first to check the data paths and pair files without loading the 7B model:

```bash
python scripts/train.py --fold-index 0 --dry-run
```

Train using the YAML defaults:

```bash
python scripts/train.py --fold-index 0 --no-wandb
```

The *example* YAML configures one epoch, per-device batch size 1, gradient accumulation 16, separate learning rates for the projector and LoRA parameters, and periodic checkpoint saving. The historical report uses five epochs and effective batch size 64; it also uses a different score-search grid. See [the experiment/code comparison](EXPERIMENTS.md#relation-to-the-public-implementation).

W&B logging is enabled by default in the YAML; `--no-wandb` opts out, and `--wandb-mode offline` avoids online logging. `--stop-after-steps N` exits a diagnostic run without shortening the configured learning-rate schedule.

Training-time pairwise monitoring is **opt-in** with a separate `--eval-pairs` file, and must use pairs held out **from the training set**. The public `test_reference_pairs.csv` is for final evaluation only. Do not use it for model selection.

Checkpoints are written under `outputs/lmol_scut_fbp5500/fold_0/`, with a final `last/` directory containing the adapter, projector, and processor.

## Evaluate

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

The full command evaluates **all target/reference pairs** and can be expensive. `--limit 256` is only a pipeline smoke test: it evaluates the first 256 pairs, **not** a representative full-test score.

You can instead produce pairwise predictions and estimate scores in two stages:

```bash
python scripts/pairwise_infer.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last

python scripts/estimate_scores.py \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

The predictions, metrics and progress files are written under `outputs/evaluation/fold_0/` and are excluded from Git, as are input data, generated pair CSVs and checkpoints.

An [earlier fold-0 checkpoint package](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500) is hosted separately from this repo. It comes from an earlier run: the current split, configuration, checkpoint layout and evaluation behavior should not be assumed to reproduce that historical run.

## Tests

```bash
python -m unittest discover -s tests -v
```

The repository's unit tests use synthetic data and do not load a 7B checkpoint or run full GPU training. They cover data preparation, label mappings, first-token classification bookkeeping, reference selection and score estimation.
