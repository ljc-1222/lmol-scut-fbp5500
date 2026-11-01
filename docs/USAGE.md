# Usage

This guide describes the **public balanced three-class implementation**, not an exact replay of the original student-project experiments. Run commands from the repository root. [README](../README.md) introduces the method, and [implementation notes](TRAINING_ALIGNMENT.md) describe the differences from the published LMOL objective.

## 1. Environment

Use Python 3.11 and a CUDA-enabled Linux environment for model training/inference. The pinned environment is listed in [`requirements.txt`](../requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

The initial base checkpoint is `llava-hf/llava-1.5-7b-hf`. A 7B multimodal model needs substantial accelerator memory; installation and unit tests do not validate that your GPU can complete a training run.

## 2. Dataset and folds

Review the [SCUT-FBP5500 usage terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) before downloading the images. The repository does not store the dataset.

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
```

Expected local layout:

```text
data/raw/SCUT-FBP5500/
├── Images/
└── labels.txt
```

`labels.txt` contains one image filename and mean score per row (whitespace-delimited or CSV). To prepare the data manually:

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
```

Build just fold 0 (recommended when first inspecting the code):

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

For each fold, `artifacts/pairs/full/fold_0/` contains `train_images.csv`, `test_images.csv`, `train_pairs.csv`, `references.csv`, and `test_reference_pairs.csv`. Folds are deterministically generated from the configured seed; **they are not imported official split files**. References are sampled only from the training split. The default cap is 10 references per 0.1 score bin; set `evaluation.reference_max_per_bin: null` to remove the cap.

## 3. Fine-tuning

First verify paths, split files, and configuration without loading LLaVA weights:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0 --dry-run
```

Then train:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

Example defaults: 1 epoch; batch size 1 with 16 gradient-accumulation steps; 4-bit QLoRA; separate projector and LoRA learning rates; checkpoint saves every 500 optimizer steps and at the end. These values are *public example defaults*, not a reconstruction of the original project's 5-epoch runs.

Training prints JSON progress lines. Weights & Biases is **off by default**. To opt in:

```bash
python scripts/train.py --fold-index 0 --wandb --wandb-mode offline
```

Training-time evaluation is **off by default**. If you create a separate **training-only held-out** pair CSV, you can set `evaluation.eval_steps` to a positive interval in your own YAML and pass `--eval-pairs path/to/validation_pairs.csv`. Do not use the generated `test_reference_pairs.csv` for model selection or early stopping. Use `--stop-after-steps N` for a diagnostic partial run; the scheduler still uses the full configured horizon.

Checkpoints store the PEFT adapter and the full multimodal projector separately, under `outputs/lmol_scut_fbp5500/fold_0/`.

## 4. Evaluation

To run pairwise comparison and score recovery with your own checkpoint:

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

This full evaluation compares every test image against the selected training references and can be expensive. A small smoke test is available by adding `--limit 256`, but it **does not** measure a representative final Pearson correlation.

Intermediate predictions and score estimation can also run separately:

```bash
python scripts/pairwise_infer.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last

python scripts/estimate_scores.py \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

All generated pair CSVs, model checkpoints, prediction CSVs, metrics, and logs are local artifacts ignored by Git. Full evaluation writes them under `outputs/evaluation/fold_0/`.

An older adapter/projector package is available at [Hugging Face](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500). It belongs to an earlier run; compatibility and split alignment with the current defaults have **not** been revalidated as part of this repository cleanup. Prefer the checkpoint produced by your own run when following this guide.

## 5. Tests

```bash
python -m unittest discover -s tests -v
```

These tests use small synthetic examples and do not require SCUT images or a 7B checkpoint. They exercise data preparation, token-choice bookkeeping, reference selection, and score estimation, not end-to-end GPU training.
