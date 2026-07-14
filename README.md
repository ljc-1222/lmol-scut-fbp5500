# LMOL on SCUT-FBP5500

Research implementation of **Large Multimodal Model based Order Learning (LMOL)** for facial beauty prediction on **SCUT-FBP5500**.

The implementation follows Liu, Liang, and Lin, [“Large Multimodal Model is a Better Comparator on Facial Beauty Prediction”](https://doi.org/10.1109/ICASSP49660.2025.10889943), ICASSP 2025. It compares two faces as `First.`, `Second.`, or `Similar.`, then recovers absolute scores from target-reference comparisons with UOL-style Bradley-Terry estimation.

## Implementation

- Base model: `llava-hf/llava-1.5-7b-hf` at 336×336 resolution.
- Input: two separately encoded images with shared vision/projector weights.
- Prompt: `<image1><image2>, which face looks more attractive?`.
- Labels: `First.`, `Second.`, and `Similar.` with score threshold `theta=0.2`.
- Training: frozen vision tower, trainable multimodal projector, and QLoRA on language-model attention projections.
- Pairs: 30,000 examples per label and fold when enough candidates exist.
- References: up to 10 train-split images per 0.1-wide score bin by default; test images are never references.
- Evaluation: UOL Bradley-Terry score recovery with PC, MAE, and RMSE.

Training and inference optimize only the first answer token as a three-way choice. They do not optimize the complete answer string or call `model.generate`. See [the method alignment note](docs/TRAINING_ALIGNMENT.md) for the paper-to-code boundary and reproducibility assumptions.

## Setup

Python 3.11 and a CUDA GPU are recommended for the pinned runtime:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

For a non-strict import and hardware summary:

```bash
python scripts/check_environment.py
```

## Dataset

SCUT-FBP5500 is distributed under the dataset authors’ non-commercial research terms. Review those terms before running:

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

`labels.txt` accepts whitespace-delimited or CSV rows containing an image name and mean score. For manual setup instructions:

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
```

## Build Pairs

Build every configured fold, or only fold 0:

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

Each fold writes:

```text
artifacts/pairs/full/fold_0/
├── train_images.csv
├── test_images.csv
├── references.csv
├── train_pairs.csv
└── test_reference_pairs.csv
```

References are selected only from that fold’s training split. Set `evaluation.reference_max_per_bin: null` to use every train image in each non-empty score bin.

## Train

Validate the fold-0 inputs without loading model weights:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0 --dry-run
```

Start training:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

The default effective batch size is 16 (`per_device_batch_size: 1`, `gradient_accumulation_steps: 16`) with seed 42. Weights & Biases is enabled by default; use `--no-wandb` or `--wandb-mode offline` when needed. `--stop-after-steps N` stops a diagnostic run without changing the configured scheduler horizon.

## Checkpoint and Evaluation

Download the fold-0 adapter and projector checkpoint:

```bash
python -m pip install -U huggingface_hub
hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" \
  --local-dir .
```

Run the complete fold evaluator:

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```

Use `--limit 256` for a checkpoint smoke test. It is not a full evaluation.

Pairwise prediction and score estimation can also run separately:

```bash
python scripts/pairwise_infer.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0

python scripts/estimate_scores.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

Generated datasets, checkpoints, predictions, metrics, and experiment logs are ignored by Git.

## Tests

```bash
python -m unittest discover -s tests
```

## Citation

```bibtex
@inproceedings{liu2025large,
  title = {Large Multimodal Model is a Better Comparator on Facial Beauty Prediction},
  author = {Liu, Zhenyou and Liang, Xuefeng and Lin, Jian},
  booktitle = {ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages = {1--5},
  year = {2025},
  doi = {10.1109/ICASSP49660.2025.10889943}
}
```
