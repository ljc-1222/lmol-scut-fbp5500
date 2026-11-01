# LMOL on SCUT-FBP5500

This repository reimplements [Large Multimodal Model based Order Learning (LMOL)](https://doi.org/10.1109/ICASSP49660.2025.10889943) for facial beauty prediction on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). It fine-tunes LLaVA-1.5-7B to compare two faces, then uses comparisons against known-score reference faces to estimate a rating from 1 to 5.

Our work includes the training and score-estimation pipeline, a **first-token classification objective** in place of full-answer supervision, and experiments on class balance, loss functions, and checkpoint ensembling.

## Results

Facial beauty score prediction on SCUT-FBP5500:

| Model | Evaluation | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | --- | ---: | ---: | ---: |
| [Original LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943) | Cross-validation | 95.65 | 0.1552 | 0.2032 |
| **This implementation** | [Fold 0](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500/blob/fbbc31831db7b3a5a3866c7a1e616930099350e1/results/fold0/metrics.json) | 95.68 | 0.1643 | 0.2130 |

PC denotes Pearson correlation. The released result covers 1,100 test images; the paper and project use different evaluation protocols. See [evaluation details and classwise metrics](docs/EXPERIMENTS.md#released-checkpoint).

## Approach

- **Pair construction.** Build a balanced dataset of `First`, `Second`, and `Similar` comparisons, with 30,000 pairs per class and a similarity threshold of θ = 0.2. [Data pipeline](data/).
- **Multimodal fine-tuning.** Use LLaVA-1.5-7B with a frozen CLIP vision encoder, a trainable multimodal projector, and 4-bit QLoRA on the language model. Projector and LoRA parameters use separate learning rates. [Model](model/lmol.py) · [Training](training/train.py).
- **First-token supervision.** The original paper supervises the answer sequence. We compute cross-entropy over only the `First` / `Second` / `Similar` token logits at the first answer position, and use the same three-way decision during inference. [Objective and implementation](docs/TRAINING_ALIGNMENT.md#first-token-objective).
- **Score recovery.** Select reference faces from the training split across the rating range, compare each target against them, and estimate its score with a Bradley–Terry likelihood. [Evaluation](evaluation/).

## Ablation studies

The training study examines how pairwise classification affects the recovered scores. These ablations form a separate experiment set from the released fold-0 evaluation above.

**Label distribution:** all three settings use 90,000 training pairs and standard cross-entropy.

| Training labels | Overall accuracy (%) ↑ | Similar accuracy (%) ↑ | PC (%) ↑ |
| --- | ---: | ---: | ---: |
| Three classes, balanced | 85.84 | 73.36 | 95.26 |
| Three classes, imbalanced (4:4:2) | 86.30 | 50.51 | 94.76 |
| Two classes, no `Similar` | 92.07 | — | 93.35 |

Higher pairwise accuracy did not translate into better scores. `Similar` comparisons constrain the gap between nearby ratings; under-sampling or removing them weakened score estimation.

The study also evaluates **weighted CE**, **forward/reverse KL consistency**, and **five-checkpoint logit ensembling**. KL reached 98.36% order-reversal consistency, but filtering inconsistent comparisons reduced the usable evidence for score recovery. Ensembling improved the ablation baseline's PC from 95.26% to 95.32%. [Full results, settings, and analysis](docs/EXPERIMENTS.md#label-distribution).

The public code provides the balanced three-class training and evaluation pipeline; the other ablation variants are documented experiments.

## Getting started

Use Python 3.11 and a CUDA GPU. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

Review the [SCUT-FBP5500 research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), then prepare the data:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
python scripts/build_pairs.py --fold-index 0
```

Train a comparator and evaluate its saved checkpoint:

```bash
python scripts/train.py --fold-index 0 --no-wandb
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

To use the [released checkpoint](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500), follow the [download and evaluation guide](docs/USAGE.md#released-checkpoint). The [usage guide](docs/USAGE.md) also covers data layout, configuration, tests, and code entry points. Differences between the example configuration and the ablation setup are listed in [implementation notes](docs/TRAINING_ALIGNMENT.md#experiment-settings-and-public-defaults).

## Credits

Developed in the Department of Electrical Engineering at **National Tsing Hua University**, supervised by **Prof. Yeong-Luh Ueng**. **Jin-Cheng Lin** primarily handled model training; **Hsin-Chang Tsai** primarily handled inference and analysis.

Based on Z. Liu, X. Liang, and J. Lin, [*Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*](https://doi.org/10.1109/ICASSP49660.2025.10889943), ICASSP 2025, and [Uncertainty-oriented Order Learning (UOL)](https://arxiv.org/abs/2409.00603). Dataset access is governed by the [SCUT-FBP5500 terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
