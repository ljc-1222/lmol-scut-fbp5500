# LMOL on SCUT-FBP5500

Multimodal order learning for facial beauty prediction with **LLaVA-1.5-7B** and **4-bit QLoRA**. The model compares two faces as `First`, `Second`, or `Similar`; Bradley–Terry estimation then converts comparisons against scored reference images into a continuous rating.

This project reimplements the [LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943) pipeline, introduces **cross-entropy classification over three first-answer-token logits**, and evaluates how label distributions, training objectives, and checkpoint ensembling affect the final scores.

## Results

Historical results on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), transcribed from the 2025 project report (Tables 3-2–3-4). Training used one **4,400 / 1,100 train/test split**, five epochs, and a single GPU.

| Configuration | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| Balanced three-class, CE | 95.26 | 0.1730 | 0.2150 |
| Imbalanced three-class (4:4:2), CE | 94.76 | 0.1760 | 0.2290 |
| Two-class, CE | 93.35 | 0.2345 | 0.2916 |
| Balanced three-class, weighted CE | 94.52 | 0.1771 | 0.2304 |
| Balanced three-class, CE + KL consistency | 95.21 | 0.1730 | 0.2160 |
| **Balanced three-class, five-checkpoint ensemble** | **95.32** | **0.1720** | **0.2141** |

The report explicitly evaluates the baseline BT scores on **200 sampled test images**; later tables do not restate the target count. The original LMOL paper reports **95.65 / 0.1552 / 0.2032** under its cross-validation protocol, so its figures are context rather than a matched-protocol comparison.

Removing `Similar` raised pairwise accuracy from **85.84% to 92.07%**, while PC fell from **95.26% to 93.35%**. KL regularization reached **98.36%** forward/reverse agreement without improving the baseline score metrics. The best result averages logits from five checkpoints between epochs 4 and 5, requiring five comparator evaluations per pair.

See [experiment settings and classwise results](docs/EXPERIMENTS.md). These are reported project measurements; the public code provides the core balanced three-class pipeline rather than every experimental variant.

## Method and implementation

- **Pairwise data.** Sample 30,000 training pairs per class using a score-difference threshold of θ = 0.2. Select known-score references only from training images, with up to 10 images per 0.1-wide score bin.
- **Multimodal fine-tuning.** Encode both images separately with a frozen CLIP ViT-L/14 at 336 × 336 resolution. Train LLaVA's multimodal projector and language-model LoRA adapters with 4-bit quantization, gradient accumulation, and separate learning rates.
- **First-token classification.** The LMOL paper trains on the answer sequence with a conditional language-modeling objective (§II-D). Here, cross-entropy is computed over only the three vocabulary logits corresponding to `First`, `Second`, and `Similar` at the first answer position. Inference uses the same three candidates directly, without generating and parsing an answer string.
- **Score estimation.** Compare each target with training-set references and recover a continuous score with a Bradley–Terry likelihood; evaluate Pearson correlation (PC), MAE, and RMSE.

The first-token objective is a project modification to LMOL. The results above do not isolate its effect against full-answer supervision. [Implementation notes](docs/TRAINING_ALIGNMENT.md) document the objective, split generation, and score-estimator differences.

## Getting started

Use **Python 3.11** and an NVIDIA CUDA GPU for model training and inference. Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime

# Review SCUT-FBP5500's non-commercial research terms before downloading.
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
python scripts/build_pairs.py --fold-index 0

python scripts/train.py --fold-index 0 --dry-run
python scripts/train.py --fold-index 0 --no-wandb

python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

The [example configuration](configs/lmol_scut_fbp5500.yaml) uses one epoch, an effective batch size of 16, generated seeded folds, and the public UOL-style estimator. It runs the core pipeline; it does not reproduce the historical table's five-epoch setup and grid-search estimator. See [Usage](docs/USAGE.md) for data layout, checkpoint options, separate inference commands, and runtime requirements.

An [earlier checkpoint package](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500) is available on Hugging Face; its layout and split must be checked before use with the current loader.

## Code guide

| Component | Entry points |
| --- | --- |
| Pair construction, splits, references | [`data/`](data/), [`scripts/build_pairs.py`](scripts/build_pairs.py) |
| LLaVA loading, QLoRA, checkpoint management | [`model/lmol.py`](model/lmol.py) |
| Prompt construction and first-token CE | [`data/instruction_dataset.py`](data/instruction_dataset.py), [`training/train.py`](training/train.py) |
| Pairwise inference and score recovery | [`evaluation/`](evaluation/), [`scripts/evaluate_fold.py`](scripts/evaluate_fold.py) |

CPU unit tests use synthetic data and do not require model weights or the dataset:

```bash
python -m unittest discover -s tests -v
```

## Credits

An undergraduate research project at **National Tsing Hua University, Department of Electrical Engineering (2025)**, supervised by **Prof. Yeong-Luh Ueng**. **Jin-Cheng Lin** primarily handled model training; **Hsin-Chang Tsai** primarily handled inference and analysis.

Based on Z. Liu, X. Liang, and J. Lin, [*Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*](https://doi.org/10.1109/ICASSP49660.2025.10889943), ICASSP 2025, and [Uncertainty-oriented Order Learning (UOL)](https://arxiv.org/abs/2409.00603). This is an independent reimplementation. SCUT-FBP5500's [non-commercial research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) apply separately from the code and base model; dataset images and labels are not distributed here.
