# LMOL on SCUT-FBP5500

**Reproduction and experimental study of multimodal pairwise facial-beauty prediction.**

This repository grew out of a 2025 undergraduate research project at **National Tsing Hua University**. We reconstructed an [LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943)-style pipeline using **LLaVA-1.5-7B**, investigated practical training choices under a single-GPU budget, and studied how a model's pairwise decisions affect the final predicted scores.

The public code contains the **core balanced three-class implementation**: data preparation, QLoRA fine-tuning, pairwise comparison, and Bradley–Terry-based score estimation. Other experiments conducted during the project are described below but are not included as runnable variants.

## Project overview

Rather than regressing a beauty score from a single image, order learning compares two images and predicts **`First.`**, **`Second.`**, or **`Similar.`**. A set of comparisons against reference images with known scores is then used to estimate a continuous score. The dataset is [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), which contains human-rated facial photographs; the task models *subjective dataset annotations*, not objective attractiveness.

Our work combined three parts:

- **Reconstruct the pipeline.** Prepare balanced offline image pairs, train an LMM comparator with frozen CLIP features and QLoRA, and connect the comparator to reference-based score recovery.
- **Adapt the classification objective.** Use the language model's first answer-token logits for a three-way cross-entropy loss and constrained inference, instead of supervising an entire generated answer string.
- **Investigate design trade-offs.** Compare label spaces and class distributions, weighted and consistency-regularized objectives, and nearby-checkpoint ensembling; examine the relationship between pairwise classification and downstream score quality.

## Implementation

| Stage | Public implementation |
| --- | --- |
| **Data** | Deterministic train/test folds, offline sampling of `First/Second/Similar` pairs (`θ = 0.2`), and score-binned references selected only from training images. [`data/`](data/) |
| **Model** | [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) with a frozen CLIP vision tower, trainable multimodal projector, and 4-bit QLoRA on language-model attention modules. [`model/`](model/) |
| **Training** | Three-way cross-entropy on restricted next-token logits, gradient accumulation, separate projector/LoRA learning rates, and checkpoint saving. [`training/`](training/) |
| **Evaluation** | First-token pairwise predictions, UOL-style Bradley–Terry score recovery, and Pearson correlation / MAE / RMSE. [`evaluation/`](evaluation/) |

### Why first-token classification?

The original LMOL paper specifies a conditional language-modeling objective over the answer text (Section II-D). In this reimplementation, we take the logits at the answer position, keep only the token IDs corresponding to the three possible answers, and train with **three-class cross-entropy**. Inference selects the highest-scoring candidate without generating or parsing a free-form answer.

This gives the comparator an explicit three-class decision interface. It is a **deliberate implementation difference**, not a claim that we established a controlled performance or speed improvement over the paper's original objective. See [method differences](docs/TRAINING_ALIGNMENT.md), [training code](training/train.py), and [inference code](evaluation/inference.py).

## What we learned in the project

Beyond the public baseline, the original project explored **balanced and imbalanced three-class training, a two-class formulation, weighted cross-entropy, reversed-pair KL consistency, and checkpoint ensembling**.

Our experiments suggested that **pairwise accuracy alone is an incomplete measure of score-estimation quality**: the treatment of ambiguous `Similar` comparisons mattered when recovering continuous scores. Reweighting classes or enforcing order-reversal consistency changed intermediate behavior without consistently improving the final scores. Combining checkpoints yielded a modest score improvement at additional inference cost.

These observations come from the original student project. **The experimental variants, logs, figures, and original report are not part of this public repository**, and the current example configuration should not be treated as the original experiment setup.

## Getting started

The pinned runtime targets **Python 3.11 on Linux/WSL2 with an NVIDIA CUDA GPU** for model training and inference. Check the CUDA/PyTorch wheel compatibility of your environment; see the [PyTorch 2.4.1 installation options](https://docs.pytorch.org/get-started/previous-versions/). The CPU unit tests do not require the dataset or model weights.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime

# Download SCUT-FBP5500 after reviewing its non-commercial research terms.
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py

python scripts/build_pairs.py --fold-index 0
python scripts/train.py --fold-index 0 --dry-run
python scripts/train.py --fold-index 0 --no-wandb
```

To evaluate a checkpoint produced by the training command:

```bash
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

Running a 7B model and evaluating all target/reference pairs requires significant compute. The example YAML uses its own seeded folds and training schedule; it does **not** recreate the original report's exact split or settings.

For data layout, separate inference/score-estimation commands, checkpoint options, and diagnostic limits, see [Usage](docs/USAGE.md). To run unit tests:

```bash
python -m unittest discover -s tests -v
```

## Project and credits

Developed as an undergraduate project in the **Department of Electrical Engineering, National Tsing Hua University (2025)**, supervised by **Prof. Yeong-Luh Ueng**. **Jin-Cheng Lin** worked primarily on model training; **Hsin-Chang Tsai** worked primarily on inference and analysis. The project jointly investigated model behavior and training alternatives.

This is an independent **LMOL-style reimplementation**, not the official implementation. The underlying approach comes from Z. Liu, X. Liang, and J. Lin, [*Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*](https://doi.org/10.1109/ICASSP49660.2025.10889943), ICASSP 2025, building on [UOL](https://arxiv.org/abs/2409.00603). SCUT-FBP5500 has separate [non-commercial research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release); dataset images and labels are not distributed here. An [earlier checkpoint package](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500) is available separately, but is not guaranteed to match this repository's example configuration.
