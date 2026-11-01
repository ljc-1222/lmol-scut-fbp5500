# LMOL on SCUT-FBP5500

**A LLaVA-1.5-7B reimplementation of pairwise facial-beauty comparison, with constrained first-token classification.**

This repository contains the core training and evaluation pipeline developed for an undergraduate research project at **National Tsing Hua University**. It builds on [LMOL (ICASSP 2025)](https://doi.org/10.1109/ICASSP49660.2025.10889943): given two facial images, a multimodal model decides whether the **first**, **second**, or **neither** is preferred under the SCUT-FBP5500 labels. Pairwise decisions can then be converted to dataset-scale scores using reference-based Bradley–Terry estimation.

The main implementation change is the **training and inference interface**: rather than optimizing the likelihood of an entire textual answer, this version directly supervises three answer-token logits.

## Key implementation: first-token classification

The original LMOL paper formulates training as conditional language modeling over an answer such as `First.`, `Second.`, or `Similar.` (Section II-D, Equations 9–10). Our implementation treats the **next token at the end of the instruction prompt** as a three-way classification decision:

1. Select the vocabulary logits for the first tokens of `First.`, `Second.`, and `Similar.`.
2. Normalize **only those three logits** with softmax.
3. Apply three-class cross-entropy to the corresponding label. At inference, choose the highest-scoring candidate without generating a free-form answer.

```python
# After a forward pass over the two images and the instruction prompt:
next_token_logits = outputs.logits[row_indices, choice_positions, :].float()
choice_logits = next_token_logits.index_select(-1, choice_token_ids)
loss = F.cross_entropy(choice_logits, choice_labels)
```

This keeps the classifier's decision space explicit and makes training and prediction use the same three candidates. It also avoids text-generation and answer-format parsing at inference. It is an **implementation adaptation**, not a claim of a controlled accuracy or speed improvement over the original LMOL objective.

**Relevant code:** [prompt & token choices](data/instruction_dataset.py) · [training objective](training/train.py) · [pairwise inference](evaluation/inference.py) · [paper-to-code differences](docs/TRAINING_ALIGNMENT.md)

## What's included

- **Pairwise dataset preparation:** deterministic 80/20 train/test folds, score-difference labels (threshold `0.2`), and balanced offline sampling.
- **Parameter-efficient multimodal fine-tuning:** `llava-hf/llava-1.5-7b-hf`, a frozen CLIP vision tower, a trainable LLaVA projector, and 4-bit QLoRA on language-model attention layers.
- **Training utilities:** separate projector/LoRA learning rates, gradient accumulation, configurable checkpointing, and optional Weights & Biases logging.
- **Evaluation:** constrained first-token predictions, training-split-only reference selection, UOL-style Bradley–Terry score estimation, and PC/MAE/RMSE metrics.
- **Checks:** data-split leakage validation, CPU unit tests, and a checkpoint-independent dry run.

## Quick start

Requires **Python 3.11**, and a **CUDA GPU with sufficient VRAM** for LLaVA fine-tuning or inference. The dataset is obtained separately under its original terms; no SCUT-FBP5500 images or labels are redistributed here.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime

python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
python scripts/build_pairs.py --fold-index 0
python scripts/train.py --fold-index 0 --dry-run
python scripts/train.py --fold-index 0
```

Once training finishes, evaluate its saved checkpoint:

```bash
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

**Training and full target/reference evaluation are GPU-intensive.** The default configuration is a lightweight *example setup*, not the exact schedule used in the original student project or LMOL paper. See [Usage](docs/USAGE.md) for dataset layout, optional evaluation and logging, checkpoint handling, and smoke tests.

Run the CPU unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Repository structure

| Location | Purpose |
| --- | --- |
| [`data/`](data/) | Annotations, pair sampling, prompt collation, reference selection |
| [`model/`](model/) | LLaVA / QLoRA setup and adapter/projector checkpoints |
| [`training/`](training/) | First-token cross-entropy, optimizer, training loop |
| [`evaluation/`](evaluation/) | Pairwise token scoring, Bradley–Terry estimation, metrics |
| [`scripts/`](scripts/) | CLI entry points for preparing, training, and evaluating |
| [`configs/`](configs/) | Typed configuration and an example YAML |
| [`tests/`](tests/) | CPU tests for data and core model/estimation logic |

## Project scope and attribution

The student project also explored **different class distributions, a two-class setting, weighted cross-entropy, KL-based order-consistency regularization, and checkpoint ensembling**. These historical experiments are **not implemented as selectable training modes in this public release**. This repository is the core balanced three-class pipeline; it does not include the student report or experimental artifacts.

**Student project:** National Tsing Hua University, Department of Electrical Engineering (2025). **Jin-Cheng Lin** focused on model training; **Hsin-Chang Tsai** worked on inference and analysis. **Advisor:** Prof. Yeong-Luh Ueng.

**Prior work:** Z. Liu, X. Liang, and J. Lin, *Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*, ICASSP 2025 ([DOI](https://doi.org/10.1109/ICASSP49660.2025.10889943)); and [Uncertainty-oriented Order Learning (UOL)](https://arxiv.org/abs/2409.00603).

**Data and weights:** [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) is released for non-commercial research under the dataset authors' terms. The [LLaVA-1.5-7B checkpoint](https://huggingface.co/llava-hf/llava-1.5-7b-hf) has separate model/license terms. The historical [checkpoint package](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500) is a separate artifact; its original run configuration should not be conflated with the current example defaults. Facial-attractiveness labels are subjective human ratings and should not be treated as objective judgments or used for consequential decisions.

No license is currently granted for reuse of this repository's source code; the dataset and model licenses are independent.
