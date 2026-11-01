# Implementation notes: published LMOL vs. this repository

This repository is an **LMOL-style reimplementation**, not a byte-for-byte reproduction of the [ICASSP 2025 paper](https://doi.org/10.1109/ICASSP49660.2025.10889943). The points below describe boundaries that are easy to miss when reading the code.

## Main adaptation: constrained answer-token classification

The original LMOL paper's Section II-D (Equations 9–10) specifies a conditional language-modeling objective over the answer sequence (`First.`, `Second.`, or `Similar.`). The published description presents natural-language answers, but does not fully specify an inference decoding implementation.

Here the task is explicitly reduced to **three candidate first tokens at the prompt boundary**:

- [`data/instruction_dataset.py`](../data/instruction_dataset.py) prepares two images, the instruction prompt, the class index, and the candidate token IDs. The three IDs must be distinct.
- [`training/train.py`](../training/train.py) takes the next-token logits at the **last non-padding prompt position** and computes cross-entropy over only the three candidate logits. It does not compute full-answer causal language-modeling loss.
- [`evaluation/inference.py`](../evaluation/inference.py) uses the same prompt boundary and token IDs, applies a three-way log-softmax, and selects the maximum candidate. It does **not** call `model.generate`.

The candidate probability is a **softmax conditional on these three candidates**, not the original model's full-vocabulary probability or the probability of the entire answer string. Punctuation and later answer tokens are not supervised. This choice removes free-form answer parsing at inference, but no controlled comparison of speed or quality against sequence-level LMOL training is established here.

## What remains aligned

- Base model: LLaVA-1.5-7B, with two images passed through the shared CLIP vision tower and built-in multimodal projector.
- Order labels: `First.`, `Second.`, `Similar.` with score-difference threshold `theta=0.2`.
- Training data: offline balanced three-class pairs, rather than all possible image pairs.
- Parameter-efficient updates: frozen vision tower, trainable projector and language-model attention LoRA; 4-bit loading when enabled.
- Prediction-to-score pipeline: compare a test target to known-score training references, then estimate a scalar with a UOL-style Bradley–Terry likelihood.

## Explicit deviations and limits

- **Splits:** this code creates seeded 80/20 folds. It does not load the paper's official/reported fold assignments.
- **Training:** [the default YAML](../configs/lmol_scut_fbp5500.yaml) is a runnable example (1 epoch, effective batch size 16); it does **not** reproduce the 5-epoch, effective-batch-64 historical student experiments.
- **Training-time validation:** disabled by default to avoid monitoring the held-out test-reference pairs. Optional monitoring requires a separately prepared validation pair file.
- **Score estimation:** the current implementation searches the distinct reference-score values as candidate scores, with configurable `bt_delta` and `bt_k`. It is not identical to the historical report's fine-grained score grid. The original LMOL paper does not fully determine these implementation choices.
- **Scope:** the two-class, imbalanced three-class, weighted CE, KL consistency, and checkpoint ensemble experiments conducted in the student project are not included as training modes in this public release.
- **Performance:** no historical metric is represented as a verified outcome of the current default configuration.

## Source

Z. Liu, X. Liang, and J. Lin, *Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*, ICASSP 2025 ([DOI](https://doi.org/10.1109/ICASSP49660.2025.10889943)). The order-learning/score-estimation lineage also includes [UOL](https://arxiv.org/abs/2409.00603). Check these publications for the original method rather than treating this repository's defaults as authoritative LMOL hyperparameters.
