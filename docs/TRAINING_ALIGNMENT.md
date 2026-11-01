# Implementation notes

This repository follows [LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943) for multimodal pairwise comparison and [UOL](https://arxiv.org/abs/2409.00603) for reference-based score estimation. The sections below identify the implemented choices; [project experiments](EXPERIMENTS.md) records the historical measurements and their evaluation scope.

## First-token objective

LMOL §II-C–D expresses the label as an answer string and trains with a conditional language-modeling objective over the answer sequence. This implementation uses a three-class distribution derived from the first answer position:

- [`data/instruction_dataset.py`](../data/instruction_dataset.py) encodes the prompt without appending the answer, supplies the ground-truth class index, and obtains three distinct candidate vocabulary IDs.
- [`training/train.py`](../training/train.py) selects those three logits at the last non-padding prompt position and applies cross-entropy.
- [`evaluation/inference.py`](../evaluation/inference.py) uses the same prompt construction, position selection, and candidate order. It chooses the highest-scoring class without `model.generate`.

The candidate order is `First.`, `Second.`, `Similar.`, as defined in [`data/pairs.py`](../data/pairs.py). A candidate is represented by the first token of its answer string; the punctuation is not an additional supervised token. Unit tests cover distinct IDs, prompt-only inputs, padding, and training/inference choice consistency. No matched full-answer-versus-first-token performance ablation is reported.

## Model and optimization

Two images are encoded separately at 336 × 336 resolution with shared LLaVA-1.5-7B vision/projector weights. The CLIP vision tower is frozen; the multimodal projector and language-model LoRA adapters are trainable. The language model is loaded in 4-bit precision.

LoRA uses rank 8 and alpha 32, giving a scaling factor of 4. AdamW uses separate projector and LoRA learning rates with cosine decay. The [example YAML](../configs/lmol_scut_fbp5500.yaml) sets these learning rates to 0.00002 and 0.0002, respectively.

## Data splits and references

[`data/scut_fbp5500.py`](../data/scut_fbp5500.py) generates deterministic seeded folds. It does not load official paper folds or the original project's random split. The default offline pair sampler uses θ = 0.2 and up to 30,000 examples per class.

References are selected only from the training split: at most 10 images per non-empty 0.1-wide score bin by default. Set `evaluation.reference_max_per_bin: null` to remove the cap. Test-reference pairs are used for final evaluation; training-time monitoring requires an explicit, training-only validation pair file.

## Score recovery and historical results

[`evaluation/uol_bradley_terry.py`](../evaluation/uol_bradley_terry.py) computes a three-class likelihood for each target/reference comparison and maximizes their joint log likelihood over the **unique reference-score support**. Default parameters are `bt_delta: 2.0` and `bt_k: 10.0`; they are public implementation choices, not confirmed LMOL settings. Here the transition offset is delta divided by k, which is 0.2 with these defaults.

The historical project report instead describes searching a **0.001-spaced grid over [1, 5]** with a threshold parameterization. Its five-epoch/effective-batch-64 runs also differ from the public example's one epoch/effective batch 16. The current Quickstart therefore runs the public pipeline rather than recreating the report's result table. See [evaluation protocol](EXPERIMENTS.md#evaluation-protocol) for the sampled baseline and the paper's separate cross-validation results.

Balanced three-class training and evaluation are implemented. Historical imbalanced/two-class training, weighted CE, reversed-pair KL, and checkpoint ensembling are documented in [Experiments](EXPERIMENTS.md), without runnable variant flags in this release.
