# Implementation notes

The implementation combines an LLaVA comparator with reference-based score estimation. This page explains the first-token objective and the settings used by the public code.

## First-token objective

LMOL §II-C–D defines natural-language answers and a conditional language-modeling objective over the complete answer sequence. Here, the decision is a three-class distribution over the first answer-token logits:

- [`data/instruction_dataset.py`](../data/instruction_dataset.py) renders the two-image prompt without appending the answer, obtains the three candidate vocabulary IDs, and supplies the class index.
- [`training/train.py`](../training/train.py) selects those logits at the last non-padding prompt position and computes cross-entropy.
- [`evaluation/inference.py`](../evaluation/inference.py) uses the same prompt and candidate order, then chooses the class with the highest probability.

The class order is `First.`, `Second.`, `Similar.`. Only the first token of each answer string is supervised; the shared period token is excluded. Candidate IDs must be distinct. Inference uses a forward pass over the prompt and does not call `model.generate`.

This aligns the loss and inference decision with the pairwise classification task. The [score comparisons](../README.md#results) include other training and evaluation differences, so they do not quantify the isolated effect of this objective.

## Model and optimization

Both images are encoded separately at **336 × 336** resolution with shared LLaVA-1.5-7B vision/projector weights. The CLIP vision tower is frozen. The full multimodal projector and language-model LoRA adapters are trainable.

The [default configuration](../configs/lmol_scut_fbp5500.yaml) uses 4-bit NF4 quantization for the language model, LoRA rank 8, alpha 32 (scaling factor 4), and dropout 0.05. LoRA targets the language model's attention projections. AdamW applies separate learning rates of 0.00002 to the projector and 0.0002 to LoRA, with cosine decay.

[Checkpoint saving](../model/lmol.py) writes the PEFT adapter, the full projector state as `projector.pt`, and processor files. The projector must be restored alongside the adapter for evaluation.

## Pair construction and references

[`data/scut_fbp5500.py`](../data/scut_fbp5500.py) creates deterministic seeded folds. The default is five folds with split seed 42; it does not load official fold files. [Offline sampling](../data/pairs.py) requests 30,000 pairs per class, with `Similar` defined by an absolute score difference at most 0.2.

[References](../data/references.py) are selected only from the training split, at most 10 per non-empty score bin of width 0.1. Set `evaluation.reference_max_per_bin: null` to remove the cap. Training-time monitoring requires an explicit validation pair file drawn from the training split; test-reference pairs are reserved for final evaluation.

## Score estimation

[`evaluation/uol_bradley_terry.py`](../evaluation/uol_bradley_terry.py) computes UOL-style three-class likelihoods for target/reference comparisons and maximizes their joint log likelihood over the **unique reference-score support**.

The defaults are `bt_delta: 2.0` and `bt_k: 10.0`, giving a transition offset of delta divided by k, or 0.2. These are implementation settings, not confirmed parameters from the LMOL paper.

## Experiment settings and public defaults

The [ablation study](EXPERIMENTS.md#training-configurations) and the example configuration use the following settings:

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Setting</th>
<th width="10000" align="center">Ablation experiments</th>
<th width="10000" align="center">Public example</th>
</tr>
</thead>
<tbody>
<tr><td align="center">Training duration</td><td align="center">Five epochs</td><td align="center">One epoch</td></tr>
<tr><td align="center">Effective batch size</td><td align="center">64</td><td align="center">16</td></tr>
<tr><td align="center">Image split</td><td align="center">One 4,400 / 1,100 split</td><td align="center">Seeded five-fold partition; selected fold used per run</td></tr>
<tr><td align="center">Score-search candidates</td><td align="center">Ratings from 1 to 5 in steps of 0.001</td><td align="center">Unique reference scores</td></tr>
<tr><td align="center">Training variants</td><td align="center">Label-distribution, loss, and ensemble studies</td><td align="center">Balanced three-class CE</td></tr>
</tbody>
</table>
</div>

The example runs the public pipeline; it is not an exact recipe for the ablation tables. See [Usage](USAGE.md) for training and checkpoint evaluation commands.
