# LMOL Method Alignment

This note records the reproducibility boundary between the LMOL paper (DOI `10.1109/ICASSP49660.2025.10889943`), the UOL formulation (arXiv `2409.00603`), and this implementation.

## Paper-Aligned Path

- LLaVA-1.5-7B uses two separately encoded images at 336×336 resolution.
- Offline training pairs are balanced across `First.`, `Second.`, and `Similar.` with `theta=0.2`.
- The vision tower is frozen; the full multimodal projector and language-model LoRA adapters are trainable.
- LoRA uses rank 8 and alpha 32, giving PEFT scaling `alpha / r = 4`.
- AdamW uses separate projector and LoRA learning rates with cosine decay.
- References come only from the fold training split; score recovery uses UOL Bradley-Terry likelihood over reference-score support.

## Deliberate Implementation Difference

The paper describes conditional language modeling over the answer text. This repository uses the same prompt and answer semantics but trains a three-way classifier from LM-head logits at the prompt boundary:

- `data/instruction_dataset.py` retains only the first answer-token label.
- `training/train.py` applies cross entropy over the three answer-token IDs.
- `evaluation/inference.py` normalizes those three logits and does not call `model.generate`.

Results from this code should therefore be described as first-token LMOL comparator results, not full-answer conditional-LM results.

## Reproducibility Assumptions

- Folds are always deterministic seeded splits produced by `data/scut_fbp5500.py`; official fold files are not detected or loaded.
- The default prompt is paper-style but configurable. Changing it is a prompt ablation.
- Reference scores and images come from the training split. Test scores are used only for final metrics.
- The default reference cap is 10 images per non-empty 0.1-wide score bin. Set `evaluation.reference_max_per_bin: null` to remove the cap.
- UOL parameters `evaluation.bt_delta: 2.0` and `evaluation.bt_k: 10.0` are implementation choices because paper-specific values were not available in the referenced LMOL materials.
- Exact official fold seed, LoRA target list, offline pair order, and chat-template implementation were not publicly specified and should not be claimed as exact reproduction facts.
