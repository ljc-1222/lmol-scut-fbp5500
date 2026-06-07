# LMOL Training Alignment

Source checked: `reference/Large_Multimodal_Model_is_a_Better_Comparator_on_Facial_Beauty_Prediction.pdf` and DOI `10.1109/ICASSP49660.2025.10889943`.

This note documents how the repository maps the paper method to the released implementation. It is intended to make reproducibility and method deviations explicit for public users.

## Paper-Visible Setup

The paper explicitly supports these implementation points:

- Base model: LLaVA-1.5-7B.
- Vision encoder: pretrained CLIP ViT-L/14-336px with 336x336 input resolution.
- Language backbone: Vicuna-1.5.
- Projector: same as LLaVA-1.5, two fully connected layers.
- Image handling: two face images are encoded separately with shared CLIP/projector weights, then visual feature sequences are concatenated in sequence dimension.
- Task: learn facial-beauty order as conditional language modeling over `First.`, `Second.`, and `Similar.`.
- Pair construction: offline sampling from all possible training pairs, balanced across the three ordinal relations.
- Dataset protocol: train on SCUT-FBP5500, five cross-validation folds, 90K training pairs per fold for one epoch.
- Training: AdamW, separate learning rates for projector and LoRA, cosine annealing to minimum learning rate 0.
- Evaluation: Bradley-Terry-based score estimation and PC/MAE/RMSE metrics.

The paper states LoRA rank `r=8` and scale `4`. In PEFT, the LoRA scaling factor is `lora_alpha / r`, so this repository uses `lora_r: 8` and `lora_alpha: 32`.

## Current Implementation Alignment

Aligned in the repo:

- `configs/lmol_scut_fbp5500.yaml` uses `llava-hf/llava-1.5-7b-hf`, 336 image size, five folds, 90K balanced training pairs, one epoch, AdamW + cosine, and separate projector/LoRA learning rates.
- `data/pairs.py` performs offline balanced pair sampling and keeps exact labels `First.`, `Second.`, `Similar.`.
- `data/instruction_dataset.py` keeps the two images as separate PIL images. They are not concatenated at pixel level.
- `model/lmol.py` freezes the vision tower, trains the full multimodal projector, and trains LoRA adapters on the language model. LoRA targets are resolved to exact language-model module names before PEFT wrapping, so CLIP modules with matching suffixes such as `q_proj`/`v_proj` are excluded.
- `evaluation/bradley_terry.py` estimates absolute scores from predicted pairwise orders.

Implementation details users should know:

- Prompts are rendered through the LLaVA processor chat template, so training and evaluation use the instruction format expected by LLaVA-1.5.
- The default config uses `per_device_batch_size: 1` and `gradient_accumulation_steps: 16`, giving effective batch size 16.
- The practical training default is `training.objective: choice_token`, which applies 3-way cross entropy to LM head logits for the first answer word tokens (`First`, `Second`, `Similar`) at the prompt boundary.
- Likelihood scoring uses normalized log-probabilities over those three first answer word tokens; the shared period in `First.` / `Second.` / `Similar.` is not scored.
- Label and eval positions are located in the processor-expanded `input_ids`, not with raw tokenizer prompt length. This matters because the LLaVA processor expands each `<image>` into many image tokens.
- 4-bit loading skips `vision_tower`, `multi_modal_projector`, `mm_projector`, and `lm_head`. This keeps CLIP frozen in floating point and keeps the full projector trainable.

## Token And Label Handling Differences

The paper defines the order-learning answer as a natural-language string `Xanswer` and writes the training objective as standard conditional language modeling over answer tokens `X_i`. The implementation keeps the paper's three answer strings and order semantics, but it does not optimize or evaluate the full answer string by default.

| Area | Paper-visible method | Current implementation | Practical consequence |
| --- | --- | --- | --- |
| Answer strings | `Similar.`, `Second.`, and `First.` are natural-language labels for `xv1 ~= xv2`, `xv1 < xv2`, and `xv1 > xv2`. | `data/pairs.py` uses the same semantic labels, ordered internally as `First.`, `Second.`, `Similar.`. The class order is fixed by `VALID_ANSWERS` / `ANSWER_TO_INDEX`. | The semantics match the paper, but code that consumes logits or confusion matrices must preserve the repo's internal order. |
| Label threshold | The paper states `theta = 0.2`. | `label_from_scores()` returns `Similar.` when `abs(score1 - score2) <= theta + 1e-12`; otherwise the larger score determines `First.` or `Second.`. | Boundary cases at exactly 0.2 are treated as `Similar.` in this code path. |
| Tokenization | The paper treats the answer as a sequence of length `L`. | The LLaVA tokenizer splits each answer string into two tokens: `First.` -> `['▁First', '.']`, `Second.` -> `['▁Second', '.']`, `Similar.` -> `['▁Similar', '.']`. `LMOLDataCollator.answer_token_ids()` keeps only the first token ID. | The shared period token is ignored by both the default training objective and pairwise likelihood evaluation. |
| Default training objective | Formula (9)/(10) describes conditional LM over all answer tokens. | `training/train.py` with `objective: choice_token` discards `labels`, takes logits at `choice_positions`, indexes only the three first answer word token IDs, and applies 3-way cross entropy. | The objective is discriminative token choice through the LM head, not full-vocabulary autoregressive answer-string training. |
| `conditional_lm` branch | A full paper-style implementation would keep every answer token unmasked and optimize the model loss over the whole answer string. | The current collator masks everything except the first answer word token (`labels[row_idx, answer_start + 1:] = -100`), so `conditional_lm` also supervises only that token. | Do not claim `conditional_lm` is a full multi-token paper objective unless the collator mask is changed. |
| Inference | The paper says the LLM outputs the order in natural language and then Bradley-Terry converts predicted orders into scores. | `evaluation/inference.py` does not call `model.generate`; it renders the prompt only, reads final-position logits, normalizes across the three first answer word tokens with `log_softmax`, and picks the max. | Pairwise accuracy and NLL are calibrated to the 3-way token-choice objective. They are not free-form generation or full-answer likelihood metrics. |

Two implementation details follow from this design:

- The current prompt text avoids the words `First`, `Second`, and `Similar`. `find_answer_token_position()` searches the processor-expanded `input_ids` for the first matching answer word token, so adding these answer words to the prompt would require a more robust answer-span locator.
- To implement the paper's full conditional-LM objective, keep all answer tokens unmasked in `LMOLDataCollator.__call__`, train with the model's native LM loss, and update evaluation to either sum full-answer log-likelihoods for all candidates or generate and parse the answer string.

## Not Publicly Specified

These details are not fully recoverable from the public paper text and should be treated as experiment knobs rather than claimed reproduction facts:

- Exact official fold files or fold seed, if any.
- Exact LoRA target module list.
- Exact reference-image selection strategy and count for Bradley-Terry score estimation.
- Exact offline sampling seed/order.
- Whether the authors used the LLaVA chat template explicitly in code. The paper describes image-text instruction format; this repository uses the processor chat template because it is the standard LLaVA-1.5 implementation path.
