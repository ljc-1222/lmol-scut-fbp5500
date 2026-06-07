---
pretty_name: LMOL SCUT-FBP5500 Fold0 Checkpoint
tags:
- lmol
- facial-beauty-prediction
- facial-attractiveness-prediction
- scut-fbp5500
- llava
- qlora
- peft
- checkpoint
---

# LMOL SCUT-FBP5500 Fold0 Checkpoint

This Hugging Face Dataset repository stores the fold0 checkpoint package and small result artifacts for the GitHub repository:

<https://github.com/ljc-1222/lmol-scut-fbp5500>

It is hosted as a Dataset repository to keep the GitHub repository lightweight. The package is not a redistribution of SCUT-FBP5500 images or labels.

## Paper

This checkpoint is for an implementation of:

> Zhenyou Liu, Xuefeng Liang, and Jian Lin, "Large Multimodal Model is a Better Comparator on Facial Beauty Prediction," ICASSP 2025. DOI: [10.1109/ICASSP49660.2025.10889943](https://doi.org/10.1109/ICASSP49660.2025.10889943)

The method uses a large multimodal model as a pairwise comparator. Given two face images, the model predicts `First.`, `Second.`, or `Similar.`, and absolute SCUT-FBP5500 beauty scores are estimated from many pairwise comparisons.

## Implementation Notes

This checkpoint follows the repository's default `choice_token` implementation. The paper writes the answer target as conditional language modeling over the natural-language answer string, but this checkpoint was trained and evaluated with a discriminative token-choice objective:

- Internal class order is `First.`, `Second.`, `Similar.`.
- With the LLaVA tokenizer, each answer string is two tokens: an answer word token plus a shared period token.
- Training and pairwise evaluation use only the first answer word token (`First`, `Second`, or `Similar`) and do not score the period token.
- Pairwise inference is prompt-only likelihood scoring over the three answer word tokens, not free-form `model.generate`.
- The repository's `conditional_lm` branch uses the same current collator mask, so it is not a full multi-token answer-string objective unless the collator is changed.

See the GitHub repository's `docs/TRAINING_ALIGNMENT.md` for the detailed paper-to-code alignment notes.

## Repository Layout

```text
.
├── README.md
├── checkpoints/
│   └── fold0/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── projector.pt
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── processor_config.json
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── chat_template.json
│       └── README.md
└── results/
    └── fold0/
        ├── metrics.json
        ├── confusion_matrix.json
        ├── progress.json
        └── README.md
```

## Download

Run the download command from the GitHub repository root. It writes files to `checkpoints/fold0/`, which is the path expected by the evaluation examples.

```bash
pip install -U huggingface_hub
hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" \
  --local-dir .
```

To download both the checkpoint and result JSON files:

```bash
hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" "results/fold0/*" \
  --local-dir .
```

Python equivalent:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ljc-1222/LMOL-fbp5500",
    repo_type="dataset",
    allow_patterns=["checkpoints/fold0/*"],
    local_dir=".",
)
```

## Use With The GitHub Repository

After installing dependencies, preparing SCUT-FBP5500, and building fold0 pairs, run:

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```

Full fold0 evaluation scores 374,000 target-reference pairs and can take many hours. Use `--limit 256` only for a checkpoint smoke test; it does not reproduce the reported metrics.

## Fold0 Result

Score metrics from `results/fold0/metrics.json`:

| Split | PC ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| Fold 0 | 0.9568 | 0.1643 | 0.2130 |

Pairwise metrics computed from `results/fold0/confusion_matrix.json` and `results/fold0/progress.json`:

| Split | Accuracy ↑ | Macro Precision ↑ | Macro Recall ↑ | Macro F1 ↑ | Weighted F1 ↑ | Pairwise NLL ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fold 0 | 0.8649 | 0.8145 | 0.8354 | 0.8195 | 0.8723 | 0.3384 |

Per-answer pairwise metrics:

| Gold Label | Support | Predicted | Precision ↑ | Recall ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `First.` | 120,836 | 110,502 | 0.9335 | 0.8537 | 0.8918 |
| `Second.` | 194,770 | 185,020 | 0.9557 | 0.9079 | 0.9312 |
| `Similar.` | 58,394 | 78,478 | 0.5541 | 0.7446 | 0.6354 |

## Checkpoint Details

- Base model: `llava-hf/llava-1.5-7b-hf`
- Trainable modules: PEFT LoRA adapters on the language model and the full multimodal projector.
- Frozen modules: CLIP visual encoder.
- LoRA: `r=8`, `lora_alpha=32`, `lora_dropout=0.05`.
- Training objective: `choice_token` 3-way cross entropy over the first answer word tokens.
- Inference: prompt-only normalized likelihood over the same three answer word tokens.
- Fold: deterministic SCUT-FBP5500 fold0 with `split_seed: 42`.

The `projector.pt` file is required. Loading only the PEFT adapter is incomplete for this implementation.

## Limitations And Terms

- This repository contains a checkpoint package and result metadata only. It does not contain SCUT-FBP5500 images or labels.
- Users must obtain SCUT-FBP5500 from the official source and comply with the dataset authors' terms.
- The GitHub repository currently does not specify a project license. Reuse rights should be clarified before redistribution.
- Facial beauty prediction is sensitive and subjective. Use the checkpoint only for research reproduction and with appropriate ethical review.

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
