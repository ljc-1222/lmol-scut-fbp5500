# Fold0 Checkpoint Files

This folder contains the fold0 LMOL checkpoint package used by the GitHub repository evaluation examples.

Required files:

- `adapter_config.json`
- `adapter_model.safetensors`
- `projector.pt`
- `tokenizer.json`
- `tokenizer_config.json`
- `processor_config.json`
- `preprocessor_config.json`
- `special_tokens_map.json`
- `chat_template.json`

`projector.pt` is required because this implementation trains and saves the full LLaVA multimodal projector separately from the PEFT adapter.

Objective and inference details:

- The fold0 checkpoint uses `training.objective: choice_token`.
- `First.`, `Second.`, and `Similar.` are the three labels, ordered internally as `First.`, `Second.`, `Similar.`.
- The LLaVA tokenizer splits each answer string into the answer word token plus a shared period token. Training and pairwise evaluation score only the answer word token.
- Evaluation uses prompt-only normalized likelihood over the three answer word tokens, not free-form generation.

Use this folder as:

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```
