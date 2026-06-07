# Hugging Face Dataset Upload

This maintainer note documents how to rebuild the Hugging Face Dataset repository package for the fold0 checkpoint and small result artifacts.

Public users normally only need the download commands in the repository `README.md`.

Target Dataset repository:

```text
https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500
```

The uploaded repository is organized as:

```text
.
├── README.md
├── checkpoints/fold0/
└── results/fold0/
```

The checkpoint files come from:

```text
outputs/lmol_scut_fbp5500/fold_0/last/
```

The result files come from:

```text
docs/results/fold_0/
```

## Rebuild Upload Staging

Run from the GitHub repository root:

```bash
STAGING="${TMPDIR:-/tmp}/lmol_hf_dataset_upload"
rm -rf "$STAGING"
mkdir -p "$STAGING/checkpoints/fold0" "$STAGING/results/fold0"

cp docs/HF_DATASET_CARD.md "$STAGING/README.md"
cp docs/HF_CHECKPOINT_README.md "$STAGING/checkpoints/fold0/README.md"
cp outputs/lmol_scut_fbp5500/fold_0/last/adapter_config.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/adapter_model.safetensors "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/chat_template.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/preprocessor_config.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/processor_config.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/projector.pt "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/special_tokens_map.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/tokenizer.json "$STAGING/checkpoints/fold0/"
cp outputs/lmol_scut_fbp5500/fold_0/last/tokenizer_config.json "$STAGING/checkpoints/fold0/"

cp docs/results/fold_0/README.md "$STAGING/results/fold0/README.md"
cp docs/results/fold_0/confusion_matrix.json "$STAGING/results/fold0/"
cp docs/results/fold_0/metrics.json "$STAGING/results/fold0/"
cp docs/results/fold_0/progress.json "$STAGING/results/fold0/"
```

## Upload

```bash
pip install -U huggingface_hub
hf auth login

hf upload ljc-1222/LMOL-fbp5500 "$STAGING" . \
  --repo-type dataset \
  --commit-message "update fold0 checkpoint package" \
  --delete adapter_config.json adapter_model.safetensors chat_template.json \
    preprocessor_config.json processor_config.json projector.pt \
    special_tokens_map.json tokenizer.json tokenizer_config.json
```

The `--delete` list removes old root-level checkpoint files if they exist. It does not delete `.gitattributes` or the root `README.md`; the root README is overwritten by `docs/HF_DATASET_CARD.md`.

## Download Verification

After upload, verify that a fresh download creates `checkpoints/fold0/` in the repository root:

```bash
rm -rf checkpoints/fold0
hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" \
  --local-dir .

python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8 \
  --limit 256
```
