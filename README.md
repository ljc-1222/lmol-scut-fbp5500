# LMOL on SCUT-FBP5500

This repository implements **Large Multimodal Model based Order Learning (LMOL)** for facial beauty prediction on **SCUT-FBP5500**. It is a focused research-code release for reproducing the included fold0 experiment, not a general-purpose facial attractiveness package.

## Paper And Method

This implementation is based on:

> Zhenyou Liu, Xuefeng Liang, and Jian Lin, "Large Multimodal Model is a Better Comparator on Facial Beauty Prediction," in *ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1-5, 2025. DOI: [10.1109/ICASSP49660.2025.10889943](https://doi.org/10.1109/ICASSP49660.2025.10889943)

The paper frames facial beauty prediction as an order-learning problem: instead of directly regressing a score from one face image, a large multimodal model compares two faces and predicts whether the first face is more attractive, the second face is more attractive, or the two are similar. Absolute beauty scores are then estimated from many pairwise comparisons.

This repository follows the paper's core setup:

- Base model: LLaVA-1.5-7B with CLIP ViT-L/14-336px and Vicuna-1.5 backbone.
- Input format: two face images are encoded separately with shared visual encoder/projector weights; images are not concatenated at pixel level.
- Pair labels: `First.`, `Second.`, and `Similar.` from SCUT score differences with `theta=0.2`.
- Training objective: the default `choice_token` objective trains the LLaVA LM head as a 3-way next-token classifier over the first answer word tokens.
- Trainable modules: the CLIP visual encoder is frozen; the full multimodal projector and QLoRA adapters on the language model are trained.
- LoRA setting: `r=8`, `lora_alpha=32`, `lora_dropout=0.05`, targeting language-model `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Score estimation: pairwise orders are converted to SCUT scores using Bradley-Terry-style reference comparison and reported with PC, MAE, and RMSE.

## Implementation Notes: Objective, Labels, And Inference

The paper describes the answer target as standard conditional language modeling over the natural-language answer string. This repository keeps the same three semantic answers, but the released training/evaluation path uses a narrower token-choice formulation:

- Class labels are ordered as `First.` = 0, `Second.` = 1, and `Similar.` = 2. They are built from SCUT score differences: `score1 - score2 > theta` gives `First.`, `< -theta` gives `Second.`, and `abs(score1 - score2) <= theta` gives `Similar.`.
- With the LLaVA tokenizer used by the checkpoint, `First.`, `Second.`, and `Similar.` are each two tokens: the answer word token plus a shared period token. The default objective and pairwise evaluator score only the first answer word token (`First`, `Second`, or `Similar`); the period token is not part of the loss or prediction score.
- `training.objective: choice_token` runs one forward pass on the prompt/answer batch, discards the masked LM labels, takes logits at the prompt boundary, restricts them to the three answer word token IDs, and applies 3-way cross entropy.
- Pairwise inference is prompt-only likelihood scoring, not `model.generate`. It takes the final prompt-position logits, normalizes only across the three answer word tokens, and selects the largest normalized log-probability.
- The `conditional_lm` branch uses the same current collator mask, so it also supervises only the first answer word token unless the collator is changed to keep all answer tokens. Treat it as a diagnostic branch rather than a full reproduction of the paper's multi-token conditional-LM objective.

See [`docs/TRAINING_ALIGNMENT.md`](docs/TRAINING_ALIGNMENT.md) for the paper-to-code alignment notes and the implementation evidence.

## Repository Scope

The repository contains:

- SCUT-FBP5500 dataset validation and download helpers.
- Deterministic 5-fold split and offline pair construction.
- LLaVA/QLoRA LMOL model setup.
- Fold training and evaluation scripts.
- Fold0 evaluation results and small release copies of the final JSON metrics.
- Hugging Face Dataset-repo upload/download instructions for the fold0 checkpoint package.

The repository does not redistribute SCUT-FBP5500 images or labels as a public dataset package, and it does not include the trained checkpoint in GitHub. Download the dataset under the dataset authors' terms and download the fold0 checkpoint from Hugging Face.

## Documentation

Additional public documentation is indexed in [`docs/README.md`](docs/README.md). The most relevant notes are [`docs/TRAINING_ALIGNMENT.md`](docs/TRAINING_ALIGNMENT.md) for paper-to-code alignment and [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) for reproducibility assumptions.

## Fold 0 Release Result

The repository currently reports **fold 0 only**, not full 5-fold cross-validation. The authoritative score metrics were extracted from `outputs/evaluation/fold_0/metrics.json`; small GitHub-visible release copies are kept in [`docs/results/fold_0/`](docs/results/fold_0/).

<div align="center">

<table>
  <thead>
    <tr>
      <th>Split</th>
      <th align="right">PC ↑</th>
      <th align="right">MAE ↓</th>
      <th align="right">RMSE ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fold 0</td>
      <td align="right">0.9568</td>
      <td align="right">0.1643</td>
      <td align="right">0.2130</td>
    </tr>
  </tbody>
</table>

</div>

Pairwise classification metrics were computed from [`confusion_matrix.json`](docs/results/fold_0/confusion_matrix.json) and [`progress.json`](docs/results/fold_0/progress.json).

<div align="center">

<table>
  <thead>
    <tr>
      <th>Split</th>
      <th align="right">Accuracy ↑</th>
      <th align="right">Macro Precision ↑</th>
      <th align="right">Macro Recall ↑</th>
      <th align="right">Macro F1 ↑</th>
      <th align="right">Weighted F1 ↑</th>
      <th align="right">Pairwise NLL ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fold 0</td>
      <td align="right">0.8649</td>
      <td align="right">0.8145</td>
      <td align="right">0.8354</td>
      <td align="right">0.8195</td>
      <td align="right">0.8723</td>
      <td align="right">0.3384</td>
    </tr>
  </tbody>
</table>

</div>

Per-answer pairwise metrics:

<div align="center">

<table>
  <thead>
    <tr>
      <th>Gold Label</th>
      <th align="right">Support</th>
      <th align="right">Predicted</th>
      <th align="right">Precision ↑</th>
      <th align="right">Recall ↑</th>
      <th align="right">F1 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>First.</code></td>
      <td align="right">120,836</td>
      <td align="right">110,502</td>
      <td align="right">0.9335</td>
      <td align="right">0.8537</td>
      <td align="right">0.8918</td>
    </tr>
    <tr>
      <td><code>Second.</code></td>
      <td align="right">194,770</td>
      <td align="right">185,020</td>
      <td align="right">0.9557</td>
      <td align="right">0.9079</td>
      <td align="right">0.9312</td>
    </tr>
    <tr>
      <td><code>Similar.</code></td>
      <td align="right">58,394</td>
      <td align="right">78,478</td>
      <td align="right">0.5541</td>
      <td align="right">0.7446</td>
      <td align="right">0.6354</td>
    </tr>
  </tbody>
</table>

</div>

Additional fold0 evidence:

- Split: deterministic 5-fold SCUT-FBP5500 split with `split_seed: 42`.
- Images: 4,400 train images and 1,100 test images.
- Pair construction: 90,000 balanced fold0 training pairs and 374,000 fold0 test-reference pairs using 340 train-reference images.
- Evaluated checkpoint: `outputs/lmol_scut_fbp5500/fold_0/last`.
- Release result files: [`metrics.json`](docs/results/fold_0/metrics.json), [`confusion_matrix.json`](docs/results/fold_0/confusion_matrix.json), and [`progress.json`](docs/results/fold_0/progress.json).

## Fold 0 Checkpoint

The trained fold0 checkpoint package is intended to be hosted as a Hugging Face **Dataset** repository:

<https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500>

Download it into the local path used by the examples:

```bash
pip install -U huggingface_hub
hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" \
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

The checkpoint directory must contain:

```text
adapter_config.json
adapter_model.safetensors
projector.pt
tokenizer.json
tokenizer_config.json
processor_config.json
preprocessor_config.json
special_tokens_map.json
chat_template.json
README.md
```

The local checkpoint package prepared for upload is `outputs/lmol_scut_fbp5500/fold_0/last/`. Maintainer upload commands are in [`docs/HUGGING_FACE_UPLOAD.md`](docs/HUGGING_FACE_UPLOAD.md).

## Quick Reproduction

From a fresh checkout, reproduce the reported fold0 metrics with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime

python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py --config configs/lmol_scut_fbp5500.yaml
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0

hf download ljc-1222/LMOL-fbp5500 \
  --repo-type dataset \
  --include "checkpoints/fold0/*" \
  --local-dir .

python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```

This writes `outputs/evaluation/fold_0/pairwise_predictions.csv`, `metrics.json`, `confusion_matrix.json`, and `progress.json`. Compare the resulting `outputs/evaluation/fold_0/metrics.json` to [`docs/results/fold_0/metrics.json`](docs/results/fold_0/metrics.json).

Full fold0 evaluation is long because it scores 374,000 target-reference pairs. Use `--limit 256` only for a checkpoint smoke test; it does not reproduce the reported metrics.

## Environment

Recommended runtime:

- Python 3.11.
- PyTorch 2.4.1, Transformers 4.45.2, Accelerate 0.34.2, PEFT 0.13.2, bitsandbytes 0.43.3.
- CUDA-capable GPU for full training and evaluation. The included fold0 run metadata reports one NVIDIA GeForce RTX 4090.

Canonical setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

For a non-strict import and hardware summary:

```bash
python scripts/check_environment.py
```

## Dataset Preparation

SCUT-FBP5500 can be installed into the expected local layout with:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
```

The `--accept-terms` flag confirms that your use is compatible with the dataset authors' non-commercial research terms. The script downloads `SCUT-FBP5500_v2.1.zip` from the official Google Drive link, extracts only the files needed by this repository, validates the result, and deletes the downloaded zip after successful installation. Add `--keep-archive` if you want to keep the zip.

Expected layout:

```text
data/raw/SCUT-FBP5500/
├── Images/
│   ├── AF1.jpg
│   └── ...
└── labels.txt
```

`labels.txt` should contain one image and score per row:

```text
AF1.jpg 2.883333
AM1.jpg 3.950000
```

Validate the installed dataset:

```bash
python scripts/prepare_scut_fbp5500.py --config configs/lmol_scut_fbp5500.yaml
```

If the dataset is missing, print setup steps:

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
```

## Pair Construction

Build all configured folds:

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml
```

Build fold0 only:

```bash
python scripts/build_pairs.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

Fold outputs use this layout:

```text
artifacts/pairs/full/fold_0/
├── train_images.csv
├── test_images.csv
├── train_pairs.csv
└── test_reference_pairs.csv
```

For the full config, each `train_pairs.csv` contains 90,000 pairs whenever enough candidates exist: 30,000 `First.`, 30,000 `Second.`, and 30,000 `Similar.`.

## Training

Dry-run the fold0 pair loading path without loading model weights:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0 --dry-run
```

Train fold0:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0
```

Outputs are written under fold-scoped directories such as `outputs/lmol_scut_fbp5500/fold_0/`. Training logs include raw `train_loss`, scaled `train_scaled_loss`, learning rates, `step_time_sec`, `samples_per_sec`, and periodic pairwise likelihood `eval_loss` / `eval_acc` from `evaluation.eval_max_pairs` examples every `evaluation.eval_steps` optimizer steps.

The default setting uses microbatch 1 with gradient accumulation 16, giving effective batch size 16. Training uses the configured `seed: 42`; exact bit-level reproducibility can still depend on CUDA kernels, driver versions, and low-level library behavior.

Weights & Biases logging is enabled by default through the `wandb` section in the config. Log in once before training:

```bash
wandb login
```

Useful overrides:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 1 --wandb-name fold1-r8-bs16
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 1 --wandb-mode offline
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 1 --no-wandb
```

For diagnostics, prefer `--stop-after-steps N` when you want to inspect early behavior without changing the configured cosine scheduler horizon:

```bash
python scripts/train.py --config configs/lmol_scut_fbp5500.yaml --fold-index 0 --stop-after-steps 100
```

Use `--max-steps N` only when you intentionally want a shorter training run with the scheduler compressed to that length.

## Evaluation

Run full fold evaluation with progress reporting and score estimation:

```bash
python scripts/evaluate_fold.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0 \
  --batch-size 8
```

For pairwise predictions only:

```bash
python scripts/pairwise_infer.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --checkpoint checkpoints/fold0
```

Estimate SCUT scores from an existing predictions CSV:

```bash
python scripts/estimate_scores.py \
  --config configs/lmol_scut_fbp5500.yaml \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

Metrics:

- `pc`: Pearson Correlation Coefficient.
- `mae`: Mean Absolute Error.
- `rmse`: Root Mean Squared Error.

## Tests

CPU-safe unit tests:

```bash
python -m unittest discover -s tests
```

The tests cover theta label construction, exact instruction and answer formatting, balanced offline sampling, SCUT annotation validation, PC/MAE/RMSE, Bradley-Terry ranking, and import/config sanity.

## Release Notes

- Do not commit `checkpoints/`, `outputs/`, `artifacts/`, `wandb/`, `data/raw/`, or model weight files. These are ignored by `.gitignore`.
- Keep the small fold0 release result files under `docs/results/fold_0/` so the reported README table has stable GitHub-visible evidence.
- The SCUT-FBP5500 dataset has external usage terms; this repository documents download/setup but should not redistribute the raw dataset.
- No license file is currently present. Choose and add a license before publishing if you want others to know their reuse rights.

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
