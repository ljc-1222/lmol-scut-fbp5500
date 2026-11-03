# LMOL on SCUT-FBP5500

Unofficial PyTorch implementation of [*Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*](https://doi.org/10.1109/ICASSP49660.2025.10889943) (ICASSP 2025), covering data preparation, LLaVA fine-tuning, pairwise inference, and score evaluation on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).

A fine-tuned LLaVA-1.5-7B compares two faces as `First.`, `Second.`, or `Similar.`. Comparisons against reference faces with known ratings are then combined through Bradley–Terry estimation to recover a score in the 1–5 range.

## Method

- **Instruction pairs:** sample 90,000 pairs per fold, balanced across the three order labels. A rating difference within θ = 0.2 is labeled `Similar.`.
- **Comparator:** encode both images separately with a shared CLIP ViT-L/14-336px encoder and multimodal projector. Freeze the vision tower and train the projector together with 4-bit QLoRA adapters on the language model.
- **Score recovery:** select references from the training split in 0.1-wide score bins, compare each test face against those references, and maximize the UOL-style Bradley–Terry likelihood.

[Method and reproduction settings](docs/TRAINING_ALIGNMENT.md) documents the paper settings and the code's training objective, fold construction, and score estimator.

## Quickstart

### Environment

Use Python 3.11 and a CUDA-capable GPU for training and model inference. Run subsequent commands from the repository root.

```bash
git clone https://github.com/ljc-1222/lmol-scut-fbp5500.git
cd lmol-scut-fbp5500
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

### Data

Review the [dataset's non-commercial research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), then download and validate the images and mean ratings:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
python scripts/build_pairs.py --fold-index 0
```

The pair builder writes the image splits, training pairs, reference manifest, and test/reference pairs under `artifacts/pairs/full/fold_0/`.

### Training and evaluation

Train one fold with the [default configuration](configs/lmol_scut_fbp5500.yaml), then evaluate its saved checkpoint:

```bash
python scripts/train.py --fold-index 0 --no-eval --no-wandb
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

The default schedule is one epoch with an effective batch size of 16. `--no-eval` leaves the held-out pairs for evaluation after training; `--no-wandb` disables external logging.

The [usage guide](docs/USAGE.md) covers manual data setup, all five folds, checkpoint files, separate inference/scoring commands, and tests.

## Evaluation

`scripts/evaluate_fold.py` writes pairwise predictions, a confusion matrix, progress, and score metrics to `outputs/evaluation/fold_0/`. The score metrics are Pearson correlation (PC), mean absolute error (MAE), and root mean squared error (RMSE).

For reference, Table I of the LMOL paper reports the following SCUT-FBP5500 results:

<div align="center">
<table width="100%" align="center">
<thead>
<tr><th width="10000" align="center">Method</th><th width="10000" align="center">PC (%) ↑</th><th width="10000" align="center">MAE ↓</th><th width="10000" align="center">RMSE ↓</th></tr>
</thead>
<tbody>
<tr><td align="center">UOL</td><td align="center">92.40</td><td align="center">0.1975</td><td align="center">0.2633</td></tr>
<tr><td align="center">LMOL</td><td align="center"><b>95.65</b></td><td align="center"><b>0.1552</b></td><td align="center"><b>0.2032</b></td></tr>
</tbody>
</table>
</div>

The evaluator stores PC as a coefficient in `metrics.json`; multiply it by 100 for the percentage convention above. MAE and RMSE use the original 1–5 rating scale.

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
