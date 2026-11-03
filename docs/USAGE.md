# Usage

Run commands from the repository root after installing [the runtime dependencies](../requirements.txt). Configuration defaults live in [`configs/lmol_scut_fbp5500.yaml`](../configs/lmol_scut_fbp5500.yaml); [method notes](TRAINING_ALIGNMENT.md) explain their relation to the paper.

## Data and pair files

The automatic downloader prepares `data/raw/SCUT-FBP5500/Images/` and `data/raw/SCUT-FBP5500/labels.txt`. For manual setup, obtain the data through the [official release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) and place the files at those paths. The annotation loader accepts whitespace-delimited or CSV rows with an image name and mean rating; a header row is optional.

```bash
python scripts/prepare_scut_fbp5500.py --print-manual-steps
python scripts/prepare_scut_fbp5500.py
```

Build all five folds, or add `--fold-index 0` to build only the quickstart fold:

```bash
python scripts/build_pairs.py
```

Each `artifacts/pairs/full/fold_<index>/` directory contains:

<div align="center">
<table width="100%" align="center">
<thead>
<tr><th width="10000" align="center">File</th><th width="10000" align="center">Contents</th></tr>
</thead>
<tbody>
<tr><td align="center"><code>train_images.csv</code></td><td align="center">Training image names, ratings, and paths</td></tr>
<tr><td align="center"><code>test_images.csv</code></td><td align="center">Held-out image names, ratings, and paths</td></tr>
<tr><td align="center"><code>train_pairs.csv</code></td><td align="center">Balanced instruction pairs from training images</td></tr>
<tr><td align="center"><code>references.csv</code></td><td align="center">Training-set references and their score bins</td></tr>
<tr><td align="center"><code>test_reference_pairs.csv</code></td><td align="center">Every held-out target paired with every selected reference</td></tr>
</tbody>
</table>
</div>

Changing the split seed, label margin, pair sampling, or reference selection requires rebuilding these files before training or evaluation.

## Training and checkpoints

Validate the configuration, training CSV, and image paths without loading the model:

```bash
python scripts/train.py --fold-index 0 --dry-run --no-eval
```

Training writes periodic `step_<N>/` checkpoints and a final `last/` checkpoint under `outputs/lmol_scut_fbp5500/fold_<index>/`. A checkpoint contains the LoRA adapter files, the full multimodal projector in `projector.pt`, and saved processor files. Evaluation loads both the adapter and projector on top of the configured base model.

The quickstart uses `--no-eval` to reserve test/reference pairs for evaluation after training. Without that flag, `scripts/train.py` uses the fold's test/reference pairs for periodic monitoring by default. To monitor a separate training-derived validation set, pass `--eval-pairs path/to/validation_pairs.csv` instead. Configure the interval and sample cap with `evaluation.eval_steps` and `evaluation.eval_max_pairs`.

Weights & Biases is enabled in the YAML. Use `--no-wandb` to disable it or `--wandb-mode offline` for local logs. `--stop-after-steps N` stops a diagnostic run while keeping the configured cosine schedule; `--max-steps N` changes the training horizon and scheduler length.

## Cross-validation

After building all five folds, train a separate model and evaluate its final checkpoint for each held-out fold:

```bash
for fold in 0 1 2 3 4; do
  python scripts/train.py --fold-index "$fold" --no-eval --no-wandb
  python scripts/evaluate_fold.py --fold-index "$fold"
done
```

The evaluator defaults to each fold's `last/` checkpoint and writes separate metrics under `outputs/evaluation/fold_<index>/`. Summarize those five metric files when reporting cross-validation; the evaluator itself operates on one fold at a time.

## Evaluation outputs and separate stages

A full `scripts/evaluate_fold.py` run produces:

<div align="center">
<table width="100%" align="center">
<thead>
<tr><th width="10000" align="center">Output</th><th width="10000" align="center">Contents</th></tr>
</thead>
<tbody>
<tr><td align="center"><code>pairwise_predictions.csv</code></td><td align="center">Target/reference IDs and ratings, true/predicted labels, and class log probabilities</td></tr>
<tr><td align="center"><code>confusion_matrix.json</code></td><td align="center">Gold-row/predicted-column counts and overall pairwise accuracy</td></tr>
<tr><td align="center"><code>metrics.json</code></td><td align="center">Target count, PC, MAE, RMSE, estimator name, and BT parameters</td></tr>
<tr><td align="center"><code>progress.json</code></td><td align="center">Current phase, completed comparisons, throughput, and elapsed time</td></tr>
</tbody>
</table>
</div>

PC is stored as a coefficient, while MAE and RMSE use the dataset's 1–5 rating scale. Pairwise accuracy evaluates order labels; PC, MAE, and RMSE evaluate the recovered target ratings.

Pair prediction and score estimation can also run separately. Score estimation reuses the saved predictions without loading the 7B model:

```bash
python scripts/pairwise_infer.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
python scripts/estimate_scores.py \
  --fold-index 0 \
  --predictions outputs/evaluation/fold_0/pairwise_predictions.csv
```

For a short checkpoint smoke test, evaluate a limited number of pairs and write to separate paths:

```bash
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --limit 256 \
  --predictions outputs/smoke/fold_0/pairwise_predictions.csv \
  --metrics outputs/smoke/fold_0/metrics.json \
  --confusion outputs/smoke/fold_0/confusion_matrix.json \
  --progress outputs/smoke/fold_0/progress.json
```

This checks loading and inference on a subset of pairs. Use the complete target/reference CSV for a fold evaluation.

## Tests and code entry points

Unit tests use synthetic inputs and CPU PyTorch, with no dataset or model-weight download:

```bash
python -m unittest discover -s tests -v
```

The [CI workflow](../.github/workflows/tests.yml) runs these tests, syntax checks, and CLI help checks with isolated CPU test dependencies.

<div align="center">
<table width="100%" align="center">
<thead>
<tr><th width="10000" align="center">Component</th><th width="10000" align="center">Entry points</th></tr>
</thead>
<tbody>
<tr><td align="center">Data loading, folds, pair sampling, and references</td><td align="center"><a href="../data/"><code>data/</code></a>, <a href="../scripts/build_pairs.py"><code>scripts/build_pairs.py</code></a></td></tr>
<tr><td align="center">Model loading, QLoRA, and checkpoints</td><td align="center"><a href="../model/lmol.py"><code>model/lmol.py</code></a></td></tr>
<tr><td align="center">Instruction formatting and training</td><td align="center"><a href="../data/instruction_dataset.py"><code>data/instruction_dataset.py</code></a>, <a href="../training/train.py"><code>training/train.py</code></a></td></tr>
<tr><td align="center">Pairwise inference and score metrics</td><td align="center"><a href="../evaluation/"><code>evaluation/</code></a>, <a href="../scripts/evaluate_fold.py"><code>scripts/evaluate_fold.py</code></a></td></tr>
</tbody>
</table>
</div>

Input data, generated pairs, checkpoints, predictions, and logs are excluded from Git.
