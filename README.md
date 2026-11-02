# LMOL on SCUT-FBP5500

This project implements [Large Multimodal Model based Order Learning (LMOL)](https://doi.org/10.1109/ICASSP49660.2025.10889943) for facial beauty prediction on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). A fine-tuned LLaVA-1.5-7B compares two faces as `First`, `Second`, or `Similar`; Bradley–Terry estimation combines comparisons against known-score references into a rating from 1 to 5.

We reconstruct the training and score-estimation pipeline, replace full-answer supervision with **three-class first-token cross-entropy**, and investigate how pair construction, training objectives, and checkpoint ensembling affect the recovered scores.

## Results

SCUT-FBP5500 score prediction. PC is Pearson correlation; MAE and RMSE are measured on the 1–5 rating scale. Bold values identify the best configuration within our experiments.

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Model</th>
<th width="10000" align="center">PC (%) ↑</th>
<th width="10000" align="center">MAE ↓</th>
<th width="10000" align="center">RMSE ↓</th>
</tr>
</thead>
<tbody>
<tr><td align="center"><a href="https://doi.org/10.1109/ICASSP49660.2025.10889943">Original LMOL</a></td><td align="center">95.65</td><td align="center">0.1552</td><td align="center">0.2032</td></tr>
<tr><td align="center">Our comparator<br>Balanced three-class CE</td><td align="center">95.26</td><td align="center">0.1730</td><td align="center">0.2150</td></tr>
<tr><td align="center">Our comparator<br>Five-checkpoint ensemble</td><td align="center"><b>95.32</b></td><td align="center"><b>0.1720</b></td><td align="center"><b>0.2141</b></td></tr>
</tbody>
</table>
</div>

The original LMOL result uses the paper's cross-validation protocol; our experiments use one train/test split. [Evaluation details](docs/EXPERIMENTS.md#evaluation-protocol).

## Implementation

- **Multimodal fine-tuning:** two separately encoded images, a frozen CLIP ViT-L/14 vision tower, a trainable multimodal projector, and 4-bit QLoRA on the language model. Projector and LoRA parameters have separate learning rates. [Model](model/lmol.py) · [Training](training/train.py).
- **First-token classification:** the original paper trains on the complete answer sequence. We select only the `First` / `Second` / `Similar` vocabulary logits at the first answer position and apply three-class CE. Inference uses the same candidate tokens directly, without generating and parsing an answer string. [Objective and implementation](docs/TRAINING_ALIGNMENT.md#first-token-objective).
- **Reference-based scoring:** offline image-pair construction, score-binned references selected only from training images, and Bradley–Terry likelihood aggregation for continuous-score prediction. [Data](data/) · [Evaluation](evaluation/).

## Experiments and analysis

The experiments below use LLaVA-1.5-7B and a similarity margin of θ = 0.2. The label-distribution and loss comparisons use five epochs and an effective batch size of 64. Each standard training set contains 90,000 pairs; the KL setting uses 45,000 unordered pairs in both directions.

### Training duration

Balanced three-class CE was first trained for three epochs, then extended to five. The later checkpoints approach a plateau, supporting the use of epoch 5 as the baseline for the following experiments.

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Checkpoint</th>
<th width="10000" align="center">Pairwise accuracy (%) ↑</th>
<th width="10000" align="center">Evaluation pairs</th>
</tr>
</thead>
<tbody>
<tr><td align="center">Epoch 1 / 3</td><td align="center">71.90</td><td align="center">1,000 sampled test pairs</td></tr>
<tr><td align="center">Epoch 2 / 3</td><td align="center">80.00</td><td align="center">1,000 sampled test pairs</td></tr>
<tr><td align="center">Epoch 3 / 3</td><td align="center">80.40</td><td align="center">1,000 sampled test pairs</td></tr>
<tr><td align="center">Epoch 4 / 5</td><td align="center">85.69</td><td align="center">Full test-pair evaluation</td></tr>
<tr><td align="center">Epoch 5 / 5</td><td align="center">85.84</td><td align="center">Full test-pair evaluation</td></tr>
</tbody>
</table>
</div>

The first three measurements use sampled pairs and the later two use full test-pair evaluation. They show the training progression under those evaluation settings, not a learning curve measured on one identical pair set.

### Label distribution

All three configurations use standard CE and 90,000 training pairs:

- **Balanced three-class:** 30,000 `First`, 30,000 `Second`, and 30,000 `Similar` pairs.
- **Imbalanced three-class (4:4:2):** 36,000 `First`, 36,000 `Second`, and 18,000 `Similar` pairs.
- **Balanced two-class:** 45,000 `First` and 45,000 `Second` pairs, excluding comparisons within the similarity margin. BT inference uses θ = 0 for this variant.

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Training labels</th>
<th width="10000" align="center">Overall<br>accuracy (%) ↑</th>
<th width="10000" align="center">Similar<br>accuracy (%) ↑</th>
<th width="10000" align="center">PC (%) ↑</th>
<th width="10000" align="center">MAE ↓</th>
<th width="10000" align="center">RMSE ↓</th>
</tr>
</thead>
<tbody>
<tr><td align="center">Three-class<br>Balanced</td><td align="center">85.84</td><td align="center">73.36</td><td align="center">95.26</td><td align="center">0.1730</td><td align="center">0.2150</td></tr>
<tr><td align="center">Three-class<br>Imbalanced (4:4:2)</td><td align="center">86.30</td><td align="center">50.51</td><td align="center">94.76</td><td align="center">0.1760</td><td align="center">0.2290</td></tr>
<tr><td align="center">Two-class<br>No Similar</td><td align="center">92.07</td><td align="center">—</td><td align="center">93.35</td><td align="center">0.2345</td><td align="center">0.2916</td></tr>
</tbody>
</table>
</div>

The imbalanced setting raises overall accuracy slightly while reducing `Similar` accuracy to 50.51%. Removing `Similar` raises pairwise accuracy further, yet gives the poorest score estimates. Strict-order decisions specify which face is preferred; `Similar` decisions also constrain how close their scores should be. Fewer reliable similarity comparisons leave BT with less local guidance when reconciling conflicting order predictions, explaining the deterioration in PC, MAE, and RMSE.

### Loss functions and order consistency

These experiments retain the balanced three-class task. **Weighted CE** assigns weights of 1.0 / 1.0 / 1.2 to `First` / `Second` / `Similar`. **CE + KL** adds symmetric forward/reverse consistency with coefficient 4, exchanging `First` and `Second` probabilities while preserving `Similar`. Each effective KL batch contains 32 forward pairs and their 32 reversed counterparts.

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Objective</th>
<th width="10000" align="center">Overall<br>accuracy (%) ↑</th>
<th width="10000" align="center">Similar<br>accuracy (%) ↑</th>
<th width="10000" align="center">PC (%) ↑</th>
<th width="10000" align="center">MAE ↓</th>
<th width="10000" align="center">RMSE ↓</th>
</tr>
</thead>
<tbody>
<tr><td align="center">CE</td><td align="center">85.84</td><td align="center">73.36</td><td align="center">95.26</td><td align="center">0.1730</td><td align="center">0.2150</td></tr>
<tr><td align="center">Weighted CE</td><td align="center">86.45</td><td align="center">49.89</td><td align="center">94.52</td><td align="center">0.1771</td><td align="center">0.2304</td></tr>
<tr><td align="center">CE + KL consistency</td><td align="center">86.40</td><td align="center">74.28</td><td align="center">95.21</td><td align="center">0.1730</td><td align="center">0.2160</td></tr>
</tbody>
</table>
</div>

**Weighted CE:** `Similar` accuracy falls to 49.89%, despite the higher class weight. The analysis attributes this decline to more `Similar` predictions around ambiguous boundaries without a corresponding gain in reliability. This leaves BT with fewer clean local constraints and offers an explanation for the poorer score metrics, even as overall accuracy rises.

**KL consistency:** forward/reverse agreement reaches **98.36%**, and `Similar` accuracy improves slightly to 74.28%. However, inconsistent predictions are discarded before BT aggregation, which also removes usable `Similar` comparisons. The small classification gain does not compensate for that loss of evidence, leaving score metrics close to the CE baseline. This result reflects the regularizer together with the filtering step, not an isolated change to the loss.

### Checkpoint ensemble

Five checkpoints between epochs 4 and 5 of the balanced three-class CE run are combined by averaging their three class logits **before softmax**. All three score metrics improve slightly over the single-checkpoint baseline, as shown in the main result table.

Averaging reduces variation in pairwise predictions and smooths the BT likelihood. The gain is modest because nearby checkpoints from the same run are highly correlated; they add limited prediction diversity. Evaluation requires five comparator passes per pair. The results favor preserving reliable `Similar` comparisons over optimizing overall pairwise accuracy or reversal agreement alone.

The public code contains the balanced three-class CE pipeline. The imbalanced/two-class, weighted-CE, KL, and ensemble settings above describe the broader project experiments. [Detailed configurations and evaluation protocol](docs/EXPERIMENTS.md).

## Getting started

Use Python 3.11 and a CUDA GPU. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py --strict-runtime
```

Review the [SCUT-FBP5500 research terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), then prepare the data:

```bash
python scripts/download_scut_fbp5500.py --accept-terms
python scripts/prepare_scut_fbp5500.py
python scripts/build_pairs.py --fold-index 0
```

Train a comparator and evaluate its saved checkpoint:

```bash
python scripts/train.py --fold-index 0 --no-wandb
python scripts/evaluate_fold.py \
  --fold-index 0 \
  --checkpoint outputs/lmol_scut_fbp5500/fold_0/last
```

The [usage guide](docs/USAGE.md) covers checkpoint download, data layout, configuration, tests, and code entry points. The [implementation notes](docs/TRAINING_ALIGNMENT.md#experiment-settings-and-public-defaults) distinguish the example configuration from the experimental settings above.

## Credits

Developed in the Department of Electrical Engineering at **National Tsing Hua University**, supervised by **Prof. Yeong-Luh Ueng**. **Jin-Cheng Lin** primarily handled model training; **Hsin-Chang Tsai** primarily handled inference and analysis.

Based on Z. Liu, X. Liang, and J. Lin, [*Large Multimodal Model is a Better Comparator on Facial Beauty Prediction*](https://doi.org/10.1109/ICASSP49660.2025.10889943), ICASSP 2025, and [Uncertainty-oriented Order Learning (UOL)](https://arxiv.org/abs/2409.00603). Dataset access is governed by the [SCUT-FBP5500 terms](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
