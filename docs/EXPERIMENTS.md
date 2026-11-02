# Experimental setup

The [README](../README.md#experiments-and-analysis) contains the complete result tables and their interpretation. This page specifies the training configurations and score-estimation protocol.

## Training configurations

The comparator is LLaVA-1.5-7B with a frozen CLIP ViT-L/14-336px encoder, a trainable multimodal projector, and 4-bit QLoRA. LoRA rank is 8 with scaling factor 4. AdamW uses separate projector/LoRA learning rates and cosine decay. The label-distribution and loss experiments run for five epochs with effective batch size 64.

<div align="center">
<table width="100%" align="center">
<thead>
<tr>
<th width="10000" align="center">Experiment</th>
<th width="10000" align="center">Pair construction</th>
<th width="10000" align="center">Training / inference setting</th>
</tr>
</thead>
<tbody>
<tr><td align="center">Balanced three-class CE</td><td align="center">30,000 pairs per class</td><td align="center">Three-class CE; epoch-5 baseline</td></tr>
<tr><td align="center">Imbalanced three-class CE</td><td align="center">36,000 / 36,000 / 18,000<br>First / Second / Similar</td><td align="center">CE; fewer Similar pairs</td></tr>
<tr><td align="center">Balanced two-class CE</td><td align="center">45,000 First + 45,000 Second</td><td align="center">Exclude the similarity-margin pairs;<br>BT threshold 0</td></tr>
<tr><td align="center">Weighted CE</td><td align="center">30,000 pairs per class</td><td align="center">Class weights 1.0 / 1.0 / 1.2<br>First / Second / Similar</td></tr>
<tr><td align="center">CE + KL consistency</td><td align="center">45,000 unordered pairs, both orders</td><td align="center">32 forward + 32 reversed per effective batch;<br>KL coefficient 4; filter inconsistent predictions for BT</td></tr>
<tr><td align="center">Checkpoint ensemble</td><td align="center">Reuse the balanced CE run</td><td align="center">Average logits from five checkpoints<br>between epochs 4 and 5 before softmax</td></tr>
</tbody>
</table>
</div>

The class order in the descriptions above is `First`, `Second`, `Similar`. With rating difference Δy = y₁ − y₂, `First` means Δy > 0.2, `Second` means Δy < −0.2, and `Similar` means |Δy| ≤ 0.2. The binary variant excludes the similarity-margin pairs during training and removes the similarity interval during BT inference.

The KL setting preserves the total of 90,000 ordered instances per epoch, but starts from 45,000 unordered pairs. Its comparison with CE therefore includes a changed pair pool, reversed-pair supervision, and inference filtering. There is no separately reported swap-augmentation-only control.

## Bradley–Terry score recovery

Known-score references are selected only from training images. Scores are divided into bins of width 0.1, retaining at most 10 images per bin, for approximately 350 references covering the rating range. Each target is compared with every selected reference.

For a candidate target score, the three-class likelihood is computed from its score difference relative to each reference and the predicted order label. The experiment estimator minimizes the summed negative log likelihood on a grid from 1 to 5 with step 0.001. Similarity predictions favor small score differences, so their reliability and availability affect the recovered score as well as the intermediate classification metrics.

## Evaluation protocol

The project uses one 4,400 / 1,100 image train/test split. The original LMOL paper reports cross-validation results, so the main table is a reference comparison across different protocols, not a controlled ablation of first-token supervision.

The balanced CE baseline's score metrics were computed on 200 randomly selected test targets. Tables for the remaining variants reuse that baseline but do not separately specify their target counts. They should not be described as confirmed full-1,100-image score evaluations.

The early three-epoch classification measurements use 1,000 sampled test pairs, while the epoch-4/5 measurements use full test-pair evaluation. These measurements support the choice of a five-epoch baseline, but should not be interpreted as one fixed-set learning curve.

No separate validation split or validation-based early stopping was used. Per-epoch checkpoints were inspected after training; the ensemble reuses nearby checkpoints from one run rather than independent runs or folds. No uncertainty estimates across independent training runs are available.

## Public implementation

The public code supports balanced three-class CE training, checkpoint management, pairwise inference, and UOL-style score estimation. Other experiment variants are documented project work, not additional CLI modes. The [default settings comparison](TRAINING_ALIGNMENT.md#experiment-settings-and-public-defaults) records the one-epoch/batch-16 example and its reference-score-support estimator, which differ from the five-epoch/batch-64 experiments and dense score grid.

Experimental values and analysis follow the project's methodology, results, and conclusion sections (§2.2–2.5, §3.1–3.4, §4). See [Usage](USAGE.md) for the runnable pipeline.
