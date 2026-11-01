# Project experiments

Results from *Order Learning with Large Multimodal Models for Facial Beauty Prediction*, National Tsing Hua University, Department of Electrical Engineering, 2025. Values below are transcribed from the project's report, Tables 3-2–3-4; they are historical measurements, not a new evaluation of the public release.

## Experimental setup

| Item | Reported setting |
| --- | --- |
| Dataset | SCUT-FBP5500; 5,500 images with mean ratings from 60 raters, on a 1–5 scale |
| Image split | One random split: 4,400 training images and 1,100 test images |
| Comparator | LLaVA-1.5-7B; frozen CLIP ViT-L/14-336px, trainable multimodal projector, 4-bit QLoRA |
| LoRA | Rank 8; scaling factor 4 |
| Training | Five epochs, effective batch size 64 via gradient accumulation, one GPU |
| Optimizer | AdamW; separate projector/LoRA learning rates with cosine decay |
| Class boundary | `Similar` when the absolute rating difference is at most 0.2 |
| References | Training images only; at most 10 per 0.1-wide rating bin, approximately 350 images |
| Historical score recovery | Three-class Bradley–Terry-style likelihood; search ratings from 1 to 5 in steps of 0.001 |
| Checkpoint selection | No separate validation set or validation-based early stopping; checkpoints inspected after training |

### Evaluation protocol

The report's general protocol reserves 1,100 test images (§2.2, §3), but **§3.1 explicitly reports the balanced baseline's PC/MAE/RMSE on 200 randomly selected test images** and their reference comparisons. Tables 3-2–3-4 reuse those baseline values without restating how many targets were scored for the later configurations. Their exact target counts and whether identical subsets were used are therefore unspecified.

Pairwise classification accuracy and continuous-score metrics also have different evaluation units: the former measures image-pair decisions; the latter measures recovered target-image scores. The early epoch-1–3 accuracy measurements used 1,000 sampled pairs, whereas later accuracy measurements used the full test split, so those early figures are not presented as one uniform learning curve here.

The original LMOL paper reports PC **95.65%**, MAE **0.1552**, and RMSE **0.2032** on SCUT-FBP5500 (Table I), using a cross-validation protocol (§III-B). The project results below come from a single split and have the sampling qualification above. They do not establish a controlled performance comparison with the paper or a cross-validation average. No independent-run uncertainty estimates are reported.

## Label distribution and training objective

All rows below use the epoch-5 checkpoint. “Overall” and “Similar” are the pairwise accuracies reported in Tables 3-2 and 3-3; PC, MAE, and RMSE describe BT-recovered scores.

| Configuration | Overall accuracy (%) ↑ | Similar accuracy (%) ↑ | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced three-class, CE | 85.84 | 73.36 | 95.26 | 0.1730 | 0.2150 |
| Imbalanced three-class (4:4:2), CE | 86.30 | 50.51 | 94.76 | 0.1760 | 0.2290 |
| Two-class, CE | 92.07 | — | 93.35 | 0.2345 | 0.2916 |
| Balanced three-class, weighted CE | 86.45 | 49.89 | 94.52 | 0.1771 | 0.2304 |
| Balanced three-class, CE + KL consistency | 86.40 | 74.28 | 95.21 | 0.1730 | 0.2160 |

The configurations differ as follows:

- **Balanced three-class:** 30,000 pairs each for `First`, `Second`, and `Similar`.
- **Imbalanced three-class:** 36,000 / 36,000 / 18,000 pairs for `First` / `Second` / `Similar`.
- **Two-class:** remove pairs within the 0.2 similarity margin; sample 45,000 `First` and 45,000 `Second` pairs. Use a zero similarity threshold for historical BT inference.
- **Weighted CE:** use class weights 1.0 / 1.0 / 1.2 for `First` / `Second` / `Similar`.
- **CE + KL:** start from 45,000 unordered pairs and present both orders, giving 90,000 ordered instances per epoch. Each effective batch contains 32 forward and 32 reversed pairs. Add symmetric KL divergence after exchanging the `First`/`Second` probabilities, with a KL coefficient of 4.

In these reported runs, increasing overall pairwise accuracy did not guarantee better recovered scores: removing `Similar` increased accuracy to 92.07% but reduced PC to 93.35%. Weighted CE also reduced `Similar` accuracy and score quality. KL achieved **98.36% forward/reverse consistency**, but its score metrics did not improve over CE. The report describes dropping inconsistent comparisons before KL-model score aggregation (§3.3), so that row reflects both the training objective and this inference treatment.

## Checkpoint ensemble

Five checkpoints from a single balanced three-class CE run, collected between epochs 4 and 5, were evaluated on each pair. Their three candidate logits were averaged **before softmax**.

| Model | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| Single checkpoint, epoch 5 | 95.26 | 0.1730 | 0.2150 |
| Five-checkpoint logit ensemble | **95.32** | **0.1720** | **0.2141** |

This reuses one training run and requires five comparator evaluations per pair. It is a checkpoint ensemble, not a five-fold ensemble or five independently trained models.

## Relation to the public implementation

The repository includes balanced three-class pair generation, first-token CE training, checkpoint saving/loading, pairwise inference, and UOL-style score estimation. The imbalanced/two-class variants, weighted CE, KL consistency, and checkpoint ensemble are documented experiments rather than available CLI modes.

The [example YAML](../configs/lmol_scut_fbp5500.yaml) uses **one epoch and effective batch size 16**, compared with five epochs and batch size 64 in the report. It also generates deterministic folds rather than loading the historical split. The public score estimator searches **unique reference scores**, while the report describes a **0.001-spaced grid across [1, 5]**. These differences prevent treating the Quickstart as an exact reproduction command for the historical table.

The first-token CE objective is described in report §2.3.3 and implemented in [training](../training/train.py) and [inference](../evaluation/inference.py). The project did not report a matched full-answer-supervision ablation, so no isolated accuracy or speed gain is assigned to that modification.

See [Usage](USAGE.md) for the runnable pipeline and [implementation notes](TRAINING_ALIGNMENT.md) for the paper-to-code details.
