# Experiments

The released checkpoint evaluation and the training ablations are separate experiment sets. The first section describes the downloadable model; the remaining sections compare label distributions, training objectives, and checkpoint ensembling within the ablation study.

## Released checkpoint

The released fold-0 comparator was evaluated on **1,100 test images** against **340 training-set references**, yielding **374,000 target/reference comparisons**. The [final progress record](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500/blob/fbbc31831db7b3a5a3866c7a1e616930099350e1/results/fold0/progress.json) marks all comparisons complete.

| Model | Evaluation | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | --- | ---: | ---: | ---: |
| [Original LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943) | Cross-validation | 95.65 | 0.1552 | 0.2032 |
| This implementation | Fold 0 | 95.68 | 0.1643 | 0.2130 |

The project scores are rounded from [`metrics.json`](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500/blob/fbbc31831db7b3a5a3866c7a1e616930099350e1/results/fold0/metrics.json). The paper's Table I reports the LMOL row; its §III-B describes cross-validation, whereas this release reports one fold. These figures do not establish a matched-protocol comparison or isolate the effect of first-token supervision.

The [confusion matrix](https://huggingface.co/datasets/ljc-1222/LMOL-fbp5500/blob/fbbc31831db7b3a5a3866c7a1e616930099350e1/results/fold0/confusion_matrix.json) gives **86.49% overall pairwise accuracy**. Classwise precision, recall, and F1 below are calculated from its gold-row/predicted-column counts:

| Class | Support | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| First | 120,836 | 93.35 | 85.37 | 89.18 |
| Second | 194,770 | 95.57 | 90.79 | 93.12 |
| Similar | 58,394 | 55.41 | 74.46 | 63.54 |

`Similar` remains the least precise class: 78,478 comparisons were predicted as `Similar`, of which 43,483 were correct. This distinguishes the released model's classification behavior from its continuous-score metrics.

## Label distribution

This experiment varies the training labels while retaining **standard CE, five epochs, and 90,000 training pairs**:

- **Balanced three-class:** 30,000 pairs each for `First`, `Second`, and `Similar`.
- **Imbalanced three-class:** 36,000 / 36,000 / 18,000 pairs, respectively (4:4:2).
- **Two-class:** remove pairs within the 0.2 similarity margin; use 45,000 `First` and 45,000 `Second` pairs. Score recovery uses a zero similarity threshold for this variant.

| Training labels | Overall accuracy (%) ↑ | Similar accuracy (%) ↑ | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three classes, balanced | 85.84 | 73.36 | 95.26 | 0.1730 | 0.2150 |
| Three classes, imbalanced (4:4:2) | 86.30 | 50.51 | 94.76 | 0.1760 | 0.2290 |
| Two classes, no Similar | 92.07 | — | 93.35 | 0.2345 | 0.2916 |

Pairwise accuracy rises as the task favors strict `First`/`Second` decisions, while all score metrics deteriorate. `Similar` comparisons supply constraints on small score differences; reducing their number or reliability leaves less local evidence for Bradley–Terry estimation. This supports retaining the balanced three-class formulation rather than selecting a model by overall pairwise accuracy alone.

## Loss and consistency settings

These runs retain the **balanced three-class task** and compare the epoch-5 checkpoints:

| Objective | Overall accuracy (%) ↑ | Similar accuracy (%) ↑ | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE | 85.84 | 73.36 | 95.26 | 0.1730 | 0.2150 |
| Weighted CE | 86.45 | 49.89 | 94.52 | 0.1771 | 0.2304 |
| CE + KL consistency | 86.40 | 74.28 | 95.21 | 0.1730 | 0.2160 |

**Weighted CE** assigns a weight of 1.2 to `Similar` and 1.0 to the other classes. In this run, the increased weight did not produce more reliable similarity decisions: `Similar` accuracy fell to 49.89%, with poorer recovered scores.

**KL consistency** compares forward and reversed pairs after exchanging the `First`/`Second` probabilities. It uses 45,000 unordered pairs, both orders of each pair, an effective batch of 32 forward plus 32 reversed instances, and a KL coefficient of 4. It achieves **98.36% forward/reverse consistency**. Inconsistent predictions are removed before score aggregation, so this row reflects both the training regularizer and that inference treatment. The resulting loss of usable comparisons, including `Similar` pairs, provides an explanation for the lack of a score-level gain despite better symmetry.

## Checkpoint ensemble

The ensemble averages the three candidate logits **before softmax**, using five checkpoints between epochs 4 and 5 of the balanced three-class CE run.

| Comparator | PC (%) ↑ | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: | ---: |
| Single checkpoint, epoch 5 | 95.26 | 0.1730 | 0.2150 |
| Five-checkpoint ensemble | 95.32 | 0.1720 | 0.2141 |

All three score metrics improve slightly. The ensemble reuses one training run and requires five comparator evaluations per pair; it is not an ensemble of cross-validation folds or independent training runs.

## Evaluation scope

The ablation study uses one **4,400 / 1,100 train/test split**, five epochs, and effective batch size 64. Reference images come only from the training split, with at most 10 images per 0.1-wide score bin. The similarity margin is 0.2 unless otherwise stated.

The balanced baseline's BT metrics were evaluated on **200 sampled test targets**. Target counts for the remaining ablation rows are not specified with the recorded tables. The 1,100-target count for the released checkpoint therefore must not be assigned to the ablation results. No separate validation set or validation-based early stopping was used in the ablation study.

Ablation values correspond to Tables 3-2–3-4 of *Order Learning with Large Multimodal Models for Facial Beauty Prediction*. They are separate from the linked release artifacts. The [implementation notes](TRAINING_ALIGNMENT.md#experiment-settings-and-public-defaults) compare the ablation settings with the public defaults.

The public repository implements balanced three-class CE training, checkpoint management, pairwise inference, and score estimation. Imbalanced/two-class training, weighted CE, KL regularization, and checkpoint ensembling are documented experiments rather than runnable modes.
