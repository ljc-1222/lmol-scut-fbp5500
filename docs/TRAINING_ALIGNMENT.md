# Method and reproduction settings

This implementation follows [LMOL](https://doi.org/10.1109/ICASSP49660.2025.10889943), with score recovery based on [UOL](https://arxiv.org/abs/2409.00603). The commands support training and evaluation on SCUT-FBP5500. The paper also evaluates generalization on Hot-Or-Not and MEBeauty; those datasets have no loaders or evaluation commands here.

## Data and folds

For two images with mean ratings $y_1$ and $y_2$, the label follows LMOL Eq. (1):

$$
\operatorname{label}(y_1,y_2)=
\begin{cases}
\text{First.}, & y_1-y_2>0.2,\\
\text{Second.}, & y_1-y_2<-0.2,\\
\text{Similar.}, & |y_1-y_2|\leq 0.2.
\end{cases}
$$

[`data/pairs.py`](../data/pairs.py) samples 30,000 ordered pairs per class from distinct training images, producing 90,000 instruction pairs per fold. The stored instruction is `<image1><image2>, which face looks more attractive?`; the processor renders the two image placeholders through its chat template.

[`data/scut_fbp5500.py`](../data/scut_fbp5500.py) sorts image names, shuffles with `dataset.split_seed`, and distributes them across five folds. Each run uses 4,400 training images and 1,100 test images. These are locally constructed folds with default seed 42; the loader does not import official split manifests. The builder saves the actual image lists in `train_images.csv` and `test_images.csv`.

## Model and optimization

LMOL §III-B specifies the backbone, QLoRA rank/scaling, pair count, epoch count, and learning rates below. The final two rows identify additional settings in the [default YAML](../configs/lmol_scut_fbp5500.yaml).

<div align="center">
<table width="100%" align="center">
<thead>
<tr><th width="10000" align="center">Setting</th><th width="10000" align="center">LMOL paper</th><th width="10000" align="center">Default configuration</th></tr>
</thead>
<tbody>
<tr><td align="center">Backbone</td><td align="center">LLaVA-1.5-7B</td><td align="center"><code>llava-hf/llava-1.5-7b-hf</code></td></tr>
<tr><td align="center">Vision encoder</td><td align="center">CLIP ViT-L/14-336px</td><td align="center">Same; frozen</td></tr>
<tr><td align="center">Trainable parameters</td><td align="center">Full projector + language-model QLoRA</td><td align="center">Same; projector and LoRA optimizer groups</td></tr>
<tr><td align="center">Pairs / epochs per fold</td><td align="center">90,000 / 1</td><td align="center">90,000 / 1</td></tr>
<tr><td align="center">LoRA rank / scaling</td><td align="center">8 / 4</td><td align="center">Rank 8; alpha 32, so alpha / rank = 4</td></tr>
<tr><td align="center">Projector / LoRA learning rate</td><td align="center">2 × 10<sup>−5</sup> / 2 × 10<sup>−4</sup></td><td align="center">Same</td></tr>
<tr><td align="center">Optimizer / schedule</td><td align="center">AdamW / cosine decay to 0</td><td align="center">Same</td></tr>
<tr><td align="center">Effective batch size</td><td align="center">Not specified</td><td align="center">16: microbatch 1 × accumulation 16</td></tr>
<tr><td align="center">LoRA target modules / dropout</td><td align="center">Not specified</td><td align="center">Language-model q/k/v/o projections / 0.05</td></tr>
</tbody>
</table>
</div>

[`model/lmol.py`](../model/lmol.py) keeps the vision tower and multimodal projector outside 4-bit linear quantization, freezes the vision tower, and enables training for the complete projector and language-model LoRA parameters.

## Training objective and decoding

LMOL §II-D, Eqs. (9)–(10), describes conditional language modeling over the answer sequence. The current code uses cross-entropy over only the three candidate first-answer-token logits. With vocabulary logit $z_{t_c}$ for candidate $c$ and gold class $g$:

$$
\mathcal{L}=-\log\frac{\exp(z_{t_g})}{\sum_{c\in\{\mathrm{First},\mathrm{Second},\mathrm{Similar}\}}\exp(z_{t_c})}.
$$

[`data/instruction_dataset.py`](../data/instruction_dataset.py) locates the first answer word, and [`training/train.py`](../training/train.py) selects the preceding logits and applies this three-class loss. Punctuation and subsequent answer tokens are excluded from the objective. [`evaluation/inference.py`](../evaluation/inference.py) applies softmax to the same three vocabulary entries at the last prompt position and selects the highest-probability class without text generation.

The quickstart disables training-time evaluation with `--no-eval`. Monitoring and final evaluation options are documented in [Usage](USAGE.md#training-and-checkpoints).

## Reference selection and score recovery

[`data/references.py`](../data/references.py) follows UOL §IV-A: divide training ratings into 0.1-wide bins and retain up to 10 images per non-empty bin. Reference selection is seeded and restricted to the training split. Setting `evaluation.reference_max_per_bin: null` retains all training images in each bin; rebuild the pair files after changing this setting.

[`evaluation/uol_bradley_terry.py`](../evaluation/uol_bradley_terry.py) implements the cumulative-logistic likelihood based on UOL Eqs. (11)–(12), using target-versus-reference labels. For candidate target rating $s$, reference rating $r$, $d=s-r$, and sigmoid $\sigma$:

$$
\begin{aligned}
P(\mathrm{First}) &= \sigma(kd-\delta),\\
P(\mathrm{Similar}) &= \sigma(kd+\delta)-\sigma(kd-\delta),\\
P(\mathrm{Second}) &= 1-\sigma(kd+\delta).
\end{aligned}
$$

The estimator sums the log probability of the predicted class over all references and selects the best target rating from the unique reference-score support. Tied maxima return the mean of the tied scores. Pairwise softmax probabilities are recorded in the prediction CSV; score recovery uses the selected class labels.

The configuration defaults are `evaluation.bt_delta: 2.0` and `evaluation.bt_k: 10.0`. These values are implementation settings: the LMOL paper refers to the UOL estimator without specifying them. Held-out target ratings are used for score metrics, while the score-recovery likelihood uses reference ratings and predicted order labels.
