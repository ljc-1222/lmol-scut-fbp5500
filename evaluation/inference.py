"""Pairwise LMOL answer-token choice scoring."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from data.instruction_dataset import LMOLDataCollator
from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR, CANONICAL_ANSWERS, PairExample

VALID_ANSWERS = CANONICAL_ANSWERS
ANSWER_LOGPROB_COLUMNS = {
    ANSWER_FIRST: "logprob_first",
    ANSWER_SECOND: "logprob_second",
    ANSWER_SIMILAR: "logprob_similar",
}
PREDICTION_DECODING_METHOD = "first_token_choice"
PREDICTION_CSV_FIELDS = [
    "target_image",
    "reference_image",
    "target_score",
    "reference_score",
    "true_answer",
    "predicted_answer",
    "decoding_method",
    "valid_choice",
    "logprob_first",
    "logprob_second",
    "logprob_similar",
    "nll",
]


@dataclass(frozen=True)
class LikelihoodPrediction:
    """One pairwise answer selected by normalized answer-token log probability."""

    prediction: str
    answer_logprobs: dict[str, float]


@dataclass(frozen=True)
class PairwiseEvalMetrics:
    """Pairwise choice-token evaluation metrics."""

    loss: float
    accuracy: float
    count: int
    label_counts: dict[str, int]
    prediction_counts: dict[str, int]


# ponytail: 512 keeps the default fold references hot; raise it only for larger reference sets.
@lru_cache(maxsize=512)
def _load_rgb_image(path: Path) -> Any:
    """Load and cache one RGB image for repeated target-reference comparisons."""

    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _load_pair_images(pair: PairExample) -> list[Any]:
    """Load the two RGB images for one pair."""

    return [_load_rgb_image(pair.image1_path), _load_rgb_image(pair.image2_path)]


def select_balanced_pairs(pairs: Sequence[PairExample], max_pairs: int | None = None) -> list[PairExample]:
    """Select a deterministic label-balanced subset for checkpoint evaluation."""

    if max_pairs is None or max_pairs >= len(pairs):
        return list(pairs)
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive when provided")

    grouped: dict[str, list[PairExample]] = {answer: [] for answer in VALID_ANSWERS}
    for pair in pairs:
        grouped.setdefault(pair.answer, []).append(pair)

    base_quota = max_pairs // len(VALID_ANSWERS)
    remainder = max_pairs % len(VALID_ANSWERS)
    selected: list[PairExample] = []
    selected_ids: set[int] = set()
    for answer_index, answer in enumerate(VALID_ANSWERS):
        quota = base_quota + int(answer_index < remainder)
        for pair in grouped.get(answer, [])[:quota]:
            selected.append(pair)
            selected_ids.add(id(pair))

    if len(selected) < max_pairs:
        for pair in pairs:
            if id(pair) in selected_ids:
                continue
            selected.append(pair)
            selected_ids.add(id(pair))
            if len(selected) >= max_pairs:
                break
    return selected[:max_pairs]


def _choice_answer_logprobs_batch(
    model: Any,
    processor: Any,
    pairs: Sequence[PairExample],
) -> list[LikelihoodPrediction]:
    """Return normalized answer-token predictions for a batch of pairs."""

    import torch
    import torch.nn.functional as F

    if not pairs:
        return []
    texts: list[str] = []
    images: list[Any] = []
    for pair in pairs:
        texts.append(LMOLDataCollator.render_prompt(processor, pair.instruction).rstrip())
        images.extend(_load_pair_images(pair))

    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    choice_positions = LMOLDataCollator.prompt_choice_positions(batch)
    choice_token_ids = torch.tensor(LMOLDataCollator.answer_token_ids(processor.tokenizer), dtype=torch.long)
    device = next(model.parameters()).device
    model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    choice_positions = choice_positions.to(device)
    choice_token_ids = choice_token_ids.to(device)

    with torch.no_grad():
        logits = model(**model_inputs).logits
        row_indices = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[row_indices, choice_positions, :].float()
        choice_logits = next_token_logits.index_select(dim=-1, index=choice_token_ids)
        log_probs = F.log_softmax(choice_logits, dim=-1)

    predictions: list[LikelihoodPrediction] = []
    for row in log_probs.detach().cpu():
        scores = [float(row[index]) for index in range(len(VALID_ANSWERS))]
        answer_logprobs = {answer: scores[index] for index, answer in enumerate(VALID_ANSWERS)}
        prediction = max(VALID_ANSWERS, key=lambda answer: answer_logprobs[answer])
        predictions.append(LikelihoodPrediction(prediction=prediction, answer_logprobs=answer_logprobs))
    return predictions


def predict_answers_by_likelihood(
    model: Any,
    processor: Any,
    pairs: Sequence[PairExample],
    batch_size: int = 1,
) -> list[LikelihoodPrediction]:
    """Select LMOL answers by maximum normalized answer-token likelihood."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    predictions: list[LikelihoodPrediction] = []
    for start in range(0, len(pairs), batch_size):
        predictions.extend(_choice_answer_logprobs_batch(model, processor, pairs[start : start + batch_size]))
    return predictions


def predict_answer_by_likelihood(model: Any, processor: Any, pair: PairExample) -> LikelihoodPrediction:
    """Select the LMOL answer with maximum normalized answer-token likelihood."""

    return predict_answers_by_likelihood(model, processor, [pair], batch_size=1)[0]


def evaluate_pairwise_likelihood(
    model: Any,
    processor: Any,
    pairs: Sequence[PairExample],
    max_pairs: int | None = None,
    batch_size: int = 1,
) -> PairwiseEvalMetrics:
    """Evaluate normalized answer-token NLL and class accuracy on pairwise examples."""

    eval_pairs = select_balanced_pairs(pairs, max_pairs=max_pairs)
    limit = len(eval_pairs)
    if limit <= 0:
        raise ValueError("Pairwise evaluation requires at least one example")
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        predictions = predict_answers_by_likelihood(model, processor, eval_pairs, batch_size=batch_size)
    finally:
        if was_training:
            model.train()
    total_loss = 0.0
    correct = 0
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    for pair, result in zip(eval_pairs, predictions):
        total_loss += -result.answer_logprobs[pair.answer]
        correct += int(result.prediction == pair.answer)
        label_counts[pair.answer] += 1
        prediction_counts[result.prediction] += 1
    return PairwiseEvalMetrics(
        loss=total_loss / limit,
        accuracy=correct / limit,
        count=limit,
        label_counts=dict(label_counts),
        prediction_counts=dict(prediction_counts),
    )


def prediction_csv_row(pair: PairExample, result: LikelihoodPrediction) -> dict[str, object]:
    """Return one first-token-choice prediction row."""

    row: dict[str, object] = {
        "target_image": pair.image1,
        "reference_image": pair.image2,
        "target_score": pair.score1,
        "reference_score": pair.score2,
        "true_answer": pair.answer,
        "predicted_answer": result.prediction,
        "decoding_method": PREDICTION_DECODING_METHOD,
        "valid_choice": result.prediction in VALID_ANSWERS,
        "nll": -result.answer_logprobs[pair.answer],
    }
    for answer, column in ANSWER_LOGPROB_COLUMNS.items():
        row[column] = result.answer_logprobs[answer]
    return row


def write_predictions_csv(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write pairwise likelihood predictions to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
