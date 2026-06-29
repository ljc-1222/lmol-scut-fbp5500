"""Shared prediction parsing, UOL BT score estimation, and score metrics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR
from evaluation.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error
from evaluation.uol_bradley_terry import ReferenceComparison, estimate_scores_uol_bt

VALID_ANSWERS = (ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR)
DEFAULT_SCORE_ESTIMATOR = "uol_bradley_terry"


@dataclass(frozen=True)
class PairwisePredictionRow:
    """One normalized target-reference prediction row."""

    target_image: str
    target_score: float
    reference_score: float
    predicted_answer: str
    valid_choice: bool


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _csv_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _row_value(row: dict[str, str], key: str, *legacy_keys: str) -> str:
    value = row.get(key)
    if value not in (None, ""):
        return value
    for legacy_key in legacy_keys:
        value = row.get(legacy_key)
        if value not in (None, ""):
            return value
    raise KeyError(f"Missing required prediction CSV column: {key}")


def _row_valid_choice(row: dict[str, str], predicted_answer: str) -> bool:
    if "valid_choice" in row:
        return _csv_bool(row.get("valid_choice"), default=False)
    return predicted_answer in VALID_ANSWERS


def load_pairwise_predictions(path: str | Path) -> list[PairwisePredictionRow]:
    """Load pairwise prediction CSV files from the current evaluator."""

    rows: list[PairwisePredictionRow] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predicted_answer = _row_value(row, "predicted_answer", "prediction")
            rows.append(
                PairwisePredictionRow(
                    target_image=_row_value(row, "target_image", "image1"),
                    target_score=float(_row_value(row, "target_score", "score1")),
                    reference_score=float(_row_value(row, "reference_score", "score2")),
                    predicted_answer=predicted_answer,
                    valid_choice=_row_valid_choice(row, predicted_answer),
                )
            )
    return rows


def build_reference_comparisons(
    predictions: Iterable[PairwisePredictionRow],
) -> tuple[dict[str, list[ReferenceComparison]], dict[str, float]]:
    """Build per-target UOL BT comparisons from predicted answer labels."""

    comparisons: dict[str, list[ReferenceComparison]] = defaultdict(list)
    true_scores: dict[str, float] = {}
    for row in predictions:
        if not row.valid_choice or row.predicted_answer not in VALID_ANSWERS:
            continue
        true_scores[row.target_image] = row.target_score
        comparisons[row.target_image].append(
            ReferenceComparison(reference_score=row.reference_score, prediction=row.predicted_answer)
        )
    return dict(comparisons), true_scores


def compute_score_metrics(true_scores: dict[str, float], estimated_scores: dict[str, float]) -> dict[str, Any]:
    """Compute PC, MAE, and RMSE for estimated target scores."""

    images = sorted(image for image in estimated_scores if image in true_scores)
    if not images:
        raise ValueError("No estimated scores have matching target ground-truth scores")
    y_true = [true_scores[image] for image in images]
    y_pred = [estimated_scores[image] for image in images]
    metrics: dict[str, Any] = {
        "count": len(images),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }
    try:
        metrics["pc"] = pearson_correlation(y_true, y_pred)
    except ValueError as exc:
        metrics["pc"] = None
        metrics["pc_error"] = str(exc)
    return metrics


def estimate_scores_from_predictions(
    predictions: Iterable[PairwisePredictionRow],
    *,
    delta: float,
    k: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate target scores from normalized prediction rows."""

    comparisons, true_scores = build_reference_comparisons(predictions)
    estimates = estimate_scores_uol_bt(comparisons, delta=delta, k=k)
    return estimates, true_scores


def estimate_metrics_from_predictions_csv(
    predictions: str | Path,
    *,
    delta: float,
    k: float,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the shared prediction CSV -> UOL BT -> metrics pipeline."""

    rows = load_pairwise_predictions(predictions)
    estimates, true_scores = estimate_scores_from_predictions(rows, delta=delta, k=k)
    metrics = compute_score_metrics(true_scores, estimates)
    metrics.update({"score_estimator": DEFAULT_SCORE_ESTIMATOR, "bt_delta": delta, "bt_k": k})
    if output_path is not None:
        _write_json(Path(output_path), metrics)
    return metrics
