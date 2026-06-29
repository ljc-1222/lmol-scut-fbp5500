"""UOL Bradley-Terry score estimation over reference-score support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR

DEFAULT_BT_DELTA = 2.0
DEFAULT_BT_K = 10.0
VALID_PREDICTIONS = (ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR)


@dataclass(frozen=True)
class ReferenceComparison:
    """One predicted target-vs-reference relation."""

    reference_score: float
    prediction: str


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def uol_bt_class_probabilities(
    candidate_target_score: float,
    reference_score: float,
    delta: float = DEFAULT_BT_DELTA,
    k: float = DEFAULT_BT_K,
) -> dict[str, float]:
    """Return UOL Eq. (11) probabilities keyed by LMOL answer strings.

    The UOL cumulative relation is interpreted as reference-vs-target order:
    reference < target maps to `First.` for this repo's target-reference pairs.
    """

    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")
    score_diff = candidate_target_score - reference_score
    cdf_reference_less = _sigmoid(-delta + k * score_diff)
    cdf_reference_similar_or_less = _sigmoid(delta + k * score_diff)
    probabilities = {
        ANSWER_FIRST: cdf_reference_less,
        ANSWER_SIMILAR: cdf_reference_similar_or_less - cdf_reference_less,
        ANSWER_SECOND: 1.0 - cdf_reference_similar_or_less,
    }
    _validate_probabilities(probabilities)
    return probabilities


def _validate_probabilities(probabilities: dict[str, float], tolerance: float = 1e-9) -> None:
    values = list(probabilities.values())
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"UOL BT probabilities must be finite: {probabilities!r}")
    if any(value < -tolerance for value in values):
        raise FloatingPointError(f"UOL BT probabilities must be non-negative: {probabilities!r}")
    total = sum(values)
    if abs(total - 1.0) > tolerance:
        raise FloatingPointError(f"UOL BT probabilities must sum to 1, got {total}: {probabilities!r}")
    for key, value in probabilities.items():
        if value < 0.0:
            probabilities[key] = 0.0


def uol_bt_log_likelihood(
    candidate_target_score: float,
    comparisons: Sequence[ReferenceComparison],
    delta: float = DEFAULT_BT_DELTA,
    k: float = DEFAULT_BT_K,
    eps: float = 1e-12,
) -> float:
    """Return UOL Eq. (12) log likelihood for one candidate target score."""

    if not comparisons:
        raise ValueError("At least one reference comparison is required")
    log_likelihood = 0.0
    for comparison in comparisons:
        if comparison.prediction not in VALID_PREDICTIONS:
            raise ValueError(f"Invalid predicted answer for UOL BT estimation: {comparison.prediction!r}")
        probabilities = uol_bt_class_probabilities(
            candidate_target_score,
            comparison.reference_score,
            delta=delta,
            k=k,
        )
        log_likelihood += math.log(max(probabilities[comparison.prediction], eps))
    return log_likelihood


def reference_score_support(comparisons: Sequence[ReferenceComparison]) -> tuple[float, ...]:
    """Return the sorted unique support scores available from the references."""

    support = tuple(sorted({float(comparison.reference_score) for comparison in comparisons}))
    if not support:
        raise ValueError("Reference-score support cannot be empty")
    return support


def estimate_score_uol_bt(
    comparisons: Sequence[ReferenceComparison],
    candidate_scores: Sequence[float] | None = None,
    delta: float = DEFAULT_BT_DELTA,
    k: float = DEFAULT_BT_K,
    tie_tolerance: float = 1e-12,
) -> float:
    """Estimate one target score by maximizing UOL BT likelihood over support scores."""

    support = tuple(sorted({float(score) for score in candidate_scores})) if candidate_scores is not None else reference_score_support(comparisons)
    if not support:
        raise ValueError("candidate_scores cannot be empty")
    scored = [
        (score, uol_bt_log_likelihood(score, comparisons, delta=delta, k=k))
        for score in support
    ]
    best = max(log_likelihood for _score, log_likelihood in scored)
    tied = [score for score, log_likelihood in scored if abs(log_likelihood - best) <= tie_tolerance]
    return float(mean(tied))


def estimate_scores_uol_bt(
    comparisons_by_image: dict[str, Sequence[ReferenceComparison]],
    delta: float = DEFAULT_BT_DELTA,
    k: float = DEFAULT_BT_K,
) -> dict[str, float]:
    """Estimate each target score independently with UOL BT MLE."""

    return {
        image: estimate_score_uol_bt(comparisons, delta=delta, k=k)
        for image, comparisons in comparisons_by_image.items()
    }
