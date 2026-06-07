"""Bradley-Terry score estimation for LMOL pairwise predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR


@dataclass(frozen=True)
class ReferenceComparison:
    """One predicted comparison between a target image and a reference image."""

    reference_score: float
    prediction: str


def bradley_terry_scores(wins: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Estimate relative Bradley-Terry abilities from a square wins matrix."""

    matrix = np.asarray(wins, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("wins must be a square matrix")
    n_items = matrix.shape[0]
    abilities = np.ones(n_items, dtype=np.float64)
    games = matrix + matrix.T
    total_wins = matrix.sum(axis=1)
    for _ in range(max_iter):
        old = abilities.copy()
        for item in range(n_items):
            denom = 0.0
            for other in range(n_items):
                if item == other or games[item, other] == 0:
                    continue
                denom += games[item, other] / (abilities[item] + abilities[other])
            abilities[item] = total_wins[item] / denom if denom > 0 else old[item]
        mean_ability = abilities.mean()
        if mean_ability > 0:
            abilities /= mean_ability
        if np.max(np.abs(abilities - old)) < tol:
            break
    return abilities


def _comparison_loss(candidate_score: float, comparisons: Sequence[ReferenceComparison], theta: float) -> float:
    loss = 0.0
    for comparison in comparisons:
        diff = candidate_score - comparison.reference_score
        if comparison.prediction == ANSWER_FIRST:
            margin = diff - theta
        elif comparison.prediction == ANSWER_SECOND:
            margin = -diff - theta
        elif comparison.prediction == ANSWER_SIMILAR:
            margin = theta - abs(diff)
        else:
            raise ValueError(f"Invalid prediction: {comparison.prediction!r}")
        loss += float(np.logaddexp(0.0, -4.0 * margin))
    return loss


def estimate_absolute_scores_from_references(
    comparisons_by_image: dict[str, Sequence[ReferenceComparison]],
    theta: float = 0.2,
    score_min: float = 1.0,
    score_max: float = 5.0,
    grid_size: int = 401,
) -> dict[str, float]:
    """Estimate absolute beauty scores using predicted orders to known references."""

    grid = np.linspace(score_min, score_max, grid_size)
    estimates: dict[str, float] = {}
    for image, comparisons in comparisons_by_image.items():
        if not comparisons:
            raise ValueError(f"No comparisons available for {image}")
        losses = [_comparison_loss(float(candidate), comparisons, theta=theta) for candidate in grid]
        estimates[image] = float(grid[int(np.argmin(losses))])
    return estimates
