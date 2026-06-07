"""SCUT-FBP5500 score metrics used by the LMOL paper."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _arrays(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {pred.shape}")
    if true.size == 0:
        raise ValueError("Metric inputs cannot be empty")
    return true, pred


def pearson_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Pearson Correlation Coefficient."""

    true, pred = _arrays(y_true, y_pred)
    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denom = math.sqrt(float(np.sum(true_centered**2) * np.sum(pred_centered**2)))
    if denom == 0:
        raise ValueError("Pearson correlation is undefined for constant inputs")
    return float(np.sum(true_centered * pred_centered) / denom)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Mean Absolute Error."""

    true, pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Root Mean Squared Error."""

    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))
