#!/usr/bin/env python
"""Estimate SCUT-FBP5500 scores from LMOL pairwise predictions."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from configs import config_for_fold, load_config
from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR
from evaluation.bradley_terry import ReferenceComparison, estimate_absolute_scores_from_references
from evaluation.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Fold index for default output paths. Defaults to dataset.fold_index from the config.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="Output metrics JSON.")
    args = parser.parse_args()
    config = config_for_fold(load_config(args.config), args.fold_index)
    comparisons: dict[str, list[ReferenceComparison]] = defaultdict(list)
    true_scores: dict[str, float] = {}
    with Path(args.predictions).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prediction = row["prediction"]
            if prediction not in {ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR}:
                continue
            image = row["image1"]
            true_scores[image] = float(row["score1"])
            comparisons[image].append(ReferenceComparison(reference_score=float(row["score2"]), prediction=prediction))
    estimates = estimate_absolute_scores_from_references(
        comparisons,
        theta=config.pairs.theta,
        score_min=config.dataset.score_min,
        score_max=config.dataset.score_max,
        grid_size=config.evaluation.bt_grid_size,
    )
    images = sorted(estimates)
    y_true = [true_scores[image] for image in images]
    y_pred = [estimates[image] for image in images]
    metrics = {
        "count": len(images),
        "pc": pearson_correlation(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }
    output = Path(args.output) if args.output else config.evaluation.output_dir / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
