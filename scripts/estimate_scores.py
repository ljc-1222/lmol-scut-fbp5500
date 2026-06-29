#!/usr/bin/env python
"""Estimate SCUT-FBP5500 scores from LMOL pairwise predictions."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - used when imported by tests
    from scripts import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
from pathlib import Path

from configs import config_for_fold, load_config
from evaluation.score_estimation import estimate_metrics_from_predictions_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Fold index for default output paths. Defaults to dataset.fold_index from the config.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="Output metrics JSON.")
    args = parser.parse_args()
    config = config_for_fold(load_config(args.config), args.fold_index)
    output = Path(args.output) if args.output else config.evaluation.output_dir / "metrics.json"
    metrics = estimate_metrics_from_predictions_csv(
        args.predictions,
        delta=config.evaluation.bt_delta,
        k=config.evaluation.bt_k,
        output_path=output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
