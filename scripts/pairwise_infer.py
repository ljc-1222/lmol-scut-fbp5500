#!/usr/bin/env python
"""Run pairwise LMOL likelihood inference for SCUT-FBP5500 pairs."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from configs import config_for_fold, load_config
from data.pairs import read_pairs_csv
from evaluation.inference import predict_answers_by_likelihood, prediction_csv_row, write_predictions_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Fold index to evaluate. Defaults to dataset.fold_index from the config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", help="Path to test_reference_pairs.csv.")
    parser.add_argument("--output", help="Output predictions CSV.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N pairs for a quick sanity check.")
    parser.add_argument("--batch-size", type=int, help="Pair batch size for likelihood scoring. Defaults to evaluation.batch_size.")
    args = parser.parse_args()
    config = config_for_fold(load_config(args.config), args.fold_index)
    try:
        from peft import PeftModel
        from model import load_lmol_model, load_projector_state
    except Exception as exc:
        raise RuntimeError("Install runtime dependencies before pairwise inference") from exc

    pairs_csv = Path(args.pairs) if args.pairs else config.pairs.output_dir / f"fold_{config.dataset.fold_index}" / "test_reference_pairs.csv"
    output = Path(args.output) if args.output else config.evaluation.output_dir / "pairwise_predictions.csv"
    base_model, processor = load_lmol_model(config, train=False)
    model = PeftModel.from_pretrained(base_model, args.checkpoint, is_trainable=False)
    load_projector_state(model, args.checkpoint)
    pairs = read_pairs_csv(pairs_csv)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    batch_size = args.batch_size if args.batch_size is not None else config.evaluation.batch_size
    predictions = predict_answers_by_likelihood(model, processor, pairs, batch_size=batch_size)
    rows = []
    total_loss = 0.0
    correct = 0
    for pair, result in zip(pairs, predictions):
        total_loss += -result.answer_logprobs[pair.answer]
        correct += int(result.prediction == pair.answer)
        rows.append(prediction_csv_row(pair, result))
    write_predictions_csv(rows, output)
    count = len(rows)
    accuracy = correct / count if count else 0.0
    loss = total_loss / count if count else 0.0
    print(f"Wrote likelihood predictions to {output}; count={count}; pairwise_loss={loss}; pairwise_acc={accuracy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
