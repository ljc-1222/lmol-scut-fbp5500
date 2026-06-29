#!/usr/bin/env python
"""Run full fold evaluation with incremental progress reporting."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - used when imported by tests
    from scripts import _bootstrap  # type: ignore  # noqa: F401

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from configs import config_for_fold, load_config
from data.pairs import ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR, read_pairs_csv
from evaluation.inference import PREDICTION_CSV_FIELDS, predict_answers_by_likelihood, prediction_csv_row
from evaluation.score_estimation import estimate_metrics_from_predictions_csv


VALID_ANSWERS = (ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _progress_payload(
    *,
    phase: str,
    total_pairs: int,
    processed_pairs: int,
    correct_pairs: int,
    loss_sum: float,
    start_time: float,
    predictions: Path,
    metrics: Path,
    confusion: Path,
) -> dict[str, Any]:
    elapsed = time.time() - start_time
    rate = processed_pairs / elapsed if elapsed > 0 and processed_pairs > 0 else 0.0
    remaining = total_pairs - processed_pairs
    eta = remaining / rate if rate > 0 else None
    return {
        "phase": phase,
        "total_pairs": total_pairs,
        "processed_pairs": processed_pairs,
        "percent": (processed_pairs / total_pairs * 100.0) if total_pairs else 0.0,
        "pairwise_accuracy_running": (correct_pairs / processed_pairs) if processed_pairs else 0.0,
        "pairwise_loss_running": (loss_sum / processed_pairs) if processed_pairs else 0.0,
        "pairs_per_sec": rate,
        "elapsed_sec": elapsed,
        "eta_sec": eta,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": str(predictions),
        "metrics": str(metrics),
        "confusion": str(confusion),
    }


def _write_confusion_matrix(path: Path, matrix: dict[str, Counter[str]], count: int, correct: int) -> None:
    payload = {
        "labels": list(VALID_ANSWERS),
        "count": count,
        "pairwise_accuracy": correct / count if count else 0.0,
        "confusion_matrix_rows_gold_cols_pred": {
            gold: [matrix[gold][pred] for pred in VALID_ANSWERS] for gold in VALID_ANSWERS
        },
    }
    _write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Fold index to evaluate. Defaults to dataset.fold_index.")
    parser.add_argument("--checkpoint", help="Checkpoint directory. Defaults to the fold-scoped training output last checkpoint.")
    parser.add_argument("--pairs", help="Path to test_reference_pairs.csv.")
    parser.add_argument("--predictions", help="Output pairwise predictions CSV.")
    parser.add_argument("--metrics", help="Output BT score metrics JSON.")
    parser.add_argument("--confusion", help="Output pairwise confusion matrix JSON.")
    parser.add_argument("--progress", help="Output progress JSON.")
    parser.add_argument("--batch-size", type=int, help="Pair batch size. Defaults to evaluation.batch_size.")
    parser.add_argument("--progress-pairs", type=int, default=2048, help="Update progress every N processed pairs.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N pairs.")
    args = parser.parse_args()

    config = config_for_fold(load_config(args.config), args.fold_index)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.training.output_dir / "last"
    pairs_csv = Path(args.pairs) if args.pairs else config.pairs.output_dir / f"fold_{config.dataset.fold_index}" / "test_reference_pairs.csv"
    predictions = Path(args.predictions) if args.predictions else config.evaluation.output_dir / "pairwise_predictions.csv"
    metrics_output = Path(args.metrics) if args.metrics else config.evaluation.output_dir / "metrics.json"
    confusion_output = Path(args.confusion) if args.confusion else config.evaluation.output_dir / "confusion_matrix.json"
    progress_output = Path(args.progress) if args.progress else config.evaluation.output_dir / "progress.json"
    batch_size = args.batch_size if args.batch_size is not None else config.evaluation.batch_size
    if batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {batch_size}")

    try:
        from peft import PeftModel
        from model import load_lmol_model, load_projector_state
    except Exception as exc:
        raise RuntimeError("Install runtime dependencies before evaluation") from exc

    pairs = read_pairs_csv(pairs_csv)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    total_pairs = len(pairs)
    start_time = time.time()
    predictions.parent.mkdir(parents=True, exist_ok=True)

    initial_progress = _progress_payload(
        phase="loading_model",
        total_pairs=total_pairs,
        processed_pairs=0,
        correct_pairs=0,
        loss_sum=0.0,
        start_time=start_time,
        predictions=predictions,
        metrics=metrics_output,
        confusion=confusion_output,
    )
    initial_progress.update({"checkpoint": str(checkpoint), "pairs_csv": str(pairs_csv), "batch_size": batch_size})
    _write_json(progress_output, initial_progress)
    print(json.dumps(initial_progress, sort_keys=True), flush=True)

    base_model, processor = load_lmol_model(config, train=False)
    model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
    load_projector_state(model, checkpoint)

    confusion = {gold: Counter({pred: 0 for pred in VALID_ANSWERS}) for gold in VALID_ANSWERS}
    processed = 0
    correct = 0
    loss_sum = 0.0
    next_progress = max(1, args.progress_pairs)

    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        for start in range(0, total_pairs, batch_size):
            batch_pairs = pairs[start : start + batch_size]
            batch_predictions = predict_answers_by_likelihood(model, processor, batch_pairs, batch_size=batch_size)
            for pair, result in zip(batch_pairs, batch_predictions):
                loss_sum += -result.answer_logprobs[pair.answer]
                correct += int(result.prediction == pair.answer)
                confusion[pair.answer][result.prediction] += 1
                writer.writerow(prediction_csv_row(pair, result))
                processed += 1
            if processed >= next_progress or processed >= total_pairs:
                handle.flush()
                payload = _progress_payload(
                    phase="pairwise_inference",
                    total_pairs=total_pairs,
                    processed_pairs=processed,
                    correct_pairs=correct,
                    loss_sum=loss_sum,
                    start_time=start_time,
                    predictions=predictions,
                    metrics=metrics_output,
                    confusion=confusion_output,
                )
                payload.update({"checkpoint": str(checkpoint), "pairs_csv": str(pairs_csv), "batch_size": batch_size})
                _write_json(progress_output, payload)
                print(json.dumps(payload, sort_keys=True), flush=True)
                next_progress = processed + max(1, args.progress_pairs)

    _write_confusion_matrix(confusion_output, confusion, processed, correct)
    metrics_payload = _progress_payload(
        phase="estimating_scores",
        total_pairs=total_pairs,
        processed_pairs=processed,
        correct_pairs=correct,
        loss_sum=loss_sum,
        start_time=start_time,
        predictions=predictions,
        metrics=metrics_output,
        confusion=confusion_output,
    )
    metrics_payload.update({"checkpoint": str(checkpoint), "pairs_csv": str(pairs_csv), "batch_size": batch_size})
    _write_json(progress_output, metrics_payload)
    print(json.dumps(metrics_payload, sort_keys=True), flush=True)

    metrics = estimate_metrics_from_predictions_csv(
        predictions,
        delta=config.evaluation.bt_delta,
        k=config.evaluation.bt_k,
        output_path=metrics_output,
    )
    complete_payload = _progress_payload(
        phase="complete",
        total_pairs=total_pairs,
        processed_pairs=processed,
        correct_pairs=correct,
        loss_sum=loss_sum,
        start_time=start_time,
        predictions=predictions,
        metrics=metrics_output,
        confusion=confusion_output,
    )
    complete_payload.update(
        {
            "checkpoint": str(checkpoint),
            "pairs_csv": str(pairs_csv),
            "batch_size": batch_size,
            "score_metrics": metrics,
        }
    )
    _write_json(progress_output, complete_payload)
    print(json.dumps(complete_payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
