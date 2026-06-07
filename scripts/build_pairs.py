#!/usr/bin/env python
"""Build offline LMOL instruction pairs for SCUT-FBP5500 folds."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - used when imported by tests
    from scripts import _bootstrap  # type: ignore  # noqa: F401

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from configs import LMOLConfig, load_config
from data.pairs import build_balanced_pairs, build_reference_pairs, write_pairs_csv
from data.scut_fbp5500 import SCUTRecord
from data.scut_fbp5500 import load_annotations, split_train_test, validate_dataset, write_image_split_csv


@dataclass(frozen=True)
class FoldBuildResult:
    """Summary of generated CSVs for one fold."""

    fold_index: int
    output_dir: Path
    train_pair_count: int
    reference_pair_count: int
    label_counts: dict[str, int]


def build_pairs_for_fold(
    config: LMOLConfig,
    records: list[SCUTRecord],
    fold_index: int,
    allow_partial: bool = False,
) -> FoldBuildResult:
    """Build train and reference pair CSVs for one fold."""

    train_records, test_records = split_train_test(
        records,
        folds=config.dataset.folds,
        fold_index=fold_index,
        seed=config.dataset.split_seed,
    )
    out_dir = config.pairs.output_dir / f"fold_{fold_index}"
    write_image_split_csv(train_records, out_dir / "train_images.csv")
    write_image_split_csv(test_records, out_dir / "test_images.csv")
    train_pairs = build_balanced_pairs(
        train_records,
        pairs_per_label=config.pairs.train_pairs_per_label,
        seed=config.pairs.seed,
        theta=config.pairs.theta,
        fold=fold_index,
        split="train",
        require_full=not allow_partial,
    )
    write_pairs_csv(train_pairs, out_dir / "train_pairs.csv")
    references = train_records[: config.evaluation.reference_count]
    reference_pairs = build_reference_pairs(test_records, references, theta=config.pairs.theta, fold=fold_index)
    write_pairs_csv(reference_pairs, out_dir / "test_reference_pairs.csv")
    return FoldBuildResult(
        fold_index=fold_index,
        output_dir=out_dir,
        train_pair_count=len(train_pairs),
        reference_pair_count=len(reference_pairs),
        label_counts=dict(Counter(pair.answer for pair in train_pairs)),
    )


def build_pairs_for_folds(
    config: LMOLConfig,
    records: list[SCUTRecord],
    fold_indices: Iterable[int],
    allow_partial: bool = False,
) -> list[FoldBuildResult]:
    """Build train and reference pair CSVs for each requested fold."""

    return [
        build_pairs_for_fold(config=config, records=records, fold_index=fold_index, allow_partial=allow_partial)
        for fold_index in fold_indices
    ]


def _fold_indices(config: LMOLConfig, fold_index: int | None) -> list[int]:
    if fold_index is None:
        return list(range(config.dataset.folds))
    if not 0 <= fold_index < config.dataset.folds:
        raise ValueError(f"fold-index must be in [0, {config.dataset.folds - 1}], got {fold_index}")
    return [fold_index]


def _print_result(result: FoldBuildResult) -> None:
    print(f"Fold {result.fold_index}: wrote {result.train_pair_count} training pairs to {result.output_dir / 'train_pairs.csv'}")
    print(
        f"Fold {result.fold_index}: wrote {result.reference_pair_count} test-reference pairs "
        f"to {result.output_dir / 'test_reference_pairs.csv'}"
    )
    print(f"Fold {result.fold_index}: training label counts: {result.label_counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Build only one fold. By default, all configured folds are built.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer pairs than requested when candidates are scarce.")
    args = parser.parse_args()
    config = load_config(args.config)
    records = load_annotations(config.dataset.root, config.dataset.image_dir, config.dataset.annotation_file)
    validate_dataset(
        records,
        expected_count=config.dataset.expected_count,
        score_min=config.dataset.score_min,
        score_max=config.dataset.score_max,
        require_images=True,
    )
    try:
        fold_indices = _fold_indices(config, args.fold_index)
    except ValueError as exc:
        parser.error(str(exc))
    results = build_pairs_for_folds(config=config, records=records, fold_indices=fold_indices, allow_partial=args.allow_partial)
    for result in results:
        _print_result(result)
    if len(results) > 1:
        print(f"Built {len(results)} folds under {config.pairs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
