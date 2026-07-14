"""UOL-style reference selection and leakage checks."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from .scut_fbp5500 import SCUTRecord


@dataclass(frozen=True)
class ReferenceRecord:
    """One selected known-score reference image for a fold."""

    record: SCUTRecord
    fold: int
    score_bin: str


def score_bin_index(score: float, score_min: float, score_max: float, bin_width: float) -> int:
    """Return the deterministic 0.1-style bin index for a score."""

    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    if not score_min <= score <= score_max:
        raise ValueError(f"score {score} is outside [{score_min}, {score_max}]")
    bin_count = max(1, int(math.ceil((score_max - score_min) / bin_width)))
    raw_index = int(math.floor((score - score_min) / bin_width + 1e-12))
    return min(max(raw_index, 0), bin_count - 1)


def score_bin_label(index: int, score_min: float, bin_width: float) -> str:
    """Return a stable label using the lower edge of the score bin."""

    return format(Decimal(str(score_min)) + index * Decimal(str(bin_width)), "f")


def assert_no_reference_leakage(
    train_records: Sequence[SCUTRecord],
    test_records: Sequence[SCUTRecord],
    references: Sequence[SCUTRecord],
) -> None:
    """Fail if train/test/reference split invariants are violated."""

    train = {record.image for record in train_records}
    test = {record.image for record in test_records}
    ref = {record.image for record in references}
    train_test_overlap = train & test
    if train_test_overlap:
        preview = ", ".join(sorted(train_test_overlap)[:5])
        raise ValueError(f"Train/test split leakage detected: {preview}")
    outside_train = ref - train
    if outside_train:
        preview = ", ".join(sorted(outside_train)[:5])
        raise ValueError(f"Reference images must be selected from the train split only: {preview}")


def select_score_bin_references(
    train_records: Sequence[SCUTRecord],
    *,
    fold: int,
    score_min: float,
    score_max: float,
    bin_width: float = 0.1,
    max_per_bin: int | None = None,
    seed: int = 42,
) -> list[ReferenceRecord]:
    """Select references from each non-empty train-score bin."""

    if max_per_bin is not None and max_per_bin <= 0:
        raise ValueError(f"max_per_bin must be positive when set, got {max_per_bin}")
    grouped: dict[int, list[SCUTRecord]] = {}
    for record in sorted(train_records, key=lambda item: item.image):
        index = score_bin_index(record.score, score_min=score_min, score_max=score_max, bin_width=bin_width)
        grouped.setdefault(index, []).append(record)

    rng = random.Random(seed)
    selected: list[ReferenceRecord] = []
    for index in sorted(grouped):
        records = list(grouped[index])
        if max_per_bin is not None and len(records) > max_per_bin:
            records = sorted(rng.sample(records, max_per_bin), key=lambda item: item.image)
        label = score_bin_label(index, score_min=score_min, bin_width=bin_width)
        selected.extend(ReferenceRecord(record=record, fold=fold, score_bin=label) for record in records)
    return selected


def write_references_csv(references: Iterable[ReferenceRecord], path: str | Path) -> None:
    """Write the selected reference manifest for a fold."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "score", "fold", "split", "score_bin"])
        writer.writeheader()
        for reference in references:
            writer.writerow(
                {
                    "image": reference.record.image,
                    "score": reference.record.score,
                    "fold": reference.fold,
                    "split": "reference",
                    "score_bin": reference.score_bin,
                }
            )
