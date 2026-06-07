"""Offline LMOL instruction pair construction for SCUT-FBP5500."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .scut_fbp5500 import SCUTRecord

ANSWER_FIRST = "First."
ANSWER_SECOND = "Second."
ANSWER_SIMILAR = "Similar."
DEFAULT_INSTRUCTION = "<image1><image2>\nWhich face looks more attractive?"


@dataclass(frozen=True)
class PairExample:
    """One offline LMOL instruction example."""

    image1: str
    image2: str
    image1_path: Path
    image2_path: Path
    score1: float
    score2: float
    instruction: str
    answer: str
    fold: int
    split: str


def label_from_scores(score1: float, score2: float, theta: float = 0.2) -> str:
    """Convert two SCUT scores into the exact LMOL answer string."""

    diff = score1 - score2
    epsilon = 1e-12
    if abs(diff) <= theta + epsilon:
        return ANSWER_SIMILAR
    if diff > theta:
        return ANSWER_FIRST
    return ANSWER_SECOND


def make_instruction() -> str:
    """Return the exact stored LMOL instruction string."""

    return DEFAULT_INSTRUCTION


def _candidate_pairs(records: Sequence[SCUTRecord], theta: float, fold: int, split: str) -> dict[str, list[PairExample]]:
    candidates = {ANSWER_FIRST: [], ANSWER_SECOND: [], ANSWER_SIMILAR: []}
    for left_idx, image1 in enumerate(records):
        for right_idx, image2 in enumerate(records):
            if left_idx == right_idx:
                continue
            answer = label_from_scores(image1.score, image2.score, theta=theta)
            candidates[answer].append(
                PairExample(
                    image1=image1.image,
                    image2=image2.image,
                    image1_path=image1.image_path,
                    image2_path=image2.image_path,
                    score1=image1.score,
                    score2=image2.score,
                    instruction=make_instruction(),
                    answer=answer,
                    fold=fold,
                    split=split,
                )
            )
    return candidates


def _make_pair_example(image1: SCUTRecord, image2: SCUTRecord, theta: float, fold: int, split: str) -> PairExample:
    answer = label_from_scores(image1.score, image2.score, theta=theta)
    return PairExample(
        image1=image1.image,
        image2=image2.image,
        image1_path=image1.image_path,
        image2_path=image2.image_path,
        score1=image1.score,
        score2=image2.score,
        instruction=make_instruction(),
        answer=answer,
        fold=fold,
        split=split,
    )


def build_balanced_pairs(
    records: Sequence[SCUTRecord],
    pairs_per_label: int,
    seed: int,
    theta: float = 0.2,
    fold: int = 0,
    split: str = "train",
    require_full: bool = True,
) -> list[PairExample]:
    """Build deterministic balanced offline pairs from a training split."""

    if pairs_per_label < 0:
        raise ValueError(f"pairs_per_label must be non-negative, got {pairs_per_label}")

    rng = random.Random(seed)
    candidate_counts = {ANSWER_FIRST: 0, ANSWER_SECOND: 0, ANSWER_SIMILAR: 0}
    reservoirs: dict[str, list[tuple[SCUTRecord, SCUTRecord]]] = {
        ANSWER_FIRST: [],
        ANSWER_SECOND: [],
        ANSWER_SIMILAR: [],
    }

    for left_idx, image1 in enumerate(records):
        for right_idx, image2 in enumerate(records):
            if left_idx == right_idx:
                continue
            answer = label_from_scores(image1.score, image2.score, theta=theta)
            candidate_counts[answer] += 1
            reservoir = reservoirs[answer]
            if len(reservoir) < pairs_per_label:
                reservoir.append((image1, image2))
                continue
            replacement_index = rng.randrange(candidate_counts[answer])
            if replacement_index < pairs_per_label:
                reservoir[replacement_index] = (image1, image2)

    selected: list[PairExample] = []
    for answer in (ANSWER_FIRST, ANSWER_SECOND, ANSWER_SIMILAR):
        if candidate_counts[answer] < pairs_per_label and require_full:
            raise ValueError(
                f"Not enough {answer} candidates: requested {pairs_per_label}, available {candidate_counts[answer]}"
            )
        selected.extend(
            _make_pair_example(image1, image2, theta=theta, fold=fold, split=split)
            for image1, image2 in reservoirs[answer]
        )
    rng.shuffle(selected)
    return selected


def build_reference_pairs(
    targets: Sequence[SCUTRecord],
    references: Sequence[SCUTRecord],
    theta: float,
    fold: int,
    split: str = "test",
) -> list[PairExample]:
    """Build target-reference pairs for pairwise inference and score estimation."""

    rows: list[PairExample] = []
    for target in targets:
        for reference in references:
            if target.image == reference.image:
                continue
            rows.append(_make_pair_example(target, reference, theta=theta, fold=fold, split=split))
    return rows


def write_pairs_csv(pairs: Iterable[PairExample], path: str | Path) -> None:
    """Write offline LMOL pairs to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image1",
        "image2",
        "image1_path",
        "image2_path",
        "score1",
        "score2",
        "instruction",
        "answer",
        "fold",
        "split",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "image1": pair.image1,
                    "image2": pair.image2,
                    "image1_path": pair.image1_path,
                    "image2_path": pair.image2_path,
                    "score1": pair.score1,
                    "score2": pair.score2,
                    "instruction": pair.instruction,
                    "answer": pair.answer,
                    "fold": pair.fold,
                    "split": pair.split,
                }
            )


def read_pairs_csv(path: str | Path) -> list[PairExample]:
    """Read offline LMOL pairs from CSV."""

    rows: list[PairExample] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PairExample(
                    image1=row["image1"],
                    image2=row["image2"],
                    image1_path=Path(row["image1_path"]),
                    image2_path=Path(row["image2_path"]),
                    score1=float(row["score1"]),
                    score2=float(row["score2"]),
                    instruction=row["instruction"],
                    answer=row["answer"],
                    fold=int(row.get("fold", 0)),
                    split=row.get("split", "train"),
                )
            )
    return rows
