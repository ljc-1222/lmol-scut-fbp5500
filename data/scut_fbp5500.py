"""SCUT-FBP5500 annotation loading, validation, and deterministic folds."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SCUTRecord:
    """One SCUT-FBP5500 image and its mean beauty score."""

    image: str
    score: float
    image_path: Path


def _parse_annotation_rows(annotation_path: Path) -> list[tuple[str, float]]:
    text = annotation_path.read_text(encoding="utf-8-sig").splitlines()
    rows: list[tuple[str, float]] = []
    for line_no, line in enumerate(text, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "," in stripped:
            parts = next(csv.reader([stripped]))
        else:
            parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid annotation row {line_no}: {line!r}")
        image = parts[0].strip()
        if image.lower() in {"image", "filename", "file", "name"}:
            continue
        try:
            score = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid score at row {line_no}: {parts[1]!r}") from exc
        rows.append((image, score))
    return rows


def load_annotations(root: str | Path, image_dir: str, annotation_file: str) -> list[SCUTRecord]:
    """Load SCUT annotations from a simple CSV or whitespace-delimited file."""

    root_path = Path(root)
    annotation_path = root_path / annotation_file
    image_root = root_path / image_dir
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    rows = _parse_annotation_rows(annotation_path)
    records = [SCUTRecord(image=name, score=score, image_path=image_root / name) for name, score in rows]
    return records


def validate_dataset(
    records: Sequence[SCUTRecord],
    expected_count: int,
    score_min: float = 1.0,
    score_max: float = 5.0,
    require_images: bool = True,
) -> None:
    """Validate record count, unique image names, score range, and image existence."""

    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, found {len(records)}")
    seen: set[str] = set()
    missing_images: list[Path] = []
    for record in records:
        if record.image in seen:
            raise ValueError(f"Duplicate image annotation: {record.image}")
        seen.add(record.image)
        if not (score_min <= record.score <= score_max):
            raise ValueError(f"Score for {record.image} is outside [{score_min}, {score_max}]: {record.score}")
        if require_images and not record.image_path.exists():
            missing_images.append(record.image_path)
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:5])
        raise FileNotFoundError(f"Missing {len(missing_images)} image files. First missing files: {preview}")


def deterministic_folds(records: Sequence[SCUTRecord], folds: int, seed: int) -> list[list[SCUTRecord]]:
    """Create deterministic folds when official fold files are unavailable."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    ordered = sorted(records, key=lambda item: item.image)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    split: list[list[SCUTRecord]] = [[] for _ in range(folds)]
    for index, record in enumerate(shuffled):
        split[index % folds].append(record)
    return [sorted(fold, key=lambda item: item.image) for fold in split]


def split_train_test(records: Sequence[SCUTRecord], folds: int, fold_index: int, seed: int) -> tuple[list[SCUTRecord], list[SCUTRecord]]:
    """Return train and test records for a deterministic fold index."""

    fold_list = deterministic_folds(records, folds=folds, seed=seed)
    if not 0 <= fold_index < folds:
        raise ValueError(f"fold_index must be in [0, {folds - 1}], got {fold_index}")
    test = fold_list[fold_index]
    train = [record for idx, fold in enumerate(fold_list) if idx != fold_index for record in fold]
    return sorted(train, key=lambda item: item.image), sorted(test, key=lambda item: item.image)


def write_image_split_csv(records: Iterable[SCUTRecord], path: str | Path) -> None:
    """Write an image split manifest."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "score", "image_path"])
        writer.writeheader()
        for record in records:
            writer.writerow({"image": record.image, "score": record.score, "image_path": record.image_path})
