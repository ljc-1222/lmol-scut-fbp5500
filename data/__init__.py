"""SCUT-FBP5500 data loading and LMOL instruction pair construction."""

from .pairs import (
    ANSWER_FIRST,
    ANSWER_SECOND,
    ANSWER_SIMILAR,
    build_balanced_pairs,
    label_from_scores,
    make_instruction,
)
from .scut_fbp5500 import SCUTRecord, deterministic_folds, load_annotations, validate_dataset

__all__ = [
    "ANSWER_FIRST",
    "ANSWER_SECOND",
    "ANSWER_SIMILAR",
    "SCUTRecord",
    "build_balanced_pairs",
    "deterministic_folds",
    "label_from_scores",
    "load_annotations",
    "make_instruction",
    "validate_dataset",
]
