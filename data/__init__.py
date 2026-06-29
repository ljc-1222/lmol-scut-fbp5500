"""SCUT-FBP5500 data loading and LMOL instruction pair construction."""

from .pairs import (
    ANSWER_FIRST,
    ANSWER_SECOND,
    ANSWER_SIMILAR,
    ANSWER_TO_INDEX,
    CANONICAL_ANSWERS,
    build_balanced_pairs,
    iter_reference_pairs,
    label_from_scores,
    make_instruction,
)
from .references import select_score_bin_references, assert_no_reference_leakage
from .scut_fbp5500 import SCUTRecord, deterministic_folds, load_annotations, validate_dataset

__all__ = [
    "ANSWER_FIRST",
    "ANSWER_SECOND",
    "ANSWER_SIMILAR",
    "ANSWER_TO_INDEX",
    "CANONICAL_ANSWERS",
    "SCUTRecord",
    "assert_no_reference_leakage",
    "build_balanced_pairs",
    "deterministic_folds",
    "iter_reference_pairs",
    "label_from_scores",
    "load_annotations",
    "make_instruction",
    "select_score_bin_references",
    "validate_dataset",
]
