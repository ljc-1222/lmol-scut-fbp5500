#!/usr/bin/env python
"""Validate a manually downloaded SCUT-FBP5500 dataset."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from configs import load_config
from data.scut_fbp5500 import load_annotations, validate_dataset


MANUAL_STEPS = """Manual SCUT-FBP5500 setup:
1. Run: python scripts/download_scut_fbp5500.py --accept-terms
2. Or manually obtain SCUT-FBP5500 from the official release channel or authors' documented dataset page.
3. Place images under data/raw/SCUT-FBP5500/Images/.
4. Place the score annotation file at data/raw/SCUT-FBP5500/labels.txt with rows: <image_name> <score>.
5. Rerun: python scripts/prepare_scut_fbp5500.py --config configs/lmol_scut_fbp5500.yaml
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--print-manual-steps", action="store_true")
    args = parser.parse_args()
    if args.print_manual_steps:
        print(MANUAL_STEPS)
        return 0
    config = load_config(args.config)
    try:
        records = load_annotations(config.dataset.root, config.dataset.image_dir, config.dataset.annotation_file)
        validate_dataset(
            records,
            expected_count=config.dataset.expected_count,
            score_min=config.dataset.score_min,
            score_max=config.dataset.score_max,
            require_images=True,
        )
    except Exception as exc:
        print(f"SCUT-FBP5500 validation failed: {exc}")
        print(MANUAL_STEPS)
        return 1
    print(f"SCUT-FBP5500 validation passed: {len(records)} records in {Path(config.dataset.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
