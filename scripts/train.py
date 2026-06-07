#!/usr/bin/env python
"""Train LMOL on one SCUT-FBP5500 fold."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from configs import config_for_fold, load_config
from data.instruction_dataset import assert_pair_images_exist
from data.pairs import read_pairs_csv
from training import train_lmol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument("--fold-index", type=int, help="Fold index to train. Defaults to dataset.fold_index from the config.")
    parser.add_argument("--pairs", help="Path to train_pairs.csv. Defaults to the configured fold output.")
    parser.add_argument("--eval-pairs", help="Path to eval pairs CSV. Defaults to the configured fold test-reference pairs.")
    parser.add_argument("--no-eval", action="store_true", help="Disable training-time pairwise likelihood eval.")
    parser.add_argument("--max-steps", type=int, help="Override the configured training horizon and scheduler length.")
    parser.add_argument("--stop-after-steps", type=int, help="Stop early without changing the configured scheduler horizon; useful for diagnostics.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging for this run.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="Override the configured wandb mode.")
    parser.add_argument("--wandb-project", help="Override the configured wandb project.")
    parser.add_argument("--wandb-name", help="Override the configured wandb run name.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, pair CSV, and image availability without loading model weights.")
    args = parser.parse_args()
    config = config_for_fold(load_config(args.config), args.fold_index)
    pairs_csv = Path(args.pairs) if args.pairs else config.pairs.output_dir / f"fold_{config.dataset.fold_index}" / "train_pairs.csv"
    eval_pairs_csv = None
    if not args.no_eval and config.evaluation.eval_steps > 0:
        eval_pairs_csv = Path(args.eval_pairs) if args.eval_pairs else config.pairs.output_dir / f"fold_{config.dataset.fold_index}" / "test_reference_pairs.csv"
    pairs = read_pairs_csv(pairs_csv)
    assert_pair_images_exist(pairs)
    print(f"Using fold {config.dataset.fold_index}; checkpoints will be written under {config.training.output_dir}")
    print(f"Loaded {len(pairs)} LMOL instruction pairs from {pairs_csv}")
    if eval_pairs_csv is not None:
        eval_pairs = read_pairs_csv(eval_pairs_csv)
        assert_pair_images_exist(eval_pairs)
        print(f"Loaded {len(eval_pairs)} LMOL eval pairs from {eval_pairs_csv}; evaluating {min(config.evaluation.eval_max_pairs, len(eval_pairs))} every {config.evaluation.eval_steps} steps")
    if args.dry_run:
        print("Dry run passed. Model loading and optimization were skipped by request.")
        return 0
    checkpoint = train_lmol(
        config,
        pairs_csv=pairs_csv,
        max_steps=args.max_steps,
        eval_pairs_csv=eval_pairs_csv,
        stop_after_steps=args.stop_after_steps,
        wandb_enabled=False if args.no_wandb else None,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    print(f"Saved checkpoint to {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
