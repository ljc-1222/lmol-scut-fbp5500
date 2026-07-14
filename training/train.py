"""Minimal LMOL first-answer-token training loop."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Any

from configs import LMOLConfig

from data.instruction_dataset import LMOLDataCollator, LMOLInstructionDataset, assert_pair_images_exist
from data.pairs import read_pairs_csv
from evaluation.inference import evaluate_pairwise_likelihood
from model.lmol import load_lmol_model, optimizer_parameter_groups, save_lmol_checkpoint


def _json_log(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _wandb_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_wandb_safe(item) for item in value]
    if isinstance(value, list):
        return [_wandb_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _wandb_safe(item) for key, item in value.items()}
    return value


def _wandb_metric_key(key: str) -> str:
    mapping = {
        "loss": "train/loss",
        "train_loss": "train/loss",
        "train_scaled_loss": "train/scaled_loss",
        "lr_projector": "lr/projector",
        "lr_lora": "lr/lora",
        "samples_per_sec": "perf/samples_per_sec",
        "step_time_sec": "perf/step_time_sec",
        "eval_loss": "eval/loss",
        "eval_acc": "eval/acc",
        "eval_count": "eval/count",
        "eval_label_counts": "eval/label_counts",
        "eval_prediction_counts": "eval/prediction_counts",
        "checkpoint": "checkpoint/path",
    }
    return mapping.get(key, key)


def _flatten_wandb_metrics(payload: dict[str, object]) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in payload.items():
        if key == "step":
            continue
        metric_key = _wandb_metric_key(key)
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                cleaned = str(subkey).strip().rstrip(".")
                flattened[f"{metric_key}/{cleaned}"] = subvalue
        else:
            flattened[metric_key] = value
    return flattened


def _init_wandb_run(
    config: LMOLConfig,
    pairs_csv: str | Path,
    eval_pairs_csv: str | Path | None,
    total_steps: int,
    run_limit: int | None,
    wandb_enabled: bool | None,
    wandb_mode: str | None,
    wandb_project: str | None,
    wandb_name: str | None,
) -> Any:
    enabled = config.wandb.enabled if wandb_enabled is None else wandb_enabled
    mode = wandb_mode or config.wandb.mode
    if not enabled or mode == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        error = RuntimeError(
            "wandb logging is enabled but the wandb package is unavailable. "
            "Install dependencies with python -m pip install -r requirements.txt, "
            "or rerun training with --no-wandb."
        )
        error.__cause__ = exc
        raise error

    run_name = wandb_name or config.wandb.name or f"{config.experiment_name}-fold_{config.dataset.fold_index}"
    wandb_config = _wandb_safe(asdict(config))
    wandb_config["runtime"] = {
        "pairs_csv": str(pairs_csv),
        "eval_pairs_csv": None if eval_pairs_csv is None else str(eval_pairs_csv),
        "total_steps": total_steps,
        "run_limit": run_limit,
        "effective_batch_size": config.training.per_device_batch_size
        * config.training.gradient_accumulation_steps,
    }
    return wandb.init(
        project=wandb_project or config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        group=config.wandb.group,
        tags=list(config.wandb.tags),
        mode=mode,
        config=wandb_config,
    )


def _wandb_log(run: Any, payload: dict[str, object]) -> None:
    if run is None:
        return
    step = payload.get("step")
    run.log(_flatten_wandb_metrics(payload), step=int(step) if isinstance(step, int) else None)


def _set_random_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accumulation_group(batch_index: int, batch_count: int, accumulation_steps: int) -> tuple[int, bool]:
    group_start = (batch_index - 1) // accumulation_steps * accumulation_steps
    group_size = min(accumulation_steps, batch_count - group_start)
    return group_size, batch_index == group_start + group_size


def train_lmol(
    config: LMOLConfig,
    pairs_csv: str | Path,
    max_steps: int | None = None,
    eval_pairs_csv: str | Path | None = None,
    stop_after_steps: int | None = None,
    wandb_enabled: bool | None = None,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> Path:
    """Train LMOL for one fold using first-answer-token classification."""

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive when provided, got {max_steps}")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError(f"stop_after_steps must be positive when provided, got {stop_after_steps}")

    _set_random_seed(config.seed)
    pairs = read_pairs_csv(pairs_csv)
    assert_pair_images_exist(pairs)
    eval_pairs = None
    if eval_pairs_csv is not None and config.evaluation.eval_steps > 0:
        eval_pairs = read_pairs_csv(eval_pairs_csv)
        assert_pair_images_exist(eval_pairs)
    dataset = LMOLInstructionDataset(pairs)
    loader_batches = max(
        1,
        (len(dataset) + config.training.per_device_batch_size - 1) // config.training.per_device_batch_size,
    )
    configured_max = config.training.max_steps if max_steps is None else max_steps
    steps_per_epoch = (
        loader_batches + config.training.gradient_accumulation_steps - 1
    ) // config.training.gradient_accumulation_steps
    total_steps = configured_max or max(1, steps_per_epoch * config.training.num_epochs)
    run_limit = stop_after_steps if stop_after_steps is not None else configured_max

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb_run(
        config,
        pairs_csv=pairs_csv,
        eval_pairs_csv=eval_pairs_csv,
        total_steps=total_steps,
        run_limit=run_limit,
        wandb_enabled=wandb_enabled,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    exit_code = 0
    try:
        model, processor = load_lmol_model(config, train=True)
        collator = LMOLDataCollator(processor)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed)
        loader = DataLoader(
            dataset,
            batch_size=config.training.per_device_batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=loader_generator,
        )
        optimizer = AdamW(
            optimizer_parameter_groups(model, config),
            weight_decay=config.training.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
        global_step = 0
        pending_loss_sum = 0.0
        pending_loss_count = 0
        step_start_time = time.perf_counter()
        model.train()
        for _epoch in range(config.training.num_epochs):
            for batch_index, batch in enumerate(loader, start=1):
                batch = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
                choice_labels = batch.pop("choice_labels")
                choice_positions = batch.pop("choice_positions")
                choice_token_ids = batch.pop("choice_token_ids")
                batch.pop("labels")
                outputs = model(**batch)
                row_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
                next_token_logits = outputs.logits[row_indices, choice_positions, :].float()
                choice_logits = next_token_logits.index_select(dim=-1, index=choice_token_ids)
                raw_loss = F.cross_entropy(choice_logits, choice_labels)
                accumulation_size, should_step = _accumulation_group(
                    batch_index,
                    len(loader),
                    config.training.gradient_accumulation_steps,
                )
                scaled_loss = raw_loss / accumulation_size
                pending_loss_sum += float(raw_loss.detach().cpu())
                pending_loss_count += 1
                scaled_loss.backward()
                if should_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    train_loss = pending_loss_sum / max(1, pending_loss_count)
                    pending_loss_sum = 0.0
                    pending_loss_count = 0
                    step_time_sec = time.perf_counter() - step_start_time
                    effective_batch_size = config.training.per_device_batch_size * config.training.gradient_accumulation_steps
                    samples_per_sec = effective_batch_size / step_time_sec if step_time_sec > 0 else 0.0
                    step_start_time = time.perf_counter()
                    if global_step % config.training.log_steps == 0:
                        train_payload = {
                            "step": global_step,
                            "loss": train_loss,
                            "train_loss": train_loss,
                            "train_scaled_loss": train_loss / config.training.gradient_accumulation_steps,
                            "lr_projector": optimizer.param_groups[0]["lr"],
                            "lr_lora": optimizer.param_groups[1]["lr"],
                            "samples_per_sec": samples_per_sec,
                            "step_time_sec": step_time_sec,
                        }
                        _json_log(train_payload)
                        _wandb_log(wandb_run, train_payload)
                    if eval_pairs is not None and global_step % config.evaluation.eval_steps == 0:
                        metrics = evaluate_pairwise_likelihood(
                            model,
                            processor,
                            eval_pairs,
                            max_pairs=config.evaluation.eval_max_pairs,
                            batch_size=config.evaluation.batch_size,
                        )
                        eval_payload = {
                            "step": global_step,
                            "eval_loss": metrics.loss,
                            "eval_acc": metrics.accuracy,
                            "eval_count": metrics.count,
                            "eval_label_counts": metrics.label_counts,
                            "eval_prediction_counts": metrics.prediction_counts,
                        }
                        _json_log(eval_payload)
                        _wandb_log(wandb_run, eval_payload)
                    if global_step % config.training.save_steps == 0:
                        checkpoint = output_dir / f"step_{global_step}"
                        save_lmol_checkpoint(model, processor, checkpoint)
                        _wandb_log(wandb_run, {"step": global_step, "checkpoint": str(checkpoint)})
                    if run_limit is not None and global_step >= run_limit:
                        checkpoint = output_dir / "last"
                        save_lmol_checkpoint(model, processor, checkpoint)
                        _wandb_log(wandb_run, {"step": global_step, "checkpoint": str(checkpoint)})
                        return checkpoint
        checkpoint = output_dir / "last"
        save_lmol_checkpoint(model, processor, checkpoint)
        _wandb_log(wandb_run, {"step": global_step, "checkpoint": str(checkpoint)})
        return checkpoint
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)
