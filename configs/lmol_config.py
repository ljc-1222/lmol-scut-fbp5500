"""Typed configuration loader for the LMOL SCUT-FBP5500 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from data.pairs import DEFAULT_INSTRUCTION


@dataclass(frozen=True)
class DatasetConfig:
    """SCUT-FBP5500 data paths and split settings."""

    root: Path
    image_dir: str
    annotation_file: str
    expected_count: int
    score_min: float
    score_max: float
    folds: int
    fold_index: int
    split_seed: int


@dataclass(frozen=True)
class PairConfig:
    """Offline pair construction settings."""

    theta: float
    train_pairs_per_label: int
    instruction: str
    output_dir: Path
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    """Model and QLoRA settings."""

    model_id: str
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters for one LMOL fold."""

    output_dir: Path
    num_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    projector_lr: float
    lora_lr: float
    weight_decay: float
    max_steps: int | None
    precision: str
    gradient_checkpointing: bool
    save_steps: int
    log_steps: int


@dataclass(frozen=True)
class WandBConfig:
    """Weights & Biases experiment tracking settings."""

    enabled: bool
    project: str
    entity: str | None
    name: str | None
    group: str | None
    mode: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Pairwise inference and score-estimation settings."""

    output_dir: Path
    batch_size: int
    reference_bin_width: float
    reference_max_per_bin: int | None
    reference_seed: int
    score_min: float
    score_max: float
    bt_delta: float
    bt_k: float
    eval_steps: int
    eval_max_pairs: int


@dataclass(frozen=True)
class LMOLConfig:
    """Complete LMOL configuration."""

    experiment_name: str
    seed: int
    dataset: DatasetConfig
    pairs: PairConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandBConfig
    evaluation: EvaluationConfig


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def _tuple_str(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values)


def _positive_float(name: str, value: Any) -> float:
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive when set, got {resolved}")
    return resolved


def _validate_score_bounds(score_min: Any, score_max: Any) -> tuple[float, float]:
    resolved_min = float(score_min)
    resolved_max = float(score_max)
    if resolved_min >= resolved_max:
        raise ValueError(f"evaluation.score_min must be less than evaluation.score_max, got {resolved_min} >= {resolved_max}")
    return resolved_min, resolved_max


def _reject_legacy_evaluation_keys(evaluation: Mapping[str, Any]) -> None:
    legacy_keys = sorted(set(evaluation) & {"reference_count", "bt_grid_size"})
    if legacy_keys:
        raise ValueError(
            "Legacy evaluation config keys are not supported by the UOL default path: "
            f"{legacy_keys}. Use reference_max_per_bin for compute-saving reference caps; "
            "UOL BT estimation uses reference-score support, not a dense grid."
        )


def validate_fold_index(config: LMOLConfig, fold_index: int | None = None) -> int:
    """Return a valid fold index, using the config default when not overridden."""

    resolved = config.dataset.fold_index if fold_index is None else int(fold_index)
    if not 0 <= resolved < config.dataset.folds:
        raise ValueError(f"fold-index must be in [0, {config.dataset.folds - 1}], got {resolved}")
    return resolved


def fold_scoped_dir(base_dir: str | Path, fold_index: int) -> Path:
    """Resolve an output directory for one fold."""

    path = _path(base_dir)
    suffix = f"fold_{fold_index}"
    text = str(path)
    if "{fold}" in text or "{fold_index}" in text:
        return Path(text.format(fold=fold_index, fold_index=fold_index)).expanduser()
    if path.name == suffix:
        return path
    return path / suffix


def config_for_fold(config: LMOLConfig, fold_index: int | None = None) -> LMOLConfig:
    """Return a config whose selected fold and output dirs are fold-specific."""

    resolved = validate_fold_index(config, fold_index)
    return replace(
        config,
        dataset=replace(config.dataset, fold_index=resolved),
        training=replace(config.training, output_dir=fold_scoped_dir(config.training.output_dir, resolved)),
        evaluation=replace(config.evaluation, output_dir=fold_scoped_dir(config.evaluation.output_dir, resolved)),
    )


def load_config(path: str | Path) -> LMOLConfig:
    """Load a YAML config file into typed dataclasses."""

    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")

    dataset = _required(raw, "dataset")
    pairs = _required(raw, "pairs")
    model = _required(raw, "model")
    training = _required(raw, "training")
    wandb = raw.get("wandb", {}) or {}
    evaluation = _required(raw, "evaluation")
    _reject_legacy_evaluation_keys(evaluation)
    evaluation_score_min, evaluation_score_max = _validate_score_bounds(
        evaluation.get("score_min", dataset.get("score_min", 1.0)),
        evaluation.get("score_max", dataset.get("score_max", 5.0)),
    )

    return LMOLConfig(
        experiment_name=str(_required(raw, "experiment_name")),
        seed=int(raw.get("seed", 42)),
        dataset=DatasetConfig(
            root=_path(_required(dataset, "root")),
            image_dir=str(_required(dataset, "image_dir")),
            annotation_file=str(_required(dataset, "annotation_file")),
            expected_count=int(dataset.get("expected_count", 5500)),
            score_min=float(dataset.get("score_min", 1.0)),
            score_max=float(dataset.get("score_max", 5.0)),
            folds=int(dataset.get("folds", 5)),
            fold_index=int(dataset.get("fold_index", 0)),
            split_seed=int(dataset.get("split_seed", raw.get("seed", 42))),
        ),
        pairs=PairConfig(
            theta=float(pairs.get("theta", 0.2)),
            train_pairs_per_label=int(pairs.get("train_pairs_per_label", 30000)),
            instruction=str(pairs.get("instruction", DEFAULT_INSTRUCTION)),
            output_dir=_path(pairs.get("output_dir", "artifacts/pairs")),
            seed=int(pairs.get("seed", raw.get("seed", 42))),
        ),
        model=ModelConfig(
            model_id=str(model.get("model_id", "llava-hf/llava-1.5-7b-hf")),
            load_in_4bit=bool(model.get("load_in_4bit", True)),
            lora_r=int(model.get("lora_r", 8)),
            lora_alpha=int(model.get("lora_alpha", 32)),
            lora_dropout=float(model.get("lora_dropout", 0.05)),
            target_modules=_tuple_str(model.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        ),
        training=TrainingConfig(
            output_dir=_path(training.get("output_dir", "outputs/lmol_scut")),
            num_epochs=int(training.get("num_epochs", 1)),
            per_device_batch_size=int(training.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
            projector_lr=float(training.get("projector_lr", 2e-5)),
            lora_lr=float(training.get("lora_lr", 2e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_steps=None if training.get("max_steps") is None else int(training["max_steps"]),
            precision=str(training.get("precision", "bf16")),
            gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
            save_steps=int(training.get("save_steps", 500)),
            log_steps=int(training.get("log_steps", 10)),
        ),
        wandb=WandBConfig(
            enabled=bool(wandb.get("enabled", True)),
            project=str(wandb.get("project", "lmol-fbp")),
            entity=None if wandb.get("entity") is None else str(wandb["entity"]),
            name=None if wandb.get("name") is None else str(wandb["name"]),
            group=None if wandb.get("group") is None else str(wandb["group"]),
            mode=str(wandb.get("mode", "online")),
            tags=_tuple_str(wandb.get("tags", ["lmol", "scut-fbp5500"])),
        ),
        evaluation=EvaluationConfig(
            output_dir=_path(evaluation.get("output_dir", "outputs/evaluation")),
            batch_size=int(evaluation.get("batch_size", 2)),
            reference_bin_width=_positive_float("evaluation.reference_bin_width", evaluation.get("reference_bin_width", 0.1)),
            reference_max_per_bin=_optional_positive_int("evaluation.reference_max_per_bin", evaluation.get("reference_max_per_bin", 10)),
            reference_seed=int(evaluation.get("reference_seed", raw.get("seed", 42))),
            score_min=evaluation_score_min,
            score_max=evaluation_score_max,
            bt_delta=_positive_float("evaluation.bt_delta", evaluation.get("bt_delta", 2.0)),
            bt_k=_positive_float("evaluation.bt_k", evaluation.get("bt_k", 10.0)),
            eval_steps=int(evaluation.get("eval_steps", 500)),
            eval_max_pairs=int(evaluation.get("eval_max_pairs", 256)),
        ),
    )
