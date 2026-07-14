"""LLaVA-1.5 based LMOL model setup with 4-bit QLoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from configs import LMOLConfig

PAPER_IMAGE_SIZE = 336


def _missing_dependency_error(name: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(
        f"Missing runtime dependency {name!r}. Install the canonical environment with "
        "python -m pip install -r requirements.txt and rerun the command."
    )
    error.__cause__ = exc
    return error


def _find_projector_name(model: Any) -> str:
    names = [name for name, _ in model.named_modules()]
    for candidate in ("multi_modal_projector", "model.multi_modal_projector", "mm_projector"):
        if candidate in names:
            return candidate
    for name in names:
        if name.endswith("multi_modal_projector") or name.endswith("mm_projector"):
            return name
    raise AttributeError("Could not find the LLaVA multimodal projector module")


def _get_module(root: Any, dotted_name: str) -> Any:
    module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def _candidate_roots(model: Any) -> list[Any]:
    """Return possible roots for plain LLaVA and PEFT-wrapped models."""

    roots = [model]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        roots.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            roots.append(nested)
    return roots


def _get_projector(model: Any) -> Any:
    """Find the multimodal projector through plain or PEFT wrappers."""

    last_error: Exception | None = None
    for root in _candidate_roots(model):
        try:
            projector_name = _find_projector_name(root)
            return _get_module(root, projector_name)
        except Exception as exc:
            last_error = exc
    raise AttributeError("Could not resolve projector from model wrapper") from last_error


def _freeze_vision_tower(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if _is_vision_parameter(name):
            parameter.requires_grad = False


def _enable_projector_training(model: Any) -> None:
    projector = _get_projector(model)
    for parameter in projector.parameters():
        if parameter.dtype.is_floating_point or parameter.dtype.is_complex:
            parameter.requires_grad = True


def _is_vision_parameter(name: str) -> bool:
    return "vision_tower" in name or "vision_model" in name


def _is_projector_parameter(name: str) -> bool:
    return "multi_modal_projector" in name or "mm_projector" in name


def _is_language_module(name: str) -> bool:
    return "language_model" in name


def _matches_target_module(name: str, target_modules: tuple[str, ...]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in target_modules)


def _language_lora_target_modules(model: Any, target_modules: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve LoRA targets to exact LLM module names, excluding CLIP modules."""

    resolved = tuple(
        name
        for name, _module in model.named_modules()
        if _is_language_module(name) and _matches_target_module(name, target_modules)
    )
    if not resolved:
        raise ValueError(
            "No language-model LoRA target modules found. Check model.target_modules "
            f"against the loaded model architecture: {target_modules}"
        )
    return resolved


def _bnb_4bit_skip_modules() -> list[str]:
    """Keep non-LLM LLaVA components in floating point while quantizing the LLM."""

    return ["lm_head", "vision_tower", "multi_modal_projector", "mm_projector"]


def _validate_trainable_scope(model: Any) -> None:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    vision_trainable = [name for name in trainable_names if _is_vision_parameter(name)]
    if vision_trainable:
        sample = ", ".join(vision_trainable[:5])
        raise RuntimeError(f"Vision tower must remain frozen, but trainable parameters were found: {sample}")

    projector_names = [name for name, _parameter in model.named_parameters() if _is_projector_parameter(name)]
    if not projector_names:
        raise RuntimeError("No multimodal projector parameters found")
    frozen_projector = [
        name
        for name, parameter in model.named_parameters()
        if _is_projector_parameter(name) and not parameter.requires_grad
    ]
    if frozen_projector:
        sample = ", ".join(frozen_projector[:5])
        raise RuntimeError(f"All projector parameters must be trainable, but frozen parameters were found: {sample}")

    lora_names = [name for name in trainable_names if "lora_" in name]
    if not lora_names:
        raise RuntimeError("No trainable LoRA parameters found")

    unexpected = [
        name
        for name in trainable_names
        if not _is_projector_parameter(name) and not ("lora_" in name and _is_language_module(name))
    ]
    if unexpected:
        sample = ", ".join(unexpected[:5])
        raise RuntimeError(f"Unexpected trainable parameters outside projector and LLM LoRA: {sample}")


def _gradient_checkpointing_kwargs() -> dict[str, bool]:
    return {"use_reentrant": False}


def _square_image_size(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            height = int(value["height"])
            width = int(value["width"])
            if height != width:
                raise ValueError(f"Loaded processor expects non-square images: height={height}, width={width}")
            return height
        if "shortest_edge" in value:
            return int(value["shortest_edge"])
    return None


def _processor_image_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", processor)
    for attr in ("crop_size", "size"):
        resolved = _square_image_size(getattr(image_processor, attr, None))
        if resolved is not None:
            return resolved
    raise ValueError("Could not determine the loaded processor image size from image_processor.crop_size or image_processor.size")


def _validate_processor_image_size(processor: Any, expected_size: int = PAPER_IMAGE_SIZE) -> None:
    actual_size = _processor_image_size(processor)
    if actual_size != expected_size:
        raise ValueError(
            "Loaded processor image size does not match the supported LLaVA-1.5/CLIP ViT-L/14-336px size. "
            f"expected image size={expected_size}, processor image size={actual_size}."
        )


def _disable_training_cache(model: Any) -> None:
    """Disable generation KV cache on every nested config used during training."""

    seen: set[int] = set()

    def visit_config(config: Any) -> None:
        if config is None or id(config) in seen:
            return
        seen.add(id(config))
        if hasattr(config, "use_cache"):
            config.use_cache = False
        for attr in ("text_config", "language_config", "llm_config", "vision_config"):
            visit_config(getattr(config, attr, None))

    visit_config(getattr(model, "config", None))
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return
    for _name, module in named_modules():
        visit_config(getattr(module, "config", None))


def load_lmol_model(config: LMOLConfig, train: bool = True) -> tuple[Any, Any]:
    """Load LLaVA-1.5-7B, freeze CLIP, and attach QLoRA adapters."""

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise _missing_dependency_error("transformers", exc)
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise _missing_dependency_error("peft", exc)

    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=_bnb_4bit_skip_modules(),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(config.model.model_id)
    _validate_processor_image_size(processor)
    model = LlavaForConditionalGeneration.from_pretrained(
        config.model.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config.training.precision == "bf16" else torch.float16,
        device_map="auto",
    )
    _freeze_vision_tower(model)
    if train and config.training.gradient_checkpointing:
        _disable_training_cache(model)
    if train and config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs=_gradient_checkpointing_kwargs()
            if config.training.gradient_checkpointing
            else None,
        )
    elif train and config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=_gradient_checkpointing_kwargs())

    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(_language_lora_target_modules(model, config.model.target_modules)),
        modules_to_save=None,
    )
    if train:
        model = get_peft_model(model, lora_config)
        if config.training.gradient_checkpointing:
            _disable_training_cache(model)
        _enable_projector_training(model)
        _freeze_vision_tower(model)
        _validate_trainable_scope(model)
    else:
        model.eval()
    return model, processor


def optimizer_parameter_groups(model: Any, config: LMOLConfig) -> list[dict[str, Any]]:
    """Create separate AdamW groups for full projector parameters and LoRA parameters."""

    projector_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name and _is_language_module(name):
            lora_params.append(parameter)
        elif _is_projector_parameter(name):
            projector_params.append(parameter)
    if not projector_params:
        raise RuntimeError("No trainable projector parameters found")
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found")
    return [
        {"params": projector_params, "lr": config.training.projector_lr},
        {"params": lora_params, "lr": config.training.lora_lr},
    ]


def _projector_state_dict(model: Any) -> dict[str, torch.Tensor]:
    projector = _get_projector(model)
    return {key: value.detach().cpu() for key, value in projector.state_dict().items()}


def load_projector_state(model: Any, checkpoint_dir: str | Path) -> None:
    """Load the separately saved full projector state if present."""

    path = Path(checkpoint_dir) / "projector.pt"
    if not path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {path}")
    projector = _get_projector(model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    projector.load_state_dict(state)


def save_lmol_checkpoint(model: Any, processor: Any, output_dir: str | Path) -> None:
    """Save LoRA adapters, full projector parameters, and the processor."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    torch.save(_projector_state_dict(model), out_path / "projector.pt")
    processor.save_pretrained(out_path)
