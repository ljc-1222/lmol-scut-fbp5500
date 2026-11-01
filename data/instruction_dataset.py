"""PyTorch dataset and collator for LMOL first-answer-token training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .pairs import ANSWER_TO_INDEX, CANONICAL_ANSWERS, PairExample

IMAGE_PLACEHOLDERS = "<image1><image2>"
VALID_ANSWERS = CANONICAL_ANSWERS


class LMOLInstructionDataset:
    """Dataset returning two separate face images, an instruction, and an answer."""

    def __init__(self, pairs: list[PairExample]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        image1 = Image.open(pair.image1_path).convert("RGB")
        image2 = Image.open(pair.image2_path).convert("RGB")
        return {
            "image1": image1,
            "image2": image2,
            "instruction": pair.instruction,
            "answer": pair.answer,
        }


class LMOLDataCollator:
    """Collate LMOL examples with first-answer-token supervision."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    @staticmethod
    def instruction_text(instruction: str) -> str:
        """Return the natural-language question without stored image placeholders."""

        return instruction.replace(IMAGE_PLACEHOLDERS, "").strip()

    @staticmethod
    def to_llava_prompt(instruction: str) -> str:
        """Fallback prompt using two raw LLaVA image tokens."""

        return instruction.replace(IMAGE_PLACEHOLDERS, "<image><image>")

    @staticmethod
    def render_prompt(processor: Any, instruction: str) -> str:
        """Render the prompt with the processor chat template when available."""

        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if apply_chat_template is None:
            return LMOLDataCollator.to_llava_prompt(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": LMOLDataCollator.instruction_text(instruction)},
                ],
            }
        ]
        return apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @staticmethod
    def answer_token_ids(tokenizer: Any) -> list[int]:
        """Return distinct vocabulary IDs for the three allowed first-token choices."""

        token_ids: list[int] = []
        for answer in VALID_ANSWERS:
            ids = tokenizer(answer, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"Could not tokenize answer: {answer!r}")
            token_ids.append(int(ids[0]))
        if len(set(token_ids)) != len(VALID_ANSWERS):
            raise ValueError("First-token classes must map to three distinct token IDs")
        return token_ids

    @staticmethod
    def prompt_choice_positions(batch: dict[str, Any]) -> Any:
        """Find the final non-padding prompt token for each row."""

        import torch

        ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            if ids.shape[0] != 1:
                raise ValueError("An attention_mask is required for batched prompt scoring")
            return torch.tensor([ids.shape[1] - 1], dtype=torch.long, device=ids.device)
        if not bool(attention_mask.bool().any(dim=1).all()):
            raise ValueError("Every prompt must contain a non-padding token")
        columns = torch.arange(ids.shape[1], device=attention_mask.device).unsqueeze(0)
        return (attention_mask.long() * columns).max(dim=1).values.long()

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Encode prompts without teacher-forcing the answer into the model input."""

        import torch

        if not examples:
            raise ValueError("Cannot collate an empty LMOL batch")
        prompts = [self.render_prompt(self.processor, item["instruction"]).rstrip() for item in examples]
        images = [image for item in examples for image in (item["image1"], item["image2"])]
        batch = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        batch["choice_positions"] = self.prompt_choice_positions(batch)
        batch["choice_labels"] = torch.tensor(
            [ANSWER_TO_INDEX[item["answer"]] for item in examples], dtype=torch.long
        )
        batch["choice_token_ids"] = torch.tensor(
            self.answer_token_ids(self.processor.tokenizer), dtype=torch.long
        )
        return batch


def assert_pair_images_exist(pairs: list[PairExample]) -> None:
    """Validate that every pair points to two separate image files."""

    missing: list[Path] = []
    same_path: list[tuple[Path, Path]] = []
    unique_paths: set[Path] = set()
    for pair in pairs:
        if pair.image1_path == pair.image2_path:
            same_path.append((pair.image1_path, pair.image2_path))
        unique_paths.add(pair.image1_path)
        unique_paths.add(pair.image2_path)
    for path in unique_paths:
        if not path.exists():
            missing.append(path)
    if same_path:
        raise ValueError("LMOL pairs must contain two separately encoded images, not one concatenated image")
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing pair image files: {preview}")
