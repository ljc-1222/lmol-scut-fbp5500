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
    def format_prompt_answer(prompt: str, answer: str) -> str:
        """Append the short LMOL answer using the separator expected by the prompt."""

        clean_prompt = prompt.rstrip()
        separator = " " if clean_prompt.endswith("ASSISTANT:") else "\n"
        return f"{clean_prompt}{separator}{answer}"

    @staticmethod
    def prompt_prefix_for_masking(prompt: str) -> str:
        """Return the exact prefix span that should be masked from loss."""

        clean_prompt = prompt.rstrip()
        if clean_prompt.endswith("ASSISTANT:"):
            return clean_prompt
        return f"{clean_prompt}\n"

    @staticmethod
    def answer_token_ids(tokenizer: Any) -> list[int]:
        """Return the single next-token IDs used for discriminative choice loss."""

        token_ids: list[int] = []
        for answer in VALID_ANSWERS:
            ids = tokenizer(answer, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"Could not tokenize answer: {answer!r}")
            token_ids.append(int(ids[0]))
        return token_ids

    @staticmethod
    def find_answer_token_position(input_ids: Any, answer_token_id: int) -> int:
        """Find the answer word token after processor image-token expansion."""

        matches = (input_ids == answer_token_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise ValueError(f"Answer token id {answer_token_id} was not found in processor input IDs")
        return int(matches[0].item())

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize a batch and mask labels so only the first answer token is supervised."""

        prompts = [self.render_prompt(self.processor, item["instruction"]) for item in examples]
        full_texts = [self.format_prompt_answer(prompt, item["answer"]) for prompt, item in zip(prompts, examples)]
        images = [image for item in examples for image in (item["image1"], item["image2"])]
        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        tokenizer = self.processor.tokenizer
        answer_token_ids = self.answer_token_ids(tokenizer)
        choice_positions: list[int] = []
        for row_idx, item in enumerate(examples):
            answer_token_id = answer_token_ids[ANSWER_TO_INDEX[item["answer"]]]
            answer_start = self.find_answer_token_position(batch["input_ids"][row_idx], answer_token_id)
            labels[row_idx, :answer_start] = -100
            labels[row_idx, answer_start + 1 :] = -100
            choice_positions.append(answer_start - 1)
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        import torch

        batch["choice_labels"] = torch.tensor([ANSWER_TO_INDEX[item["answer"]] for item in examples], dtype=torch.long)
        batch["choice_positions"] = torch.tensor(choice_positions, dtype=torch.long)
        batch["choice_token_ids"] = torch.tensor(answer_token_ids, dtype=torch.long)
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
