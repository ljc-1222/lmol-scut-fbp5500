"""Unit tests for the clean LMOL-on-SCUT implementation."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from configs import config_for_fold, fold_scoped_dir, load_config
from data.instruction_dataset import assert_pair_images_exist
from data.pairs import (
    ANSWER_FIRST,
    ANSWER_SECOND,
    ANSWER_SIMILAR,
    PairExample,
    build_balanced_pairs,
    label_from_scores,
    make_instruction,
    read_pairs_csv,
)
from data.scut_fbp5500 import SCUTRecord, load_annotations, validate_dataset
from evaluation.bradley_terry import bradley_terry_scores
from evaluation.inference import evaluate_pairwise_likelihood, predict_answer_by_likelihood, write_predictions_csv
from evaluation.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error
from model.lmol import _bnb_4bit_skip_modules, _language_lora_target_modules, _validate_trainable_scope
from scripts.build_pairs import build_pairs_for_folds
from scripts.download_scut_fbp5500 import delete_archive_file, extract_dataset_archive


class LMOLCoreTests(unittest.TestCase):
    """CPU-safe tests required by plan.md."""

    def test_lora_targets_are_limited_to_language_model_modules(self) -> None:
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_tower = torch.nn.Module()
                self.vision_tower.vision_model = torch.nn.Module()
                self.vision_tower.vision_model.q_proj = torch.nn.Linear(2, 2)
                self.language_model = torch.nn.Module()
                self.language_model.q_proj = torch.nn.Linear(2, 2)
                self.language_model.mlp = torch.nn.Module()
                self.language_model.mlp.down_proj = torch.nn.Linear(2, 2)
                self.multi_modal_projector = torch.nn.Linear(2, 2)

        targets = _language_lora_target_modules(FakeModel(), ("q_proj", "down_proj"))

        self.assertEqual(set(targets), {"language_model.q_proj", "language_model.mlp.down_proj"})
        self.assertFalse(any("vision" in target for target in targets))
        self.assertIn("vision_tower", _bnb_4bit_skip_modules())
        self.assertIn("multi_modal_projector", _bnb_4bit_skip_modules())

    def test_trainable_scope_rejects_vision_lora_and_frozen_projector(self) -> None:
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_tower = torch.nn.Module()
                self.vision_tower.lora_A = torch.nn.Parameter(torch.ones(2))
                self.language_model = torch.nn.Module()
                self.language_model.lora_A = torch.nn.Parameter(torch.ones(2))
                self.multi_modal_projector = torch.nn.Linear(2, 2)

        model = FakeModel()
        with self.assertRaisesRegex(RuntimeError, "Vision tower must remain frozen"):
            _validate_trainable_scope(model)

        model.vision_tower.lora_A.requires_grad = False
        model.multi_modal_projector.weight.requires_grad = False
        with self.assertRaisesRegex(RuntimeError, "All projector parameters must be trainable"):
            _validate_trainable_scope(model)

        model.multi_modal_projector.weight.requires_grad = True
        _validate_trainable_scope(model)

    def test_theta_label_construction(self) -> None:
        self.assertEqual(label_from_scores(4.0, 4.0, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.8, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.81, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.79, theta=0.2), ANSWER_FIRST)
        self.assertEqual(label_from_scores(2.0, 2.21, theta=0.2), ANSWER_SECOND)

    def test_instruction_and_answer_formatting(self) -> None:
        self.assertEqual(make_instruction(), "<image1><image2>\nWhich face looks more attractive?")
        self.assertEqual(ANSWER_FIRST, "First.")
        self.assertEqual(ANSWER_SECOND, "Second.")
        self.assertEqual(ANSWER_SIMILAR, "Similar.")

    def test_pair_image_validation_checks_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image1_path = root / "a.jpg"
            image2_path = root / "b.jpg"
            Image.new("RGB", (8, 8)).save(image1_path)
            Image.new("RGB", (8, 8)).save(image2_path)
            pairs = [
                PairExample(
                    image1="a.jpg",
                    image2="b.jpg",
                    image1_path=image1_path,
                    image2_path=image2_path,
                    score1=1.0,
                    score2=2.0,
                    instruction=make_instruction(),
                    answer=ANSWER_SECOND,
                    fold=0,
                    split="train",
                )
                for _ in range(3)
            ]

            assert_pair_images_exist(pairs)

            bad_pair = PairExample(
                image1="a.jpg",
                image2="a.jpg",
                image1_path=image1_path,
                image2_path=image1_path,
                score1=1.0,
                score2=1.0,
                instruction=make_instruction(),
                answer=ANSWER_SIMILAR,
                fold=0,
                split="train",
            )
            with self.assertRaises(ValueError):
                assert_pair_images_exist([bad_pair])

    def test_balanced_offline_sampling(self) -> None:
        records = [
            SCUTRecord("a.jpg", 1.0, Path("a.jpg")),
            SCUTRecord("b.jpg", 1.1, Path("b.jpg")),
            SCUTRecord("c.jpg", 3.0, Path("c.jpg")),
            SCUTRecord("d.jpg", 4.5, Path("d.jpg")),
        ]
        pairs = build_balanced_pairs(records, pairs_per_label=2, seed=3, theta=0.2)
        counts = {ANSWER_FIRST: 0, ANSWER_SECOND: 0, ANSWER_SIMILAR: 0}
        for pair in pairs:
            counts[pair.answer] += 1
            self.assertEqual(pair.instruction, make_instruction())
        self.assertEqual(counts, {ANSWER_FIRST: 2, ANSWER_SECOND: 2, ANSWER_SIMILAR: 2})

    def test_build_pairs_generates_all_configured_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            image_dir = dataset_root / "Images"
            image_dir.mkdir(parents=True)
            scores = [1.2, 1.35, 2.8, 3.0, 4.2, 4.55]
            label_lines: list[str] = []
            for idx, score in enumerate(scores, start=1):
                image_name = f"sample_{idx:03d}.jpg"
                Image.new("RGB", (8, 8)).save(image_dir / image_name)
                label_lines.append(f"{image_name} {score:.2f}\n")
            (dataset_root / "labels.txt").write_text("".join(label_lines), encoding="utf-8")
            output_dir = root / "pairs"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
experiment_name: all_fold_test
seed: 7

dataset:
  root: {dataset_root}
  image_dir: Images
  annotation_file: labels.txt
  expected_count: 6
  score_min: 1.0
  score_max: 5.0
  folds: 3
  fold_index: 0
  split_seed: 7

pairs:
  theta: 0.2
  train_pairs_per_label: 2
  instruction: "<image1><image2>\\nWhich face looks more attractive?"
  answers: ["First.", "Second.", "Similar."]
  output_dir: {output_dir}
  seed: 7

model:
  model_id: llava-hf/llava-1.5-7b-hf
  image_size: 336
  load_in_4bit: true
  lora_r: 8
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj"]

training:
  output_dir: {root / "outputs"}
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 1
  projector_lr: 0.00002
  lora_lr: 0.0002
  weight_decay: 0.0
  max_steps:
  precision: bf16
  gradient_checkpointing: true
  save_steps: 1
  log_steps: 1
  objective: choice_token

evaluation:
  output_dir: {root / "eval"}
  batch_size: 1
  max_new_tokens: 4
  reference_count: 2
  bt_grid_size: 11
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            records = load_annotations(dataset_root, "Images", "labels.txt")

            results = build_pairs_for_folds(config, records, range(config.dataset.folds))

            self.assertEqual([result.fold_index for result in results], [0, 1, 2])
            for fold in range(3):
                fold_dir = output_dir / f"fold_{fold}"
                self.assertTrue((fold_dir / "train_images.csv").exists())
                self.assertTrue((fold_dir / "test_images.csv").exists())
                self.assertTrue((fold_dir / "test_reference_pairs.csv").exists())
                train_pairs = read_pairs_csv(fold_dir / "train_pairs.csv")
                self.assertEqual(len(train_pairs), 6)
                self.assertEqual({pair.fold for pair in train_pairs}, {fold})
                self.assertEqual(
                    Counter(pair.answer for pair in train_pairs),
                    Counter({ANSWER_FIRST: 2, ANSWER_SECOND: 2, ANSWER_SIMILAR: 2}),
                )

    def test_likelihood_inference_selects_highest_probability_answer(self) -> None:
        import torch

        class FakeTokenizer:
            pad_token_id = 0

            def __init__(self) -> None:
                self.vocab = {"<pad>": 0, "<bos>": 1, ANSWER_FIRST: 10, ANSWER_SECOND: 11, ANSWER_SIMILAR: 12}

            def __call__(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
                if not add_special_tokens and text in self.vocab:
                    return SimpleNamespace(input_ids=[self.vocab[text]])
                ids = [self.vocab["<bos>"]] if add_special_tokens else []
                for char in text:
                    if char not in self.vocab:
                        self.vocab[char] = len(self.vocab)
                    ids.append(self.vocab[char])
                return SimpleNamespace(input_ids=ids)

        class FakeProcessor:
            def __init__(self) -> None:
                self.tokenizer = FakeTokenizer()

            def __call__(self, text: list[str], images: list[object], padding: bool, return_tensors: str) -> dict[str, torch.Tensor]:
                del images, padding, return_tensors
                rows = [self.tokenizer(item, add_special_tokens=True).input_ids for item in text]
                width = max(len(row) for row in rows)
                padded = [row + [self.tokenizer.pad_token_id] * (width - len(row)) for row in rows]
                return {"input_ids": torch.tensor(padded, dtype=torch.long)}

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 16)
                logits[:, -1, 10] = 1.0
                logits[:, -1, 11] = 5.0
                logits[:, -1, 12] = 2.0
                return SimpleNamespace(logits=logits)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image1_path = root / "a.jpg"
            image2_path = root / "b.jpg"
            Image.new("RGB", (8, 8)).save(image1_path)
            Image.new("RGB", (8, 8)).save(image2_path)
            pair_example = PairExample(
                image1="a.jpg",
                image2="b.jpg",
                image1_path=image1_path,
                image2_path=image2_path,
                score1=1.0,
                score2=2.0,
                instruction=make_instruction(),
                answer=ANSWER_SECOND,
                fold=0,
                split="test",
            )
            model = FakeModel()
            model.train()

            result = predict_answer_by_likelihood(model, FakeProcessor(), pair_example)

            self.assertEqual(result.prediction, ANSWER_SECOND)
            self.assertGreater(result.answer_logprobs[ANSWER_SECOND], result.answer_logprobs[ANSWER_FIRST])
            self.assertGreater(result.answer_logprobs[ANSWER_SECOND], result.answer_logprobs[ANSWER_SIMILAR])
            metrics = evaluate_pairwise_likelihood(model, FakeProcessor(), [pair_example])
            self.assertEqual(metrics.count, 1)
            self.assertEqual(metrics.accuracy, 1.0)
            self.assertAlmostEqual(metrics.loss, -result.answer_logprobs[ANSWER_SECOND])
            self.assertLess(metrics.loss, 0.1)
            self.assertTrue(model.training)
            output = root / "predictions.csv"
            write_predictions_csv(
                [
                    {
                        "image1": "a.jpg",
                        "image2": "b.jpg",
                        "score1": 1.0,
                        "score2": 2.0,
                        "label": ANSWER_SECOND,
                        "prediction": result.prediction,
                        "valid": True,
                        "generation": "",
                        "first_logprob": result.answer_logprobs[ANSWER_FIRST],
                        "second_logprob": result.answer_logprobs[ANSWER_SECOND],
                        "similar_logprob": result.answer_logprobs[ANSWER_SIMILAR],
                    }
                ],
                output,
            )
            self.assertIn("second_logprob", output.read_text(encoding="utf-8"))

    def test_scut_annotation_loading_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "Images"
            image_dir.mkdir()
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (8, 8)).save(image_dir / name)
            (root / "labels.txt").write_text("a.jpg 1.5\nb.jpg 4.2\n", encoding="utf-8")
            records = load_annotations(root, "Images", "labels.txt")
            validate_dataset(records, expected_count=2)
            self.assertEqual(records[1].score, 4.2)

    def test_scut_archive_extraction_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "SCUT-FBP5500_v2.1.zip"
            output_root = root / "installed"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("SCUT-FBP5500_v2/Images/a.jpg", b"fake-a")
                archive.writestr("SCUT-FBP5500_v2/Images/b.jpg", b"fake-b")
                archive.writestr("SCUT-FBP5500_v2/train_test_files/All_labels.txt", "a.jpg 1.5\nb.jpg 4.2\n")

            result = extract_dataset_archive(archive_path, output_root, expected_count=2, show_progress=False)

            self.assertEqual(result.image_count, 2)
            self.assertTrue((output_root / "Images" / "a.jpg").exists())
            self.assertTrue((output_root / "Images" / "b.jpg").exists())
            self.assertEqual((output_root / "labels.txt").read_text(encoding="utf-8"), "a.jpg 1.5\nb.jpg 4.2\n")

    def test_downloaded_archive_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "SCUT-FBP5500_v2.1.zip"
            archive_path.write_bytes(b"archive")

            self.assertTrue(delete_archive_file(archive_path))
            self.assertFalse(archive_path.exists())
            self.assertFalse(delete_archive_file(archive_path))

    def test_metric_calculation(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.5, 2.5]
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 1.0 / 3.0)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), float(np.sqrt(0.5 / 3.0)))
        self.assertGreater(pearson_correlation(y_true, y_pred), 0.8)

    def test_bradley_terry_solver_toy_matrix(self) -> None:
        wins = np.array(
            [
                [0, 3, 3],
                [0, 0, 2],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
        scores = bradley_terry_scores(wins)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_import_config_sanity(self) -> None:
        config = load_config("configs/lmol_scut_fbp5500.yaml")
        self.assertEqual(config.model.model_id, "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(config.model.lora_r, 8)
        self.assertEqual(config.model.lora_alpha, 32)
        self.assertEqual(config.training.projector_lr, 2e-5)
        self.assertEqual(config.training.lora_lr, 2e-4)
        self.assertEqual(config.training.num_epochs, 1)
        self.assertEqual(config.training.per_device_batch_size, 1)
        self.assertEqual(config.training.gradient_accumulation_steps, 16)
        self.assertEqual(config.evaluation.batch_size, 2)
        self.assertFalse(config.training.gradient_checkpointing)
        self.assertEqual(config.training.objective, "choice_token")
        self.assertTrue(config.wandb.enabled)
        self.assertEqual(config.wandb.project, "lmol-fbp")
        self.assertEqual(config.wandb.mode, "online")
        self.assertIn("scut-fbp5500", config.wandb.tags)
        self.assertEqual(config.evaluation.eval_steps, 500)
        self.assertEqual(config.evaluation.eval_max_pairs, 256)

    def test_fold_scoped_config_outputs(self) -> None:
        config = load_config("configs/lmol_scut_fbp5500.yaml")
        fold_config = config_for_fold(config, 2)

        self.assertEqual(fold_config.dataset.fold_index, 2)
        self.assertEqual(fold_config.training.output_dir, config.training.output_dir / "fold_2")
        self.assertEqual(fold_config.evaluation.output_dir, config.evaluation.output_dir / "fold_2")
        self.assertEqual(fold_scoped_dir(config.training.output_dir / "fold_2", 2), config.training.output_dir / "fold_2")
        with self.assertRaisesRegex(ValueError, "fold-index"):
            config_for_fold(config, config.dataset.folds)


if __name__ == "__main__":
    unittest.main()
