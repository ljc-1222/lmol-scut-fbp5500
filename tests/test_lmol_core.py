"""Unit tests for the clean LMOL-on-SCUT implementation."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from configs import config_for_fold, fold_scoped_dir, load_config
from data.instruction_dataset import LMOLDataCollator, assert_pair_images_exist
from data.pairs import (
    ANSWER_FIRST,
    ANSWER_SECOND,
    ANSWER_SIMILAR,
    PairExample,
    build_balanced_pairs,
    iter_reference_pairs,
    label_from_scores,
    make_instruction,
    read_pairs_csv,
)
from data.references import assert_no_reference_leakage, score_bin_index, select_score_bin_references
from data.scut_fbp5500 import SCUTRecord, load_annotations, validate_dataset
from evaluation.inference import (
    PREDICTION_CSV_FIELDS,
    evaluate_pairwise_likelihood,
    predict_answer_by_likelihood,
    prediction_csv_row,
    write_predictions_csv,
)
from evaluation.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error
from evaluation.score_estimation import estimate_metrics_from_predictions_csv, estimate_scores_from_predictions, load_pairwise_predictions
from evaluation.uol_bradley_terry import (
    ReferenceComparison,
    estimate_score_uol_bt,
    reference_score_support,
    uol_bt_class_probabilities,
    uol_bt_log_likelihood,
)
from model.lmol import _bnb_4bit_skip_modules, _language_lora_target_modules, _validate_processor_image_size, _validate_trainable_scope
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

    def test_processor_image_size_is_validated(self) -> None:
        matching = SimpleNamespace(image_processor=SimpleNamespace(crop_size={"height": 336, "width": 336}))
        _validate_processor_image_size(matching, 336)

        shortest_edge = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 336}))
        _validate_processor_image_size(shortest_edge, 336)

        mismatched = SimpleNamespace(image_processor=SimpleNamespace(crop_size={"height": 224, "width": 224}))
        with self.assertRaisesRegex(ValueError, "processor image size"):
            _validate_processor_image_size(mismatched, 336)

    def test_collator_supervises_only_first_answer_token(self) -> None:
        import torch

        class FakeTokenizer:
            pad_token_id = 0

            def __init__(self) -> None:
                self.answer_ids = {ANSWER_FIRST: [10, 13], ANSWER_SECOND: [11, 13], ANSWER_SIMILAR: [12, 13]}

            def __call__(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
                if not add_special_tokens and text in self.answer_ids:
                    return SimpleNamespace(input_ids=self.answer_ids[text])
                ids = [1, 2, 3]
                for answer, answer_ids in self.answer_ids.items():
                    if text.endswith(answer):
                        ids.extend(answer_ids)
                        break
                return SimpleNamespace(input_ids=ids)

        class FakeProcessor:
            def __init__(self) -> None:
                self.tokenizer = FakeTokenizer()

            def __call__(self, text: list[str], images: list[object], padding: bool, return_tensors: str) -> dict[str, torch.Tensor]:
                del images, padding, return_tensors
                rows = [self.tokenizer(item).input_ids for item in text]
                return {"input_ids": torch.tensor(rows, dtype=torch.long)}

        batch = LMOLDataCollator(FakeProcessor())(
            [{"image1": object(), "image2": object(), "instruction": make_instruction(), "answer": ANSWER_FIRST}]
        )
        supervised = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0].tolist()

        self.assertEqual(supervised, [3])
        self.assertEqual(int(batch["labels"][0, 3]), 10)
        self.assertEqual(int(batch["choice_labels"][0]), 0)
        self.assertEqual(int(batch["choice_positions"][0]), 2)

    def test_theta_label_construction(self) -> None:
        self.assertEqual(label_from_scores(4.0, 4.0, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.8, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.81, theta=0.2), ANSWER_SIMILAR)
        self.assertEqual(label_from_scores(4.0, 3.79, theta=0.2), ANSWER_FIRST)
        self.assertEqual(label_from_scores(2.0, 2.21, theta=0.2), ANSWER_SECOND)

    def test_instruction_and_answer_formatting(self) -> None:
        self.assertEqual(make_instruction(), "<image1><image2>, which face looks more attractive?")
        self.assertEqual(ANSWER_FIRST, "First.")
        self.assertEqual(ANSWER_SECOND, "Second.")
        self.assertEqual(ANSWER_SIMILAR, "Similar.")

    def test_legacy_evaluation_grid_config_keys_fail_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "legacy_eval.yaml"
            text = Path("configs/lmol_scut_fbp5500.yaml").read_text(encoding="utf-8")
            config_path.write_text(text.replace("eval_steps: 500", "reference_count: 340\n  bt_grid_size: 401\n  eval_steps: 500"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Legacy evaluation config keys"):
                load_config(config_path)

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
            custom_instruction = "<image1><image2>, choose the more attractive face."
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
  instruction: "{custom_instruction}"
  output_dir: {output_dir}
  seed: 7

model:
  model_id: llava-hf/llava-1.5-7b-hf
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

evaluation:
  output_dir: {root / "eval"}
  batch_size: 1
  max_new_tokens: 4
  reference_bin_width: 0.1
  reference_max_per_bin:
  reference_seed: 7
  score_min: 1.0
  score_max: 5.0
  bt_delta: 2.0
  bt_k: 10.0
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
                self.assertTrue((fold_dir / "references.csv").exists())
                self.assertTrue((fold_dir / "test_reference_pairs.csv").exists())
                train_pairs = read_pairs_csv(fold_dir / "train_pairs.csv")
                reference_pairs = read_pairs_csv(fold_dir / "test_reference_pairs.csv")
                with (fold_dir / "train_images.csv").open("r", newline="", encoding="utf-8") as handle:
                    train_images = {row["image"] for row in csv.DictReader(handle)}
                with (fold_dir / "test_images.csv").open("r", newline="", encoding="utf-8") as handle:
                    test_images = {row["image"] for row in csv.DictReader(handle)}
                with (fold_dir / "references.csv").open("r", newline="", encoding="utf-8") as handle:
                    reference_rows = list(csv.DictReader(handle))
                reference_images = {row["image"] for row in reference_rows}
                self.assertEqual(len(train_pairs), 6)
                self.assertEqual({pair.fold for pair in train_pairs}, {fold})
                self.assertEqual({pair.instruction for pair in train_pairs}, {custom_instruction})
                self.assertEqual({pair.instruction for pair in reference_pairs}, {custom_instruction})
                self.assertFalse(train_images & test_images)
                self.assertTrue(reference_images <= train_images)
                self.assertFalse(reference_images & test_images)
                self.assertTrue(all(row["split"] == "reference" for row in reference_rows))
                self.assertEqual(
                    Counter(pair.answer for pair in train_pairs),
                    Counter({ANSWER_FIRST: 2, ANSWER_SECOND: 2, ANSWER_SIMILAR: 2}),
                )
                self.assertFalse(any(pair.image1 == pair.image2 for pair in reference_pairs))
                self.assertTrue(all(pair.split == "test" for pair in reference_pairs))
                self.assertTrue(all(pair.image1 in test_images for pair in reference_pairs))
                self.assertTrue(all(pair.image2 in reference_images for pair in reference_pairs))

    def test_score_bin_reference_selection_and_leakage_checks(self) -> None:
        train_records = [
            SCUTRecord("a.jpg", 1.01, Path("a.jpg")),
            SCUTRecord("b.jpg", 1.04, Path("b.jpg")),
            SCUTRecord("c.jpg", 1.12, Path("c.jpg")),
            SCUTRecord("d.jpg", 3.25, Path("d.jpg")),
            SCUTRecord("e.jpg", 4.98, Path("e.jpg")),
        ]
        test_records = [SCUTRecord("z.jpg", 3.0, Path("z.jpg"))]

        self.assertEqual(score_bin_index(1.0, 1.0, 5.0, 0.1), 0)
        self.assertEqual(score_bin_index(5.0, 1.0, 5.0, 0.1), 39)
        references = select_score_bin_references(
            train_records,
            fold=0,
            score_min=1.0,
            score_max=5.0,
            bin_width=0.1,
            max_per_bin=None,
            seed=3,
        )
        self.assertEqual([row.record.image for row in references], ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
        self.assertEqual(Counter(row.score_bin for row in references)["1.0"], 2)
        assert_no_reference_leakage(train_records, test_records, [row.record for row in references])

        capped = select_score_bin_references(
            train_records,
            fold=0,
            score_min=1.0,
            score_max=5.0,
            bin_width=0.1,
            max_per_bin=1,
            seed=3,
        )
        self.assertLessEqual(max(Counter(row.score_bin for row in capped).values()), 1)
        capped_again = select_score_bin_references(
            train_records,
            fold=0,
            score_min=1.0,
            score_max=5.0,
            bin_width=0.1,
            max_per_bin=1,
            seed=3,
        )
        self.assertEqual([row.record.image for row in capped], [row.record.image for row in capped_again])
        with self.assertRaisesRegex(ValueError, "train split only"):
            assert_no_reference_leakage(train_records, test_records, [test_records[0]])

    def test_synthetic_reference_pipeline_without_images_or_checkpoint(self) -> None:
        train_records = [
            SCUTRecord("a.jpg", 1.0, Path("a.jpg")),
            SCUTRecord("b.jpg", 1.1, Path("b.jpg")),
            SCUTRecord("c.jpg", 2.0, Path("c.jpg")),
            SCUTRecord("d.jpg", 3.2, Path("d.jpg")),
            SCUTRecord("e.jpg", 4.0, Path("e.jpg")),
            SCUTRecord("f.jpg", 5.0, Path("f.jpg")),
        ]
        test_records = [
            SCUTRecord("t1.jpg", 3.2, Path("t1.jpg")),
            SCUTRecord("t2.jpg", 4.8, Path("t2.jpg")),
        ]
        references = select_score_bin_references(
            train_records,
            fold=0,
            score_min=1.0,
            score_max=5.0,
            bin_width=0.1,
            max_per_bin=None,
            seed=42,
        )
        reference_records = [row.record for row in references]
        reference_images = {record.image for record in reference_records}
        train_images = {record.image for record in train_records}
        test_images = {record.image for record in test_records}

        self.assertEqual(reference_images, train_images)
        self.assertFalse(train_images & test_images)
        self.assertFalse(reference_images & test_images)
        self.assertEqual(Counter(row.score_bin for row in references)["1.0"], 1)
        self.assertEqual(Counter(row.score_bin for row in references)["1.1"], 1)
        assert_no_reference_leakage(train_records, test_records, reference_records)

        pairs = list(iter_reference_pairs(test_records, reference_records, theta=0.2, fold=0))
        self.assertEqual(len(pairs), len(test_records) * len(reference_records))
        self.assertTrue(all(pair.image1 in test_images for pair in pairs))
        self.assertTrue(all(pair.image2 in reference_images for pair in pairs))
        self.assertTrue(all(pair.split == "test" for pair in pairs))

        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "synthetic_predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
                writer.writeheader()
                for pair in pairs:
                    writer.writerow(
                        {
                            "target_image": pair.image1,
                            "reference_image": pair.image2,
                            "target_score": pair.score1,
                            "reference_score": pair.score2,
                            "true_answer": pair.answer,
                            "predicted_answer": pair.answer,
                            "decoding_method": "first_token_choice",
                            "valid_choice": True,
                        }
                    )

            rows = load_pairwise_predictions(predictions)
            estimates, true_scores = estimate_scores_from_predictions(rows, delta=2.0, k=10.0)
            self.assertEqual(set(estimates), {"t1.jpg", "t2.jpg"})
            self.assertEqual(set(true_scores), {"t1.jpg", "t2.jpg"})
            self.assertTrue(all(1.0 <= score <= 5.0 for score in estimates.values()))

            metrics = estimate_metrics_from_predictions_csv(
                predictions,
                delta=2.0,
                k=10.0,
            )
            self.assertEqual(metrics["count"], 2)
            self.assertIn("mae", metrics)
            self.assertIn("rmse", metrics)
            self.assertIsNotNone(metrics["pc"])

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
            write_predictions_csv([prediction_csv_row(pair_example, result)], output)
            with output.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, PREDICTION_CSV_FIELDS)
                row = next(reader)
            self.assertEqual(row["decoding_method"], "first_token_choice")
            self.assertEqual(row["valid_choice"], "True")
            self.assertIn("logprob_second", row)
            self.assertNotIn("generation", row)

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

    def test_uol_bt_probabilities_likelihood_and_support_estimation(self) -> None:
        negative = uol_bt_class_probabilities(1.0, 4.0, delta=2.0, k=10.0)
        centered = uol_bt_class_probabilities(3.2, 3.2, delta=2.0, k=10.0)
        positive = uol_bt_class_probabilities(5.0, 2.0, delta=2.0, k=10.0)

        for probabilities in (negative, centered, positive):
            self.assertTrue(all(np.isfinite(value) for value in probabilities.values()))
            self.assertTrue(all(value >= 0.0 for value in probabilities.values()))
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(negative[ANSWER_SECOND], negative[ANSWER_SIMILAR])
        self.assertGreater(centered[ANSWER_SIMILAR], centered[ANSWER_FIRST])
        self.assertGreater(centered[ANSWER_SIMILAR], centered[ANSWER_SECOND])
        self.assertGreater(positive[ANSWER_FIRST], positive[ANSWER_SIMILAR])

        high = estimate_score_uol_bt(
            [
                ReferenceComparison(reference_score=1.5, prediction=ANSWER_FIRST),
                ReferenceComparison(reference_score=2.5, prediction=ANSWER_FIRST),
                ReferenceComparison(reference_score=3.5, prediction=ANSWER_FIRST),
            ],
            delta=2.0,
            k=10.0,
        )
        self.assertEqual(high, 3.5)

        low = estimate_score_uol_bt(
            [
                ReferenceComparison(reference_score=1.5, prediction=ANSWER_SECOND),
                ReferenceComparison(reference_score=2.5, prediction=ANSWER_SECOND),
                ReferenceComparison(reference_score=3.5, prediction=ANSWER_SECOND),
            ],
            delta=2.0,
            k=10.0,
        )
        self.assertEqual(low, 1.5)

        similar = estimate_score_uol_bt(
            [
                ReferenceComparison(reference_score=3.1, prediction=ANSWER_SIMILAR),
                ReferenceComparison(reference_score=3.2, prediction=ANSWER_SIMILAR),
                ReferenceComparison(reference_score=3.3, prediction=ANSWER_SIMILAR),
            ],
            delta=2.0,
            k=10.0,
        )
        self.assertEqual(similar, 3.2)
        self.assertLess(
            uol_bt_log_likelihood(3.2, [ReferenceComparison(3.2, ANSWER_FIRST)], delta=2.0, k=10.0),
            uol_bt_log_likelihood(3.2, [ReferenceComparison(3.2, ANSWER_SIMILAR)], delta=2.0, k=10.0),
        )
        self.assertEqual(
            reference_score_support(
                [
                    ReferenceComparison(reference_score=3.3, prediction=ANSWER_SIMILAR),
                    ReferenceComparison(reference_score=3.1, prediction=ANSWER_SIMILAR),
                    ReferenceComparison(reference_score=3.3, prediction=ANSWER_FIRST),
                ]
            ),
            (3.1, 3.3),
        )

    def test_uol_bt_tie_handling_uses_mean_of_tied_support(self) -> None:
        estimate = estimate_score_uol_bt(
            [ReferenceComparison(reference_score=3.0, prediction=ANSWER_SIMILAR)],
            candidate_scores=[2.0, 4.0],
            delta=2.0,
            k=1.0,
        )

        self.assertEqual(estimate, 3.0)

    def test_score_estimation_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            metrics_path = root / "metrics.json"
            rows = [
                ("low.jpg", 2.0, "r1.jpg", 1.8, ANSWER_FIRST),
                ("low.jpg", 2.0, "r2.jpg", 2.0, ANSWER_SIMILAR),
                ("low.jpg", 2.0, "r3.jpg", 2.2, ANSWER_SECOND),
                ("high.jpg", 4.0, "r4.jpg", 3.8, ANSWER_FIRST),
                ("high.jpg", 4.0, "r5.jpg", 4.0, ANSWER_SIMILAR),
                ("high.jpg", 4.0, "r6.jpg", 4.2, ANSWER_SECOND),
            ]
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
                writer.writeheader()
                for target, target_score, reference, reference_score, answer in rows:
                    writer.writerow(
                        {
                            "target_image": target,
                            "reference_image": reference,
                            "target_score": target_score,
                            "reference_score": reference_score,
                            "true_answer": answer,
                            "predicted_answer": answer,
                            "decoding_method": "first_token_choice",
                            "valid_choice": True,
                        }
                    )

            metrics = estimate_metrics_from_predictions_csv(
                predictions,
                delta=2.0,
                k=10.0,
                output_path=metrics_path,
            )

            self.assertEqual(json.loads(metrics_path.read_text(encoding="utf-8")), metrics)
            self.assertEqual(metrics["score_estimator"], "uol_bradley_terry")
            self.assertEqual(metrics["count"], 2)

    def test_score_estimation_reads_legacy_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "legacy_predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image1", "score1", "score2", "prediction"])
                writer.writeheader()
                writer.writerow({"image1": "a.jpg", "score1": 3.0, "score2": 2.8, "prediction": ANSWER_FIRST})

            rows = load_pairwise_predictions(predictions)

        self.assertEqual(rows[0].target_image, "a.jpg")
        self.assertEqual(rows[0].target_score, 3.0)
        self.assertEqual(rows[0].reference_score, 2.8)
        self.assertEqual(rows[0].predicted_answer, ANSWER_FIRST)
        self.assertTrue(rows[0].valid_choice)

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
        self.assertTrue(config.wandb.enabled)
        self.assertEqual(config.wandb.project, "lmol-fbp")
        self.assertEqual(config.wandb.mode, "online")
        self.assertIn("scut-fbp5500", config.wandb.tags)
        self.assertEqual(config.pairs.instruction, "<image1><image2>, which face looks more attractive?")
        self.assertEqual(config.evaluation.reference_bin_width, 0.1)
        self.assertIsNone(config.evaluation.reference_max_per_bin)
        self.assertEqual(config.evaluation.bt_delta, 2.0)
        self.assertEqual(config.evaluation.bt_k, 10.0)
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
