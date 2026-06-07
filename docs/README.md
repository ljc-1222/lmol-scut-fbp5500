# Documentation

This directory keeps public-facing documentation and small result artifacts for the LMOL SCUT-FBP5500 release.

## Public Notes

- [`ASSUMPTIONS.md`](ASSUMPTIONS.md): reproducibility assumptions that are not fully specified by the paper or dataset package.
- [`TRAINING_ALIGNMENT.md`](TRAINING_ALIGNMENT.md): paper-to-code alignment notes, including the token-choice objective used by this implementation.
- [`HUGGING_FACE_UPLOAD.md`](HUGGING_FACE_UPLOAD.md): maintainer commands for rebuilding and uploading the Hugging Face Dataset repository package.

## Hugging Face Card Sources

- [`HF_DATASET_CARD.md`](HF_DATASET_CARD.md): source for the root `README.md` in the Hugging Face Dataset repository.
- [`HF_CHECKPOINT_README.md`](HF_CHECKPOINT_README.md): source for `checkpoints/fold0/README.md` in the Hugging Face Dataset repository.

## Result Artifacts

- [`results/fold_0/`](results/fold_0/): small JSON copies of the completed fold0 evaluation metrics, confusion matrix, and progress summary.

Internal cleanup logs, implementation plans, status notes, and release-review scratch reports are intentionally not kept in the public docs tree.
