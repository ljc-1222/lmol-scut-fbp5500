# Reproducibility Assumptions

- SCUT-FBP5500 can be installed with `scripts/download_scut_fbp5500.py`, which downloads from the official Google Drive link documented by the dataset authors. The script requires explicit `--accept-terms` confirmation because the official release limits use to non-commercial research.
- The expected local dataset layout is `data/raw/SCUT-FBP5500/Images/` plus `data/raw/SCUT-FBP5500/labels.txt`.
- The annotation file format is one image and one mean score per row: `<image_name> <score>`. CSV with the image in the first column and score in the second column is also accepted.
- If official SCUT folds are not present, this implementation uses deterministic k-fold splits from sorted image names shuffled by the configured seed. This is documented because the current repository did not contain official fold files.
- The stored instruction string is `<image1><image2>\nWhich face looks more attractive?`; the model collator renders it through the LLaVA processor chat template at runtime while keeping the two images as separate PIL images.
- The LoRA alpha is set to `32` for rank `8`, giving `alpha / r = 4`, which matches the paper's LoRA scaling statement.
- Score estimation uses predicted target-vs-reference orders and reference scores from the training split. Test labels are used only to compute PC, MAE, and RMSE after estimation.
