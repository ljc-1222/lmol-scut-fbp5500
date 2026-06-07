#!/usr/bin/env python
"""Download and install SCUT-FBP5500 into this repository's expected layout."""

from __future__ import annotations

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - used when imported by tests
    from scripts import _bootstrap  # type: ignore  # noqa: F401

import argparse
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from configs import load_config
from data.scut_fbp5500 import load_annotations, validate_dataset


OFFICIAL_GDRIVE_FILE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
OFFICIAL_RELEASE_PAGE = "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release"
ARCHIVE_NAME = "SCUT-FBP5500_v2.1.zip"
TERMS_ENV_VAR = "SCUT_FBP5500_ACCEPT_TERMS"


@dataclass(frozen=True)
class ArchiveInstallResult:
    """Summary of files installed from the SCUT-FBP5500 archive."""

    image_count: int
    label_member: str
    output_root: Path


def _accepted(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {member_name!r}")
    return path.parts


def _find_archive_members(
    archive: zipfile.ZipFile,
) -> tuple[list[tuple[zipfile.ZipInfo, str]], zipfile.ZipInfo]:
    image_members: list[tuple[zipfile.ZipInfo, str]] = []
    label_candidates: list[zipfile.ZipInfo] = []
    seen_images: set[str] = set()

    for member in archive.infolist():
        if member.is_dir():
            continue
        parts = _safe_parts(member.filename)
        lower_parts = tuple(part.lower() for part in parts)
        filename = parts[-1]
        lower_filename = filename.lower()

        if "images" in lower_parts and lower_filename.endswith((".jpg", ".jpeg")):
            if filename in seen_images:
                raise ValueError(f"Duplicate image filename in archive: {filename}")
            seen_images.add(filename)
            image_members.append((member, filename))

        if lower_filename == "all_labels.txt":
            label_candidates.append(member)

    if not image_members:
        raise ValueError("No image files were found under an Images/ directory in the archive")
    if not label_candidates:
        raise ValueError("All_labels.txt was not found in the archive")

    for candidate in label_candidates:
        lower_parts = tuple(part.lower() for part in _safe_parts(candidate.filename))
        if "train_test_files" in lower_parts:
            return image_members, candidate
    return image_members, label_candidates[0]


def _copy_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def extract_dataset_archive(
    archive_path: str | Path,
    output_root: str | Path,
    expected_count: int = 5500,
    show_progress: bool = True,
) -> ArchiveInstallResult:
    """Extract the official SCUT-FBP5500 zip into Images/ and labels.txt."""

    archive_path = Path(archive_path)
    output_root = Path(output_root)
    images_dir = output_root / "Images"
    labels_path = output_root / "labels.txt"

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        image_members, label_member = _find_archive_members(archive)
        if len(image_members) != expected_count:
            raise ValueError(f"Expected {expected_count} images in archive, found {len(image_members)}")

        output_root.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        image_iterator = image_members
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError:  # pragma: no cover - tqdm is in requirements
                tqdm = lambda items, **_: items  # type: ignore
            image_iterator = tqdm(image_members, desc="Extracting images", unit="image")

        for member, filename in image_iterator:
            _copy_member(archive, member, images_dir / filename)
        _copy_member(archive, label_member, labels_path)

    return ArchiveInstallResult(
        image_count=len(image_members),
        label_member=label_member.filename,
        output_root=output_root,
    )


def download_official_archive(archive_path: str | Path, force: bool = False) -> Path:
    """Download the official SCUT-FBP5500 archive from Google Drive."""

    archive_path = Path(archive_path)
    if archive_path.exists() and not force:
        print(f"Using existing archive: {archive_path}")
        return archive_path

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required. Install dependencies with: python -m pip install -r requirements.txt") from exc

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    if partial_path.exists():
        partial_path.unlink()

    print(f"Downloading SCUT-FBP5500 from the official release channel: {OFFICIAL_RELEASE_PAGE}")
    result = gdown.download(id=OFFICIAL_GDRIVE_FILE_ID, output=str(partial_path), quiet=False)
    if result is None or not partial_path.exists():
        raise RuntimeError("Google Drive download failed")
    partial_path.replace(archive_path)
    return archive_path


def validate_installed_dataset(config_path: str | Path) -> int:
    """Validate the configured SCUT-FBP5500 dataset and return the record count."""

    config = load_config(config_path)
    records = load_annotations(config.dataset.root, config.dataset.image_dir, config.dataset.annotation_file)
    validate_dataset(
        records,
        expected_count=config.dataset.expected_count,
        score_min=config.dataset.score_min,
        score_max=config.dataset.score_max,
        require_images=True,
    )
    return len(records)


def delete_archive_file(archive_path: str | Path) -> bool:
    """Delete the downloaded archive if it exists."""

    archive_path = Path(archive_path)
    if not archive_path.exists():
        return False
    archive_path.unlink()
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmol_scut_fbp5500.yaml")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing SCUT-FBP5500_v2.1.zip instead of downloading it.",
    )
    parser.add_argument("--force-download", action="store_true", help="Redownload the archive even if it already exists.")
    parser.add_argument("--force-install", action="store_true", help="Reinstall files even if validation already passes.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded zip after installation.")
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="Confirm SCUT-FBP5500 is being used under the official non-commercial research terms.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_root = Path(config.dataset.root)
    archive_path = args.archive or dataset_root.parent / ARCHIVE_NAME
    should_delete_archive = args.archive is None and not args.keep_archive

    if not (args.accept_terms or _accepted(os.environ.get(TERMS_ENV_VAR))):
        print("SCUT-FBP5500 is released by its authors for non-commercial research use only.")
        print(f"Review the official release page: {OFFICIAL_RELEASE_PAGE}")
        print(f"Rerun with --accept-terms, or set {TERMS_ENV_VAR}=1, to confirm compatible use.")
        return 2

    if not args.force_install:
        try:
            count = validate_installed_dataset(args.config)
        except Exception as exc:
            print(f"Existing dataset validation did not pass: {exc}")
        else:
            print(f"SCUT-FBP5500 is already installed and valid: {count} records in {dataset_root}")
            return 0

    try:
        if args.archive is None:
            archive_path = download_official_archive(archive_path, force=args.force_download)
        result = extract_dataset_archive(
            archive_path=archive_path,
            output_root=dataset_root,
            expected_count=config.dataset.expected_count,
        )
        count = validate_installed_dataset(args.config)
    except Exception as exc:
        print(f"SCUT-FBP5500 download/install failed: {exc}", file=sys.stderr)
        return 1

    print(f"Installed {result.image_count} images into {result.output_root / 'Images'}")
    print(f"Installed labels from {result.label_member} into {result.output_root / 'labels.txt'}")
    print(f"SCUT-FBP5500 validation passed: {count} records in {dataset_root}")
    if should_delete_archive:
        try:
            if delete_archive_file(archive_path):
                print(f"Deleted downloaded archive: {archive_path}")
        except OSError as exc:
            print(f"Warning: could not delete downloaded archive {archive_path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
