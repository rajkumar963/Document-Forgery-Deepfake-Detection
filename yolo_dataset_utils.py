"""Utilities for preparing and validating YOLO dataset structure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

LOGGER = logging.getLogger(__name__)

YOLO_CLASS_NAMES = {
    0: "stamp",
    1: "signature",
    2: "seal",
    3: "added_mark",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_label_format(label_path: str | Path, num_classes: int = 4) -> bool:
    """Validate one YOLO label file format.

    Each line must follow: class_id x_center y_center width height
    where coordinates are normalized to [0, 1].
    """
    path = Path(label_path).expanduser().resolve()
    if not path.exists():
        LOGGER.warning("Missing label file: %s", path)
        return False

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        LOGGER.error("Failed to read label file %s: %s", path, exc)
        return False

    valid = True
    for line_index, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            LOGGER.warning("Invalid label format in %s line %d", path, line_index)
            valid = False
            continue

        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            LOGGER.warning("Non-numeric label values in %s line %d", path, line_index)
            valid = False
            continue

        if class_id < 0 or class_id >= num_classes:
            LOGGER.warning("Class id out of range in %s line %d", path, line_index)
            valid = False

        for value in coords:
            if value < 0.0 or value > 1.0:
                LOGGER.warning("Normalized coordinate out of range in %s line %d", path, line_index)
                valid = False

    return valid


def _collect_images(folder: Path) -> List[Path]:
    """Collect images recursively under a folder."""
    if not folder.exists():
        return []
    images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    return images


def summarize_dataset(dataset_root: str | Path) -> Dict[str, int]:
    """Summarize YOLO dataset split counts and label coverage."""
    root = Path(dataset_root).expanduser().resolve()

    train_images_dir = root / "images" / "train"
    val_images_dir = root / "images" / "val"
    train_labels_dir = root / "labels" / "train"
    val_labels_dir = root / "labels" / "val"

    train_images = _collect_images(train_images_dir)
    val_images = _collect_images(val_images_dir)

    def count_labels(images: List[Path], labels_dir: Path) -> Tuple[int, int]:
        label_files = 0
        missing = 0
        for image in images:
            expected = labels_dir / f"{image.stem}.txt"
            if expected.exists():
                label_files += 1
            else:
                missing += 1
        return label_files, missing

    train_label_files, train_missing = count_labels(train_images, train_labels_dir)
    val_label_files, val_missing = count_labels(val_images, val_labels_dir)

    summary = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_label_files": train_label_files,
        "val_label_files": val_label_files,
        "train_missing_labels": train_missing,
        "val_missing_labels": val_missing,
    }
    LOGGER.info("Dataset summary: %s", summary)
    return summary


def validate_yolo_dataset(dataset_root: str | Path, num_classes: int = 4) -> bool:
    """Validate expected YOLO dataset structure and label files."""
    root = Path(dataset_root).expanduser().resolve()

    required_dirs = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]

    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        LOGGER.error("Missing required dataset directories: %s", missing_dirs)
        return False

    summary = summarize_dataset(root)

    train_images = _collect_images(root / "images" / "train")
    val_images = _collect_images(root / "images" / "val")

    all_valid = True

    for split, images, labels_dir in [
        ("train", train_images, root / "labels" / "train"),
        ("val", val_images, root / "labels" / "val"),
    ]:
        for image in images:
            label_file = labels_dir / f"{image.stem}.txt"
            if not label_file.exists():
                LOGGER.warning("Missing label for %s image: %s", split, image)
                all_valid = False
                continue
            if not check_label_format(label_file, num_classes=num_classes):
                all_valid = False

    if summary["train_images"] == 0 or summary["val_images"] == 0:
        LOGGER.warning("Dataset has empty split(s).")
        all_valid = False

    LOGGER.info("YOLO dataset validation result: %s", all_valid)
    return all_valid


def create_dataset_yaml(dataset_root: str | Path, output_yaml_path: str | Path | None = None) -> Path:
    """Create YOLO dataset.yaml with required class names.

    Returns:
        Path to generated dataset yaml.
    """
    root = Path(dataset_root).expanduser().resolve()
    out_path = Path(output_yaml_path).expanduser().resolve() if output_yaml_path else (root / "dataset.yaml")

    yaml_text = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: stamp\n"
        "  1: signature\n"
        "  2: seal\n"
        "  3: added_mark\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")
    LOGGER.info("Created dataset yaml: %s", out_path)
    return out_path
