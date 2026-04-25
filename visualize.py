"""Visualization utilities for suspicious region bounding boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

BoundingBox = Tuple[int, int, int, int]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Iterable[BoundingBox],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: Input BGR image.
        boxes: Iterable of bounding boxes ``(x, y, w, h)``.
        color: BGR color tuple for rectangles.
        thickness: Rectangle border thickness.

    Returns:
        A copy of the input image with rectangles drawn.

    Raises:
        ValueError: If the input image is invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image for visualization is empty or None.")

    canvas = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    return canvas


def save_output_image(output_image: np.ndarray, output_path: str | Path) -> Path:
    """Save an output image to disk, creating parent folders if needed.

    Args:
        output_image: BGR image to save.
        output_path: Destination path.

    Returns:
        Resolved path to the saved image.

    Raises:
        ValueError: If output image is invalid.
        IOError: If the write operation fails.
    """
    if output_image is None or output_image.size == 0:
        raise ValueError("Output image is empty or None.")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), output_image)
    if not success:
        raise IOError(f"Failed to save output image to: {path}")

    return path
