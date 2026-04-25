"""Heuristic suspicious region detection for documents."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

BoundingBox = Tuple[int, int, int, int]


def detect_suspicious_regions(
    processed_image: np.ndarray,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    min_area: int = 400,
) -> List[BoundingBox]:
    """Detect suspicious regions using edge and contour heuristics.

    Pipeline:
        1. Canny edge detection
        2. Morphological close + dilation to connect nearby edges
        3. Contour extraction
        4. Area filtering

    Args:
        processed_image: Preprocessed single-channel (grayscale) image.
        canny_threshold1: Lower Canny threshold.
        canny_threshold2: Upper Canny threshold.
        min_area: Minimum contour area to keep.

    Returns:
        List of bounding boxes as ``(x, y, w, h)``.

    Raises:
        ValueError: If the input image is invalid.
    """
    if processed_image is None or processed_image.size == 0:
        raise ValueError("Processed image is empty or None.")

    if len(processed_image.shape) != 2:
        raise ValueError("Processed image must be single-channel grayscale.")

    edges = cv2.Canny(processed_image, threshold1=canny_threshold1, threshold2=canny_threshold2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore tiny or degenerate boxes even if area passed due to irregular shape.
        if w < 8 or h < 8:
            continue

        boxes.append((x, y, w, h))

    # Sort for stable output top-to-bottom then left-to-right.
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes
