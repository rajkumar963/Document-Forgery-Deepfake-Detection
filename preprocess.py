"""Image preprocessing module for document forgery detection."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_keep_aspect(image: np.ndarray, target_width: int = 1024) -> Tuple[np.ndarray, float]:
    """Resize an image to a fixed width while preserving aspect ratio.

    Args:
        image: Input BGR image.
        target_width: Desired output width.

    Returns:
        Tuple of:
            - resized image
            - scale factor (new_width / original_width)

    Raises:
        ValueError: If image is invalid or target_width is non-positive.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if target_width <= 0:
        raise ValueError("target_width must be > 0.")

    h, w = image.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Input image has invalid dimensions.")

    if w == target_width:
        return image.copy(), 1.0

    scale = target_width / float(w)
    target_height = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_image(image: np.ndarray, target_width: int = 1024) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run document preprocessing pipeline.

    Steps:
        1. Resize to fixed width (keep aspect ratio)
        2. Convert to grayscale
        3. Gaussian blur
        4. CLAHE contrast enhancement

    Args:
        image: Input BGR image.
        target_width: Width used for resizing.

    Returns:
        Tuple of:
            - processed grayscale image (uint8)
            - resized BGR image used as drawing canvas
            - scale factor relative to the original input

    Raises:
        ValueError: If input image is invalid.
    """
    resized_bgr, scale = resize_keep_aspect(image=image, target_width=target_width)

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE helps reveal low-contrast artifacts in scanned/printed documents.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return enhanced, resized_bgr, scale
