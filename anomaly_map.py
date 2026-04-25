"""Anomaly map post-processing utilities.

Converts continuous anomaly scores into a robust binary suspicious mask and
candidate contours for downstream bounding box generation.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

ThresholdStrategy = Literal["adaptive", "fixed", "otsu", "percentile"]


def _normalize_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a float map to [0, 1]."""
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v + eps)


def _to_odd(value: int, minimum: int = 3) -> int:
    """Clamp to a positive odd integer."""
    val = max(minimum, int(value))
    return val if val % 2 == 1 else val + 1


def _threshold_map(
    smoothed_map: np.ndarray,
    strategy: ThresholdStrategy,
    threshold_value: float,
    percentile: float,
    adaptive_block_size: int,
    adaptive_c: float,
) -> np.ndarray:
    """Threshold smoothed anomaly map into a binary mask."""
    map_01 = _normalize_map(smoothed_map)
    map_u8 = np.clip(map_01 * 255.0, 0, 255).astype(np.uint8)

    if strategy == "fixed":
        threshold_clamped = float(np.clip(threshold_value, 0.0, 1.0))
        mask = (map_01 >= threshold_clamped).astype(np.uint8) * 255
        return mask

    if strategy == "percentile":
        percentile_clamped = float(np.clip(percentile, 0.0, 100.0))
        threshold_dyn = float(np.percentile(map_01, percentile_clamped))
        mask = (map_01 >= threshold_dyn).astype(np.uint8) * 255
        return mask

    if strategy == "otsu":
        _, mask = cv2.threshold(map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    if strategy == "adaptive":
        block_size = _to_odd(adaptive_block_size, minimum=3)
        mask = cv2.adaptiveThreshold(
            map_u8,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=adaptive_c,
        )
        return mask

    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def _remove_small_components(mask: np.ndarray, min_region_area: int) -> np.ndarray:
    """Remove connected components smaller than min_region_area."""
    if min_region_area <= 1:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_region_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def build_anomaly_outputs(
    feature_map: np.ndarray,
    threshold_strategy: ThresholdStrategy = "percentile",
    threshold_value: float = 0.6,
    percentile: float = 90.0,
    smoothing_ksize: int = 7,
    adaptive_block_size: int = 35,
    adaptive_c: float = -2.0,
    min_region_area: int = 200,
    morph_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate raw anomaly map, binary suspicious mask, and candidate contours.

    Args:
        feature_map: Input score map from feature extraction (any numeric range).
        threshold_strategy: One of adaptive, fixed, otsu, percentile.
        threshold_value: Fixed threshold in [0, 1] when strategy is fixed.
        percentile: Percentile threshold in [0, 100] for percentile strategy.
        smoothing_ksize: Gaussian smoothing kernel size.
        adaptive_block_size: Local block size for adaptive thresholding.
        adaptive_c: Constant subtraction term for adaptive thresholding.
        min_region_area: Minimum connected component area to keep.
        morph_kernel_size: Morphological cleanup kernel size.

    Returns:
        Tuple of:
            - raw anomaly map in [0, 1]
            - cleaned binary mask in {0, 255}
            - candidate contours from the cleaned mask

    Raises:
        ValueError: If input map is invalid.
    """
    if feature_map is None or feature_map.size == 0:
        raise ValueError("feature_map is empty or None.")
    if feature_map.ndim != 2:
        raise ValueError("feature_map must be a 2D array.")

    raw_map = _normalize_map(feature_map)

    ksize = _to_odd(smoothing_ksize, minimum=3)
    smoothed = cv2.GaussianBlur(raw_map, (ksize, ksize), sigmaX=0)

    thresholded = _threshold_map(
        smoothed_map=smoothed,
        strategy=threshold_strategy,
        threshold_value=threshold_value,
        percentile=percentile,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )

    morph_size = _to_odd(morph_kernel_size, minimum=3)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    cleaned = _remove_small_components(cleaned, min_region_area=min_region_area)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    LOGGER.info("Anomaly post-processing produced %d candidate contour(s).", len(contours))

    return smoothed.astype(np.float32), cleaned.astype(np.uint8), contours
