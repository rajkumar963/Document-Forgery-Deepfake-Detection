"""Reusable feature extraction utilities for suspicious document regions."""

from __future__ import annotations

import logging
from typing import Any, Dict

import cv2
import numpy as np
from skimage.measure import shannon_entropy

LOGGER = logging.getLogger(__name__)

def compute_edge_density(gray_region: np.ndarray) -> float:
    """Compute edge density using Canny edges.

    Returns the fraction of edge pixels in [0, 1].
    """
    if gray_region.size == 0:
        return 0.0

    median_val = float(np.median(gray_region))
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val + 30))

    edges = cv2.Canny(gray_region, low, high)
    return float(np.count_nonzero(edges) / max(1, edges.size))


def compute_local_variance(gray_region: np.ndarray, kernel_size: int = 5) -> float:
    """Compute mean local variance in a grayscale crop."""
    if gray_region.size == 0:
        return 0.0

    k = max(3, int(kernel_size) | 1)
    region_f = gray_region.astype(np.float32)
    mean = cv2.GaussianBlur(region_f, (k, k), 0)
    mean_sq = cv2.GaussianBlur(region_f * region_f, (k, k), 0)
    local_var = np.maximum(0.0, mean_sq - mean * mean)
    return float(np.mean(local_var))


def compute_entropy(gray_region: np.ndarray) -> float:
    """Compute Shannon entropy for a grayscale crop."""
    if gray_region.size == 0:
        return 0.0
    return float(shannon_entropy(gray_region))


def compute_connected_components_stats(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute connected-component and foreground statistics.

    Uses Otsu thresholding on an inverted binary image to focus on text-like
    foreground components.
    """
    if gray_region.size == 0:
        return {
            "cc_count": 0.0,
            "cc_area_mean": 0.0,
            "cc_area_std": 0.0,
            "fg_bg_ratio": 0.0,
            "small_cc_ratio": 0.0,
        }

    _, binary_inv = cv2.threshold(
        gray_region,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    if num_labels <= 1:
        fg_ratio = float(np.count_nonzero(binary_inv) / max(1, binary_inv.size))
        return {
            "cc_count": 0.0,
            "cc_area_mean": 0.0,
            "cc_area_std": 0.0,
            "fg_bg_ratio": fg_ratio,
            "small_cc_ratio": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    small_ratio = float(np.mean(areas < 12.0)) if areas.size > 0 else 0.0
    fg_ratio = float(np.count_nonzero(binary_inv) / max(1, binary_inv.size))

    return {
        "cc_count": float(len(areas)),
        "cc_area_mean": float(np.mean(areas)) if areas.size else 0.0,
        "cc_area_std": float(np.std(areas)) if areas.size else 0.0,
        "fg_bg_ratio": fg_ratio,
        "small_cc_ratio": small_ratio,
    }


def compute_projection_features(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute horizontal and vertical projection irregularity features."""
    if gray_region.size == 0:
        return {
            "h_proj_irregularity": 0.0,
            "v_proj_irregularity": 0.0,
            "h_proj_cv": 0.0,
            "v_proj_cv": 0.0,
        }

    _, binary_inv = cv2.threshold(
        gray_region,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    h_proj = np.sum(binary_inv > 0, axis=1).astype(np.float32)
    v_proj = np.sum(binary_inv > 0, axis=0).astype(np.float32)

    def _stats(arr: np.ndarray) -> tuple[float, float]:
        if arr.size == 0:
            return 0.0, 0.0
        cv = float(np.std(arr) / max(1e-6, np.mean(arr)))
        irregular = float(np.std(np.diff(arr))) if arr.size > 1 else 0.0
        return irregular, cv

    h_irr, h_cv = _stats(h_proj)
    v_irr, v_cv = _stats(v_proj)

    return {
        "h_proj_irregularity": h_irr,
        "v_proj_irregularity": v_irr,
        "h_proj_cv": h_cv,
        "v_proj_cv": v_cv,
    }


def _compute_contour_overlap_density(gray_region: np.ndarray) -> float:
    """Estimate contour overlap/crowding density for overwrite cues."""
    if gray_region.size == 0:
        return 0.0

    edges = cv2.Canny(gray_region, 40, 120)
    if not np.any(edges):
        return 0.0

    dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    overlap = np.logical_and(edges > 0, dilated > 0)
    return float(np.count_nonzero(overlap) / max(1, gray_region.size))


def normalize_feature_dict(features: Dict[str, float]) -> Dict[str, float]:
    """Create normalized helper values for more stable rule-based scoring."""
    norm = dict(features)

    norm["norm_std_intensity"] = float(np.clip(norm.get("std_intensity", 0.0) / 64.0, 0.0, 1.0))
    norm["norm_local_variance"] = float(np.clip(norm.get("local_variance", 0.0) / 1200.0, 0.0, 1.0))
    norm["norm_laplacian_variance"] = float(np.clip(norm.get("laplacian_variance", 0.0) / 400.0, 0.0, 1.0))
    norm["norm_entropy"] = float(np.clip(norm.get("entropy", 0.0) / 8.0, 0.0, 1.0))
    norm["norm_cc_count"] = float(np.clip(norm.get("cc_count", 0.0) / 120.0, 0.0, 1.0))
    norm["norm_proj_irregularity"] = float(
        np.clip(
            (norm.get("h_proj_irregularity", 0.0) + norm.get("v_proj_irregularity", 0.0)) / 120.0,
            0.0,
            1.0,
        )
    )

    return norm


def extract_region_features(gray_region: np.ndarray) -> Dict[str, Any]:
    """Extract handcrafted features from one suspicious grayscale crop.

    Args:
        gray_region: 2D grayscale crop.

    Returns:
        Dictionary of numerical features used by Day 6 classification.
    """
    if gray_region is None or gray_region.size == 0:
        LOGGER.warning("extract_region_features received empty crop.")
        return {
            "valid": False,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "local_variance": 0.0,
            "edge_density": 0.0,
            "laplacian_variance": 0.0,
            "entropy": 0.0,
            "stroke_density": 0.0,
            "cc_count": 0.0,
            "cc_area_mean": 0.0,
            "cc_area_std": 0.0,
            "fg_bg_ratio": 0.0,
            "small_cc_ratio": 0.0,
            "contour_overlap_density": 0.0,
            "h_proj_irregularity": 0.0,
            "v_proj_irregularity": 0.0,
            "h_proj_cv": 0.0,
            "v_proj_cv": 0.0,
        }

    if gray_region.ndim != 2:
        raise ValueError("extract_region_features expects a 2D grayscale region.")

    mean_intensity = float(np.mean(gray_region))
    std_intensity = float(np.std(gray_region))
    local_variance = compute_local_variance(gray_region)
    edge_density = compute_edge_density(gray_region)

    lap = cv2.Laplacian(gray_region, cv2.CV_32F, ksize=3)
    laplacian_variance = float(np.var(lap))

    entropy = compute_entropy(gray_region)

    adaptive = cv2.adaptiveThreshold(
        gray_region,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8,
    )
    stroke_density = float(np.count_nonzero(adaptive) / max(1, adaptive.size))

    cc_stats = compute_connected_components_stats(gray_region)
    proj_stats = compute_projection_features(gray_region)
    contour_overlap_density = _compute_contour_overlap_density(gray_region)

    features: Dict[str, Any] = {
        "valid": True,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "local_variance": local_variance,
        "edge_density": edge_density,
        "laplacian_variance": laplacian_variance,
        "entropy": entropy,
        "stroke_density": stroke_density,
        "contour_overlap_density": contour_overlap_density,
    }
    features.update(cc_stats)
    features.update(proj_stats)

    return normalize_feature_dict(features)
