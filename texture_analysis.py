"""Texture-level helpers for distinguishing erasure vs overwrite tampering."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def detect_smooth_patch_score(gray_region: np.ndarray, features: Dict[str, float]) -> float:
    """Return a higher score for smoother inpainted/erased-like regions."""
    if gray_region.size == 0:
        return 0.0

    std_term = 1.0 - _clip01(features.get("norm_std_intensity", 0.0))
    edge_term = 1.0 - _clip01(features.get("edge_density", 0.0) / 0.22)
    lap_term = 1.0 - _clip01(features.get("norm_laplacian_variance", 0.0))
    var_term = 1.0 - _clip01(features.get("norm_local_variance", 0.0))

    smooth_score = 0.28 * std_term + 0.28 * edge_term + 0.22 * lap_term + 0.22 * var_term
    return _clip01(smooth_score)


def detect_stroke_overlap_score(gray_region: np.ndarray, features: Dict[str, float]) -> float:
    """Return higher score for stacked/overwritten stroke patterns."""
    if gray_region.size == 0:
        return 0.0

    edge_term = _clip01(features.get("edge_density", 0.0) / 0.28)
    stroke_term = _clip01(features.get("stroke_density", 0.0) / 0.35)
    overlap_term = _clip01(features.get("contour_overlap_density", 0.0) / 0.18)
    proj_term = _clip01(features.get("norm_proj_irregularity", 0.0))

    return _clip01(0.30 * edge_term + 0.30 * stroke_term + 0.25 * overlap_term + 0.15 * proj_term)


def detect_background_inconsistency_score(gray_region: np.ndarray, features: Dict[str, float]) -> float:
    """Detect uneven patch blending and local illumination inconsistency."""
    if gray_region.size == 0:
        return 0.0

    region_f = gray_region.astype(np.float32)
    large_bg = cv2.GaussianBlur(region_f, (0, 0), sigmaX=7.0)
    residual = np.abs(region_f - large_bg)

    residual_mean = float(np.mean(residual))
    residual_std = float(np.std(residual))

    contrast_term = _clip01((features.get("std_intensity", 0.0) + residual_std) / 80.0)
    residual_term = _clip01(residual_mean / 24.0)

    return _clip01(0.55 * residual_term + 0.45 * contrast_term)


def detect_contour_crowding_score(gray_region: np.ndarray, features: Dict[str, float]) -> float:
    """Detect contour crowding typical of overwritten text/marks."""
    if gray_region.size == 0:
        return 0.0

    cc_term = _clip01(features.get("norm_cc_count", 0.0))
    small_cc_term = _clip01(features.get("small_cc_ratio", 0.0))
    overlap_term = _clip01(features.get("contour_overlap_density", 0.0) / 0.2)

    return _clip01(0.35 * cc_term + 0.30 * small_cc_term + 0.35 * overlap_term)


def detect_blank_area_score(gray_region: np.ndarray, features: Dict[str, float]) -> float:
    """Detect suspicious blanked-out areas while suppressing normal white margins."""
    if gray_region.size == 0:
        return 0.0

    white_ratio = float(np.mean(gray_region >= 245))
    edge_density = float(features.get("edge_density", 0.0))
    fg_ratio = float(features.get("fg_bg_ratio", 0.0))

    # Margins tend to be uniformly blank with very low local inconsistency,
    # so include background inconsistency via projection CV to reduce false positives.
    proj_cv = 0.5 * float(features.get("h_proj_cv", 0.0)) + 0.5 * float(features.get("v_proj_cv", 0.0))

    blankness = _clip01((white_ratio - 0.55) / 0.45)
    low_edge = 1.0 - _clip01(edge_density / 0.15)
    low_fg = 1.0 - _clip01(fg_ratio / 0.22)
    structure_penalty = _clip01(proj_cv / 2.0)

    return _clip01(0.38 * blankness + 0.27 * low_edge + 0.20 * low_fg + 0.15 * structure_penalty)
