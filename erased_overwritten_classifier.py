"""Day 6 suspicious-region classifier: erased_content vs overwritten_text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from region_features import extract_region_features
from texture_analysis import (
    detect_background_inconsistency_score,
    detect_blank_area_score,
    detect_contour_crowding_score,
    detect_smooth_patch_score,
    detect_stroke_overlap_score,
)

LOGGER = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]


@dataclass
class RegionClassifierConfig:
    """Configuration for erased/overwritten suspicious region classification."""

    crop_expand_px: int = 8
    min_region_area: int = 64
    min_width: int = 8
    min_height: int = 8
    erasure_score_threshold: float = 0.52
    overwrite_score_threshold: float = 0.52
    decision_margin: float = 0.05
    max_regions: int = 600


def expand_and_crop_region(
    image: np.ndarray,
    bbox: BoundingBox,
    expand_px: int = 8,
) -> tuple[BoundingBox, np.ndarray]:
    """Expand and crop a region with clipping to image bounds."""
    if image is None or image.size == 0:
        raise ValueError("expand_and_crop_region received empty image.")

    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    x1 = max(0, int(x) - expand_px)
    y1 = max(0, int(y) - expand_px)
    x2 = min(w, int(x + bw) + expand_px)
    y2 = min(h, int(y + bh) + expand_px)

    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0), image[0:0, 0:0]

    clipped = (x1, y1, x2 - x1, y2 - y1)
    crop = image[y1:y2, x1:x2]
    return clipped, crop


def summarize_region_features(features: Dict[str, Any]) -> Dict[str, float]:
    """Return compact feature summary useful for logging and debug export."""
    keys = [
        "mean_intensity",
        "std_intensity",
        "edge_density",
        "local_variance",
        "laplacian_variance",
        "entropy",
        "stroke_density",
        "contour_overlap_density",
        "fg_bg_ratio",
        "cc_count",
        "h_proj_irregularity",
        "v_proj_irregularity",
        "smooth_patch_score",
        "stroke_overlap_score",
        "background_inconsistency_score",
        "contour_crowding_score",
        "blank_area_score",
    ]
    summary: Dict[str, float] = {}
    for key in keys:
        value = features.get(key, 0.0)
        summary[key] = float(value) if isinstance(value, (int, float, np.floating)) else 0.0
    return summary


def compute_erasure_score(features: Dict[str, float], cfg: RegionClassifierConfig) -> float:
    """Compute confidence-like score for erased_content hypothesis."""
    smooth_score = float(features.get("smooth_patch_score", 0.0))
    blank_score = float(features.get("blank_area_score", 0.0))
    low_edge_score = 1.0 - float(np.clip(features.get("edge_density", 0.0) / 0.22, 0.0, 1.0))
    low_var_score = 1.0 - float(np.clip(features.get("norm_local_variance", 0.0), 0.0, 1.0))
    low_stroke_score = 1.0 - float(np.clip(features.get("stroke_density", 0.0) / 0.28, 0.0, 1.0))
    bg_inconsistency = float(features.get("background_inconsistency_score", 0.0))

    score = (
        0.25 * smooth_score
        + 0.21 * blank_score
        + 0.17 * low_edge_score
        + 0.15 * low_var_score
        + 0.12 * low_stroke_score
        + 0.10 * bg_inconsistency
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_overwrite_score(features: Dict[str, float], cfg: RegionClassifierConfig) -> float:
    """Compute confidence-like score for overwritten_text hypothesis."""
    stroke_overlap = float(features.get("stroke_overlap_score", 0.0))
    contour_crowding = float(features.get("contour_crowding_score", 0.0))
    edge_term = float(np.clip(features.get("edge_density", 0.0) / 0.30, 0.0, 1.0))
    stroke_term = float(np.clip(features.get("stroke_density", 0.0) / 0.36, 0.0, 1.0))
    contrast_term = float(np.clip(features.get("norm_std_intensity", 0.0), 0.0, 1.0))
    projection_term = float(np.clip(features.get("norm_proj_irregularity", 0.0), 0.0, 1.0))

    score = (
        0.28 * stroke_overlap
        + 0.22 * contour_crowding
        + 0.18 * edge_term
        + 0.15 * stroke_term
        + 0.10 * contrast_term
        + 0.07 * projection_term
    )
    return float(np.clip(score, 0.0, 1.0))


def classify_region(
    original_image: np.ndarray,
    preprocessed_gray: np.ndarray,
    bbox: BoundingBox,
    config: RegionClassifierConfig | None = None,
) -> Dict[str, Any] | None:
    """Classify one suspicious region as erased_content or overwritten_text."""
    cfg = config or RegionClassifierConfig()
    _, _, bw, bh = bbox
    area = int(max(0, bw) * max(0, bh))

    if bw < cfg.min_width or bh < cfg.min_height or area < cfg.min_region_area:
        LOGGER.debug("Skipping tiny region bbox=%s", bbox)
        return None

    clipped_bbox, gray_crop = expand_and_crop_region(
        preprocessed_gray,
        bbox,
        expand_px=cfg.crop_expand_px,
    )
    if gray_crop.size == 0:
        LOGGER.debug("Skipping empty clipped region bbox=%s", bbox)
        return None

    if gray_crop.ndim != 2:
        gray_crop = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)

    features = extract_region_features(gray_crop)
    if not features.get("valid", False):
        return None

    features["smooth_patch_score"] = detect_smooth_patch_score(gray_crop, features)
    features["stroke_overlap_score"] = detect_stroke_overlap_score(gray_crop, features)
    features["background_inconsistency_score"] = detect_background_inconsistency_score(gray_crop, features)
    features["contour_crowding_score"] = detect_contour_crowding_score(gray_crop, features)
    features["blank_area_score"] = detect_blank_area_score(gray_crop, features)

    erasure_score = compute_erasure_score(features, cfg)
    overwrite_score = compute_overwrite_score(features, cfg)

    margin = abs(overwrite_score - erasure_score)
    if overwrite_score >= erasure_score:
        label = "overwritten_text"
        score = overwrite_score
    else:
        label = "erased_content"
        score = erasure_score

    # If scores are weak and close, avoid overconfident relabeling.
    if margin < cfg.decision_margin:
        if max(erasure_score, overwrite_score) < min(cfg.erasure_score_threshold, cfg.overwrite_score_threshold):
            LOGGER.debug(
                "Low-confidence region kept generic bbox=%s erase=%.3f overwrite=%.3f",
                clipped_bbox,
                erasure_score,
                overwrite_score,
            )
            return {
                "bbox": clipped_bbox,
                "label": "suspicious_region",
                "score": float(max(erasure_score, overwrite_score)),
                "source": "region_classifier",
                "features": summarize_region_features(features),
                "erasure_score": erasure_score,
                "overwrite_score": overwrite_score,
            }

    if label == "erased_content" and erasure_score < cfg.erasure_score_threshold:
        label = "suspicious_region"
    if label == "overwritten_text" and overwrite_score < cfg.overwrite_score_threshold:
        label = "suspicious_region"

    summary = summarize_region_features(features)

    LOGGER.info(
        "Region classified bbox=%s label=%s score=%.3f erase=%.3f overwrite=%.3f edge=%.3f stroke=%.3f",
        clipped_bbox,
        label,
        score,
        erasure_score,
        overwrite_score,
        summary.get("edge_density", 0.0),
        summary.get("stroke_density", 0.0),
    )

    return {
        "bbox": clipped_bbox,
        "label": label,
        "score": float(score),
        "source": "region_classifier",
        "features": summary,
        "erasure_score": float(erasure_score),
        "overwrite_score": float(overwrite_score),
    }


def classify_suspicious_regions(
    original_image: np.ndarray,
    preprocessed_gray: np.ndarray,
    candidate_boxes: Sequence[BoundingBox],
    config: RegionClassifierConfig | None = None,
) -> List[Dict[str, Any]]:
    """Classify candidate suspicious boxes into erased/overwritten labels."""
    cfg = config or RegionClassifierConfig()

    if original_image is None or original_image.size == 0:
        raise ValueError("classify_suspicious_regions received empty original_image")
    if preprocessed_gray is None or preprocessed_gray.size == 0:
        raise ValueError("classify_suspicious_regions received empty preprocessed_gray")

    if not candidate_boxes:
        LOGGER.info("No candidate boxes provided for region classification.")
        return []

    results: List[Dict[str, Any]] = []
    for idx, box in enumerate(candidate_boxes[: cfg.max_regions]):
        try:
            classified = classify_region(original_image, preprocessed_gray, box, config=cfg)
            if classified is not None:
                results.append(classified)
        except Exception as exc:
            LOGGER.exception("Failed to classify candidate region %d bbox=%s: %s", idx, box, exc)

    LOGGER.info(
        "Region classifier processed=%d valid_results=%d",
        min(len(candidate_boxes), cfg.max_regions),
        len(results),
    )
    return results
