"""Region-level AI editing detection module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np

from ai_artifact_features import extract_region_ai_features, normalize_feature_vector

LOGGER = logging.getLogger(__name__)


@dataclass
class RegionAIEditConfig:
    """Configuration for region-level AI-edit detector."""

    ai_edit_region_threshold: float = 0.65
    min_region_area: int = 100
    context_margin: int = 20
    use_context_comparison: bool = True


def compute_context_contrast(
    gray_region: np.ndarray,
    gray_context: np.ndarray,
) -> float:
    """Compare AI-edit candidate region to surrounding context.

    Args:
        gray_region: Candidate patch (e.g., 64x64).
        gray_context: Surrounding context region.

    Returns:
        Anomaly score [0, 1] indicating deviation from context.
    """
    if gray_context is None or gray_context.size == 0:
        return 0.0

    # Compute statistics
    region_mean = np.mean(gray_region)
    context_mean = np.mean(gray_context)
    mean_diff = abs(region_mean - context_mean)

    region_std = np.std(gray_region)
    context_std = np.std(gray_context)
    std_ratio = region_std / max(1e-6, context_std)

    # Texture comparison: entropy difference
    region_hist, _ = np.histogram(gray_region, bins=32, range=(0, 256))
    region_hist = region_hist / max(1, np.sum(region_hist))
    region_entropy = -np.sum(region_hist[region_hist > 0] * np.log2(region_hist[region_hist > 0]))

    context_hist, _ = np.histogram(gray_context, bins=32, range=(0, 256))
    context_hist = context_hist / max(1, np.sum(context_hist))
    context_entropy = -np.sum(context_hist[context_hist > 0] * np.log2(context_hist[context_hist > 0]))

    entropy_diff = abs(region_entropy - context_entropy)

    # Combine into contrast score
    contrast_score = (
        0.40 * np.clip(mean_diff / 50.0, 0.0, 1.0) +
        0.35 * np.clip(abs(std_ratio - 1.0), 0.0, 1.0) +
        0.25 * np.clip(entropy_diff / 4.0, 0.0, 1.0)
    )

    return float(np.clip(contrast_score, 0.0, 1.0))


def compute_ai_edit_score(
    region_features: Dict[str, float],
    context_contrast: float = 0.0,
    cfg: RegionAIEditConfig | None = None,
) -> float:
    """Compute region-level AI-edit suspicion score.

    Args:
        region_features: Normalized features from suspicious region.
        context_contrast: Anomaly vs context (0 = indistinguishable).
        cfg: Configuration.

    Returns:
        AI-edit suspicion score [0, 1].
    """
    cfg = cfg or RegionAIEditConfig()

    # AI-edited regions often exhibit:
    # 1. High-frequency artifacts (blending boundary)
    # 2. Inconsistent noise patterns
    # 3. Unnatural sharpness
    # 4. Regular texture patterns (copy-paste or generation)

    score_components = []

    fft_ratio = float(region_features.get("norm_fft_energy_ratio", 0.0))
    score_components.append(("fft_boundary_artifact", fft_ratio, 0.18))

    noise_cv = float(region_features.get("norm_noise_cv", 0.0))
    noise_score = abs(1.0 - noise_cv)  # High CV or very low both suspect
    score_components.append(("noise_inconsistency", noise_score, 0.15))

    edge_sharpness = float(region_features.get("norm_edge_sharpness", 0.0))
    sharpness_anomaly = abs(0.3 - edge_sharpness)  # Extremes are suspicious
    score_components.append(("sharpness_anomaly", sharpness_anomaly, 0.12))

    lbp_entropy = float(region_features.get("norm_lbp_entropy", 0.0))
    lbp_score = 1.0 - lbp_entropy  # Regular texture
    score_components.append(("texture_regularity", lbp_score, 0.15))

    # Gradient consistency: synthetic edits have inconsistent gradients
    grad_consistency = float(region_features.get("norm_gradient_consistency", 0.0))
    score_components.append(("gradient_inconsistency", grad_consistency, 0.12))

    # Character boundary smoothness: AI edits may have unnatural boundaries
    char_smooth = float(region_features.get("character_boundary_smoothness", 0.0))
    char_score = 1.0 - char_smooth if char_smooth > 0 else 0.0
    score_components.append(("boundary_artifact", char_score, 0.10))

    # Context contrast: deviation from surroundings
    score_components.append(("context_contrast", float(context_contrast), 0.18))

    total_score = 0.0
    total_weight = 0.0
    for _, value, weight in score_components:
        total_score += weight * float(np.clip(value, 0.0, 1.0))
        total_weight += weight

    final_score = total_score / max(1e-6, total_weight)
    return float(np.clip(final_score, 0.0, 1.0))


def classify_region_as_ai_edited(
    gray_region: np.ndarray,
    gray_context: np.ndarray | None = None,
    config: RegionAIEditConfig | None = None,
) -> Dict[str, Any]:
    """Classify a region as AI-edited or not.

    Args:
        gray_region: Candidate region crop.
        gray_context: Surrounding context for comparison.
        config: Classifier configuration.

    Returns:
        {
            "label": "ai_edited_region" or "not_ai_edited",
            "score": float,
            "context_contrast": float,
            "features": {...}
        }
    """
    cfg = config or RegionAIEditConfig()

    if gray_region is None or gray_region.size == 0:
        raise ValueError("classify_region_as_ai_edited received empty region.")

    # Extract features
    features = extract_region_ai_features(gray_region)

    if not features.get("valid", False):
        LOGGER.warning("Failed to extract region AI features.")
        return {
            "label": "not_ai_edited",
            "score": 0.0,
            "context_contrast": 0.0,
            "features": features,
        }

    features = normalize_feature_vector(features)

    # Compute context contrast if available
    context_contrast = 0.0
    if cfg.use_context_comparison and gray_context is not None:
        context_contrast = compute_context_contrast(gray_region, gray_context)

    # Compute AI-edit score
    score = compute_ai_edit_score(features, context_contrast, cfg)

    # Classify
    if score >= cfg.ai_edit_region_threshold:
        label = "ai_edited_region"
    else:
        label = "not_ai_edited"

    return {
        "label": label,
        "score": float(score),
        "context_contrast": float(context_contrast),
        "features": features,
    }


def detect_ai_edited_regions_from_boxes(
    gray_image: np.ndarray,
    suspicious_boxes: list[tuple[int, int, int, int]],
    config: RegionAIEditConfig | None = None,
) -> list[Dict[str, Any]]:
    """Classify each suspicious box for AI-editing signs.

    Args:
        gray_image: Full grayscale page.
        suspicious_boxes: List of (x, y, w, h) bounding boxes.
        config: Classifier configuration.

    Returns:
        List of classification results matching box order.
    """
    cfg = config or RegionAIEditConfig()
    results = []

    for box_idx, (x, y, w, h) in enumerate(suspicious_boxes):
        if w < 10 or h < 10:
            results.append({
                "box_index": box_idx,
                "label": "not_ai_edited",
                "score": 0.0,
                "context_contrast": 0.0,
                "skip_reason": "box_too_small",
            })
            continue

        # Extract region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray_image.shape[1], x + w)
        y2 = min(gray_image.shape[0], y + h)

        gray_region = gray_image[y1:y2, x1:x2]

        if gray_region.size < cfg.min_region_area:
            results.append({
                "box_index": box_idx,
                "label": "not_ai_edited",
                "score": 0.0,
                "context_contrast": 0.0,
                "skip_reason": "too_small",
            })
            continue

        # Extract context (surrounding region)
        ctx_x1 = max(0, x - cfg.context_margin)
        ctx_y1 = max(0, y - cfg.context_margin)
        ctx_x2 = min(gray_image.shape[1], x + w + cfg.context_margin)
        ctx_y2 = min(gray_image.shape[0], y + h + cfg.context_margin)

        gray_context = gray_image[ctx_y1:ctx_y2, ctx_x1:ctx_x2]

        # Classify
        classification = classify_region_as_ai_edited(gray_region, gray_context, cfg)
        classification["box_index"] = box_idx
        classification["bbox"] = (x1, y1, x2, y2)

        results.append(classification)

    return results
