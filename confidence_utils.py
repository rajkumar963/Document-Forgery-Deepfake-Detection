"""Utilities for score calibration and confidence refinement across modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfidenceConfig:
    """Configuration for confidence score processing."""

    # Module-specific weight multipliers (used to normalize different scales)
    module_weights: Dict[str, float] = None
    
    # Agreement boost factor when multiple modules support same region
    agreement_boost_factor: float = 1.15
    
    # Penalty for weak isolated detections
    weak_isolated_penalty: float = 0.75
    
    # Minimum score to avoid penalties
    min_score_threshold: float = 0.50
    
    # Per-label score thresholds (any score below is suppressed)
    label_thresholds: Dict[str, float] = None

    def __post_init__(self):
        """Initialize defaults."""
        if self.module_weights is None:
            self.module_weights = {
                "anomaly": 1.0,
                "copy_move": 1.1,
                "ocr_spacing": 0.95,
                "added_content": 1.2,
                "region_classifier": 1.05,
                "ai_region_detector": 1.0,
                "ai_document_detector": 1.0,
                "final_refiner": 1.0,
            }
        
        if self.label_thresholds is None:
            self.label_thresholds = {
                "copy_paste": 0.70,
                "stamp": 0.60,
                "signature": 0.60,
                "seal": 0.60,
                "added_mark": 0.55,
                "overwritten_text": 0.50,
                "erased_content": 0.45,
                "ai_edited_region": 0.55,
                "irregular_spacing": 0.50,
                "suspicious_region": 0.40,
            }


def calibrate_score(
    raw_score: float,
    source: str,
    config: ConfidenceConfig | None = None,
) -> float:
    """Calibrate raw score using module-specific weights.

    Args:
        raw_score: Original confidence score [0, 1].
        source: Detection source module name.
        config: Confidence configuration.

    Returns:
        Calibrated score [0, 1].
    """
    config = config or ConfidenceConfig()
    weight = config.module_weights.get(source, 1.0)
    calibrated = float(np.clip(raw_score * weight, 0.0, 1.0))
    return calibrated


def combine_scores(
    scores: List[float],
    strategy: str = "geometric_mean",
) -> float:
    """Combine multiple scores into a single consensus score.

    Args:
        scores: List of confidence scores.
        strategy: Combination strategy: "mean", "max", "geometric_mean", "harmonic_mean".

    Returns:
        Combined score [0, 1].
    """
    if not scores:
        return 0.0

    scores_arr = np.array([float(s) for s in scores])
    scores_arr = np.clip(scores_arr, 0.0, 1.0)

    if strategy == "mean":
        combined = float(np.mean(scores_arr))
    elif strategy == "max":
        combined = float(np.max(scores_arr))
    elif strategy == "geometric_mean":
        # Geometric mean is less sensitive to outliers than arithmetic mean
        combined = float(np.exp(np.mean(np.log(np.maximum(scores_arr, 1e-6)))))
    elif strategy == "harmonic_mean":
        # Harmonic mean emphasizes agreement
        combined = float(len(scores_arr) / np.sum(1.0 / np.maximum(scores_arr, 1e-6)))
    else:
        combined = float(np.mean(scores_arr))

    return float(np.clip(combined, 0.0, 1.0))


def agreement_boost(
    base_score: float,
    supporting_count: int,
    config: ConfidenceConfig | None = None,
) -> float:
    """Boost score when multiple modules agree on same detection.

    Args:
        base_score: Baseline confidence score.
        supporting_count: Number of supporting sources (including base).
        config: Confidence configuration.

    Returns:
        Boosted score [0, 1].
    """
    config = config or ConfidenceConfig()

    if supporting_count < 2:
        return base_score

    # Boost factor increases with support (diminishing returns)
    support_boost = (config.agreement_boost_factor - 1.0) * min(1.0, (supporting_count - 1) / 3.0)
    boosted = base_score * (1.0 + support_boost)

    return float(np.clip(boosted, 0.0, 1.0))


def penalty_for_isolation(
    base_score: float,
    supporting_count: int,
    label: str,
    config: ConfidenceConfig | None = None,
) -> float:
    """Apply penalty to weak isolated detections.

    Args:
        base_score: Baseline confidence score.
        supporting_count: Number of supporting sources.
        label: Detection label.
        config: Confidence configuration.

    Returns:
        Score after penalty [0, 1].
    """
    config = config or ConfidenceConfig()

    # Generic "suspicious_region" with single weak support should be penalized
    if label == "suspicious_region" and supporting_count == 1 and base_score < config.min_score_threshold:
        penalized = base_score * config.weak_isolated_penalty
        return float(np.clip(penalized, 0.0, 1.0))

    return base_score


def is_score_above_threshold(
    score: float,
    label: str,
    config: ConfidenceConfig | None = None,
) -> bool:
    """Check if score exceeds label-specific threshold.

    Args:
        score: Confidence score.
        label: Detection label.
        config: Confidence configuration.

    Returns:
        True if score exceeds threshold for this label.
    """
    config = config or ConfidenceConfig()
    threshold = config.label_thresholds.get(label, 0.50)
    return score >= threshold


def compute_final_confidence(
    detections_list: List[Dict],
    config: ConfidenceConfig | None = None,
) -> Dict[str, float]:
    """Compute final calibrated confidence for a group of detections.

    Args:
        detections_list: Group of overlapping detections.
        config: Confidence configuration.

    Returns:
        {
            "final_score": float,
            "calibrated_scores": [float],
            "agreement_boost": float,
            "isolation_penalty": float,
        }
    """
    config = config or ConfidenceConfig()

    if not detections_list:
        return {
            "final_score": 0.0,
            "calibrated_scores": [],
            "agreement_boost": 0.0,
            "isolation_penalty": 0.0,
        }

    # Calibrate each score
    calibrated_scores = []
    for det in detections_list:
        score = float(det.get("score", 0.0))
        source = str(det.get("source", "unknown"))
        cal_score = calibrate_score(score, source, config)
        calibrated_scores.append(cal_score)

    # Combine calibrated scores (use geometric mean for robustness)
    combined = combine_scores(calibrated_scores, strategy="geometric_mean")

    # Apply agreement boost
    agreement = agreement_boost(combined, len(detections_list), config)
    boosted = combined * (agreement / combined) if combined > 0 else combined

    # Apply isolation penalty if needed
    label = str(detections_list[0].get("label", "suspicious_region"))
    penalized = penalty_for_isolation(boosted, len(detections_list), label, config)

    return {
        "final_score": float(np.clip(penalized, 0.0, 1.0)),
        "calibrated_scores": [float(s) for s in calibrated_scores],
        "agreement_boost": float(agreement - combined) if combined > 0 else 0.0,
        "isolation_penalty": float(boosted - penalized) if boosted > penalized else 0.0,
    }


def filter_detections_by_score(
    detections: List[Dict],
    config: ConfidenceConfig | None = None,
) -> List[Dict]:
    """Filter detections below label-specific thresholds.

    Args:
        detections: List of detections.
        config: Confidence configuration.

    Returns:
        Filtered list of detections above threshold.
    """
    config = config or ConfidenceConfig()
    filtered = []

    for det in detections:
        score = float(det.get("score", 0.0))
        label = str(det.get("label", "suspicious_region"))

        if is_score_above_threshold(score, label, config):
            filtered.append(det)
        else:
            LOGGER.debug(
                "Filtering low-score detection: label=%s score=%.3f threshold=%.3f source=%s",
                label,
                score,
                config.label_thresholds.get(label, 0.50),
                det.get("source", "unknown"),
            )

    return filtered


def recalibrate_detection_group(
    detection_group: List[Dict],
    final_label: str,
    config: ConfidenceConfig | None = None,
) -> Dict:
    """Recalibrate and finalize a group of detections with agreed label.

    Args:
        detection_group: List of related detections.
        final_label: Agreed final label for group.
        config: Confidence configuration.

    Returns:
        Finalized detection record with calibrated score.
    """
    config = config or ConfidenceConfig()

    confidence_result = compute_final_confidence(detection_group, config)
    final_score = confidence_result["final_score"]

    # Collect source modules
    sources = list(set(str(d.get("source", "")) for d in detection_group if d.get("source")))

    return {
        "label": final_label,
        "score": final_score,
        "supporting_sources": sources,
        "calibrated_scores": confidence_result["calibrated_scores"],
        "agreement_boost": confidence_result["agreement_boost"],
        "isolation_penalty": confidence_result["isolation_penalty"],
        "group_size": len(detection_group),
    }
