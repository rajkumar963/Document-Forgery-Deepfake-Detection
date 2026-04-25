"""Whole-document AI-generation detection module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ai_artifact_features import extract_document_ai_features, normalize_feature_vector

LOGGER = logging.getLogger(__name__)


@dataclass
class DocumentClassifierConfig:
    """Configuration for document-level AI-generation classifier."""

    ai_generated_score_threshold: float = 0.70
    enable_shallow_model: bool = False
    model_path: str | None = None
    use_margin_uniformity: bool = True
    use_fft_analysis: bool = True
    use_noise_consistency: bool = True


def _load_shallow_model(model_path: str | Path) -> Any:
    """Load optional shallow classifier model (e.g., sklearn pickle).

    Returns None if model does not exist or fails to load.
    """
    try:
        import pickle

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            LOGGER.warning("Shallow model not found at: %s", path)
            return None

        with open(path, "rb") as f:
            model = pickle.load(f)
            LOGGER.info("Loaded shallow classifier from: %s", path)
            return model
    except Exception as exc:
        LOGGER.warning("Failed to load shallow model: %s", exc)
        return None


def compute_ai_generated_score(
    features: Dict[str, float],
    cfg: DocumentClassifierConfig,
) -> float:
    """Compute document-level AI-generation suspicion score.

    Uses handcrafted features and weighted combination.
    """
    # Synthetic documents often have:
    # 1. High-frequency energy (digital artifacts)
    # 2. Low noise variation (unnaturally clean)
    # 3. High texture uniformity (repeated patterns)
    # 4. Uniform margins
    # 5. Uniform edge direction

    score_components = []

    if cfg.use_fft_analysis:
        fft_ratio = float(features.get("norm_fft_energy_ratio", 0.0))
        score_components.append(("fft_high_freq", fft_ratio, 0.20))

    if cfg.use_noise_consistency:
        noise_cv = float(features.get("norm_noise_cv", 0.0))
        # Low CV suggests synthetic uniformity
        noise_score = 1.0 - noise_cv
        score_components.append(("noise_uniformity", noise_score, 0.18))

    # Texture regularity: synthetic has high uniformity
    lbp_entropy = float(features.get("norm_lbp_entropy", 0.0))
    # Lower entropy = more regular = more synthetic
    texture_score = 1.0 - lbp_entropy
    score_components.append(("texture_regularity", texture_score, 0.18))

    # Edge sharpness consistency
    edge_sharpness = float(features.get("norm_edge_sharpness", 0.0))
    score_components.append(("edge_sharpness", edge_sharpness, 0.12))

    # Laplacian variance: synthetic text often has low variance
    lap_var = float(features.get("norm_laplacian_variance", 0.0))
    # Lower = more uniform = more synthetic
    lap_score = 1.0 - lap_var
    score_components.append(("laplacian_uniformity", lap_score, 0.15))

    if cfg.use_margin_uniformity:
        margin_uni = float(features.get("norm_margin_uniformity", 0.0))
        score_components.append(("margin_uniformity", margin_uni, 0.10))

    # Directional bias: real scans have scanning artifacts
    dir_bias = float(features.get("norm_directional_edge_bias", 0.0))
    # Balanced (near 1.0) suggests synthetic; biased suggests real scan
    dir_score = 1.0 - abs(1.0 - min(3.0, dir_bias)) / 3.0
    score_components.append(("directional_uniformity", dir_score, 0.07))

    total_score = 0.0
    total_weight = 0.0
    for name, value, weight in score_components:
        total_score += weight * float(np.clip(value, 0.0, 1.0))
        total_weight += weight

    final_score = total_score / max(1e-6, total_weight)
    return float(np.clip(final_score, 0.0, 1.0))


def classify_document_from_features(
    features: Dict[str, float],
    cfg: DocumentClassifierConfig,
    shallow_model: Any = None,
) -> tuple[str, float]:
    """Classify document as fully_ai_generated_document or authentic_or_unknown.

    Args:
        features: Normalized document-level features.
        cfg: Classifier configuration.
        shallow_model: Optional pre-trained shallow classifier.

    Returns:
        (label, score) tuple.
    """
    if shallow_model is not None:
        try:
            feature_names = [
                "fft_high_freq_energy",
                "fft_low_freq_energy",
                "fft_energy_ratio",
                "noise_std",
                "noise_cv",
                "local_noise_variance",
                "noise_homogeneity",
                "laplacian_variance",
                "edge_sharpness",
                "gradient_consistency",
                "sobel_magnitude_mean",
                "texture_entropy",
                "texture_uniformity",
                "autocorr_peak",
                "lbp_entropy",
                "lbp_uniformity",
                "character_boundary_smoothness",
                "interior_stroke_regularity",
                "directional_edge_bias",
                "margin_uniformity",
            ]
            feature_vec = np.array([features.get(name, 0.0) for name in feature_names]).reshape(1, -1)

            if hasattr(shallow_model, "predict_proba"):
                proba = shallow_model.predict_proba(feature_vec)[0]
                ai_gen_prob = float(proba[1]) if len(proba) > 1 else 0.0
            elif hasattr(shallow_model, "decision_function"):
                raw_score = shallow_model.decision_function(feature_vec)[0]
                ai_gen_prob = float(1.0 / (1.0 + np.exp(-raw_score)))
            else:
                ai_gen_prob = 0.0

            if ai_gen_prob >= cfg.ai_generated_score_threshold:
                label = "fully_ai_generated_document"
            else:
                label = "authentic_or_unknown"

            return label, ai_gen_prob
        except Exception as exc:
            LOGGER.warning("Shallow model inference failed: %s, falling back to rule-based", exc)

    # Fallback: rule-based scoring
    score = compute_ai_generated_score(features, cfg)

    if score >= cfg.ai_generated_score_threshold:
        label = "fully_ai_generated_document"
    else:
        label = "authentic_or_unknown"

    return label, score


def detect_fully_ai_generated_document(
    preprocessed_gray: np.ndarray,
    config: DocumentClassifierConfig | None = None,
) -> Dict[str, Any]:
    """Classify entire document as AI-generated or authentic.

    Args:
        preprocessed_gray: Full-page grayscale image.
        config: Classifier configuration.

    Returns:
        {
            "label": str,
            "score": float,
            "features": {...}
        }
    """
    cfg = config or DocumentClassifierConfig()

    if preprocessed_gray is None or preprocessed_gray.size == 0:
        raise ValueError("detect_fully_ai_generated_document received empty image.")

    LOGGER.info("Analyzing document for AI-generation signs...")
    features = extract_document_ai_features(preprocessed_gray)

    if not features.get("valid", False):
        LOGGER.warning("Failed to extract document AI features. Returning uncertain result.")
        return {
            "label": "authentic_or_unknown",
            "score": 0.0,
            "features": features,
        }

    features = normalize_feature_vector(features)

    shallow_model = None
    if cfg.enable_shallow_model and cfg.model_path:
        shallow_model = _load_shallow_model(cfg.model_path)

    label, score = classify_document_from_features(features, cfg, shallow_model)

    LOGGER.info("Document classification: label=%s score=%.3f", label, score)

    return {
        "label": label,
        "score": float(score),
        "features": features,
    }
