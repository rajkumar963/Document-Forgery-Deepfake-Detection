"""Feature extraction for detecting AI-generated or AI-edited document artifacts."""

from __future__ import annotations

import logging
from typing import Any, Dict

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

LOGGER = logging.getLogger(__name__)


def compute_fft_features(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute frequency-domain features using FFT.

    Higher frequency components suggest digital/synthetic content.
    """
    if gray_region.size == 0:
        return {
            "fft_high_freq_energy": 0.0,
            "fft_low_freq_energy": 0.0,
            "fft_energy_ratio": 0.0,
        }

    # Compute 2D FFT and magnitude spectrum
    f_transform = np.fft.fft2(gray_region.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    # Split into low and high frequency components based on distance from center
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    radius_low = min(h, w) // 4
    low_freq_mask = dist <= radius_low
    high_freq_mask = dist > radius_low

    low_energy = float(np.sum(magnitude[low_freq_mask] ** 2))
    high_energy = float(np.sum(magnitude[high_freq_mask] ** 2))
    total_energy = low_energy + high_energy

    ratio = high_energy / max(1e-6, total_energy)

    return {
        "fft_high_freq_energy": high_energy,
        "fft_low_freq_energy": low_energy,
        "fft_energy_ratio": ratio,
    }


def compute_noise_statistics(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute noise and texture consistency metrics.

    AI-generated or edited patches often have unnaturally uniform noise.
    """
    if gray_region.size == 0:
        return {
            "noise_std": 0.0,
            "noise_cv": 0.0,
            "local_noise_variance": 0.0,
            "noise_homogeneity": 0.0,
        }

    # Estimate noise via Laplacian
    lap = cv2.Laplacian(gray_region, cv2.CV_32F, ksize=3)
    noise_std = float(np.std(lap))
    noise_mean = float(np.abs(np.mean(lap)))
    noise_cv = noise_std / max(1e-6, noise_mean)

    # Local noise variance
    k = 7
    region_f = gray_region.astype(np.float32)
    mean = cv2.GaussianBlur(region_f, (k, k), 0)
    mean_sq = cv2.GaussianBlur(region_f * region_f, (k, k), 0)
    local_var = np.maximum(0.0, mean_sq - mean * mean)
    local_noise_variance = float(np.mean(local_var))

    # Homogeneity: low variance across space suggests synthetic uniformity
    homogeneity = float(np.std(local_var)) / max(1e-6, np.mean(local_var))

    return {
        "noise_std": noise_std,
        "noise_cv": noise_cv,
        "local_noise_variance": local_noise_variance,
        "noise_homogeneity": homogeneity,
    }


def compute_sharpness_metrics(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute edge sharpness and gradient consistency metrics.

    AI-generated text may have overly uniform or unnatural sharpness.
    """
    if gray_region.size == 0:
        return {
            "laplacian_variance": 0.0,
            "edge_sharpness": 0.0,
            "gradient_consistency": 0.0,
            "sobel_magnitude_mean": 0.0,
        }

    lap = cv2.Laplacian(gray_region, cv2.CV_32F, ksize=3)
    laplacian_var = float(np.var(lap))

    # Canny edges as sharpness indicator
    edges = cv2.Canny(gray_region, 40, 120)
    edge_sharpness = float(np.count_nonzero(edges) / max(1, edges.size))

    # Sobel gradients
    sobelx = cv2.Sobel(gray_region, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_region, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_mean = float(np.mean(magnitude))

    # Consistency: low variance in gradient suggests synthetic uniformity
    gradient_consistency = float(np.std(magnitude) / max(1e-6, np.mean(magnitude)))

    return {
        "laplacian_variance": laplacian_var,
        "edge_sharpness": edge_sharpness,
        "gradient_consistency": gradient_consistency,
        "sobel_magnitude_mean": sobel_mean,
    }


def compute_texture_regularity(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute texture regularity and repetition scores.

    Synthetic patches may have repeating textures or unnaturally regular patterns.
    """
    if gray_region.size == 0:
        return {
            "texture_entropy": 0.0,
            "texture_uniformity": 0.0,
            "autocorr_peak": 0.0,
        }

    # Entropy: lower entropy suggests more regular/synthetic texture
    hist, _ = np.histogram(gray_region, bins=256, range=(0, 256))
    hist = hist / max(1, np.sum(hist))
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    # Uniformity: fraction of most common intensity values
    max_freq = float(np.max(hist))
    uniformity = max_freq

    # Simple autocorrelation to detect repetition
    gray_norm = (gray_region.astype(np.float32) - np.mean(gray_region)) / max(1e-6, np.std(gray_region))
    if gray_region.size >= 16:
        shift = 8
        tile_height = gray_norm.shape[0] - shift
        tile_width = gray_norm.shape[1] - shift
        tile_size = min(tile_height, tile_width)
        if tile_size > 0:
            tile1 = gray_norm[:tile_size, :tile_size]
            tile2 = gray_norm[shift : shift + tile_size, shift : shift + tile_size]
            autocorr = float(np.mean(tile1 * tile2))
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    return {
        "texture_entropy": float(entropy),
        "texture_uniformity": uniformity,
        "autocorr_peak": autocorr,
    }


def compute_lbp_histogram(gray_region: np.ndarray, n_bins: int = 32) -> Dict[str, float]:
    """Compute Local Binary Pattern histogram.

    LBP captures local texture patterns useful for spotting synthetic uniformity.
    """
    if gray_region.size == 0:
        return {"lbp_entropy": 0.0, "lbp_uniformity": 0.0}

    try:
        lbp = local_binary_pattern(gray_region.astype(np.uint8), P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hist = hist / max(1, np.sum(hist))

        # Entropy of LBP distribution
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))

        # Uniformity
        max_freq = float(np.max(hist))

        return {
            "lbp_entropy": float(entropy),
            "lbp_uniformity": max_freq,
        }
    except Exception as exc:
        LOGGER.warning("LBP computation failed: %s", exc)
        return {"lbp_entropy": 0.0, "lbp_uniformity": 0.0}


def compute_ocr_consistency_features(gray_region: np.ndarray) -> Dict[str, float]:
    """Compute OCR-related consistency features.

    Synthetic text may have unusual patterns in character rendering.
    """
    if gray_region.size == 0:
        return {
            "character_boundary_smoothness": 0.0,
            "interior_stroke_regularity": 0.0,
        }

    # Binary threshold to isolate text
    _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if not np.any(binary):
        return {
            "character_boundary_smoothness": 0.0,
            "interior_stroke_regularity": 0.0,
        }

    # Character boundary smoothness: ratio of edge pixels to foreground
    edges = cv2.Canny(binary, 40, 120)
    fg_pixels = np.count_nonzero(binary)
    edge_pixels = np.count_nonzero(edges)
    smoothness = edge_pixels / max(1, fg_pixels)

    # Interior stroke regularity: consistency of stroke widths via medial axis
    # Approximated by dilate-erode difference
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    thickness_var = float(np.std(dilated.astype(float) - eroded.astype(float)))

    return {
        "character_boundary_smoothness": float(smoothness),
        "interior_stroke_regularity": thickness_var,
    }


def extract_region_ai_features(gray_region: np.ndarray) -> Dict[str, Any]:
    """Extract AI-artifact features from a region crop.

    Args:
        gray_region: 2D grayscale crop.

    Returns:
        Dictionary of numeric features for AI-edit/generation detection.
    """
    if gray_region is None or gray_region.size == 0:
        LOGGER.warning("extract_region_ai_features received empty crop.")
        return {"valid": False}

    if gray_region.ndim != 2:
        raise ValueError("extract_region_ai_features expects 2D grayscale image.")

    features: Dict[str, Any] = {"valid": True}

    features.update(compute_fft_features(gray_region))
    features.update(compute_noise_statistics(gray_region))
    features.update(compute_sharpness_metrics(gray_region))
    features.update(compute_texture_regularity(gray_region))
    features.update(compute_lbp_histogram(gray_region))
    features.update(compute_ocr_consistency_features(gray_region))

    return features


def extract_document_ai_features(gray_image: np.ndarray) -> Dict[str, Any]:
    """Extract document-level AI-artifact features from full page.

    Args:
        gray_image: Full-page 2D grayscale image.

    Returns:
        Dictionary of document-level features.
    """
    if gray_image is None or gray_image.size == 0:
        LOGGER.warning("extract_document_ai_features received empty image.")
        return {"valid": False}

    if gray_image.ndim != 2:
        raise ValueError("extract_document_ai_features expects 2D grayscale image.")

    features: Dict[str, Any] = {"valid": True}

    # Full-page features
    features.update(compute_fft_features(gray_image))
    features.update(compute_noise_statistics(gray_image))
    features.update(compute_sharpness_metrics(gray_image))
    features.update(compute_texture_regularity(gray_image))
    features.update(compute_lbp_histogram(gray_image))

    # Page-level document statistics
    # Scan line detection: real scans often have horizontal scan artifacts
    edges_h = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=5)
    edges_v = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=5)
    h_edge_strength = float(np.mean(np.abs(edges_h)))
    v_edge_strength = float(np.mean(np.abs(edges_v)))
    directional_bias = h_edge_strength / max(1e-6, v_edge_strength)

    features["directional_edge_bias"] = directional_bias

    # Margin consistency: AI-generated documents often have very uniform margins
    h, w = gray_image.shape
    margin_size = min(10, w // 8, h // 8)
    if margin_size > 1:
        top_margin = np.std(gray_image[:margin_size, :])
        bottom_margin = np.std(gray_image[-margin_size:, :])
        left_margin = np.std(gray_image[:, :margin_size])
        right_margin = np.std(gray_image[:, -margin_size:])
        margin_uniformity = 1.0 - (np.std([top_margin, bottom_margin, left_margin, right_margin]) / max(1e-6, np.mean([top_margin, bottom_margin, left_margin, right_margin])))
        features["margin_uniformity"] = float(margin_uniformity)
    else:
        features["margin_uniformity"] = 0.0

    return features


def normalize_feature_vector(features: Dict[str, float]) -> Dict[str, float]:
    """Normalize feature values to stable ranges for scoring.

    Args:
        features: Raw feature dictionary.

    Returns:
        Normalized feature dictionary with clipped [0, 1] values.
    """
    norm = dict(features)

    # Clip and normalize common features to [0, 1]
    norm["norm_fft_energy_ratio"] = float(np.clip(norm.get("fft_energy_ratio", 0.0), 0.0, 1.0))
    norm["norm_noise_cv"] = float(np.clip(norm.get("noise_cv", 0.0) / 3.0, 0.0, 1.0))
    norm["norm_laplacian_variance"] = float(np.clip(norm.get("laplacian_variance", 0.0) / 500.0, 0.0, 1.0))
    norm["norm_texture_entropy"] = float(np.clip(norm.get("texture_entropy", 0.0) / 8.0, 0.0, 1.0))
    norm["norm_lbp_entropy"] = float(np.clip(norm.get("lbp_entropy", 0.0) / 5.0, 0.0, 1.0))
    norm["norm_edge_sharpness"] = float(np.clip(norm.get("edge_sharpness", 0.0), 0.0, 1.0))
    norm["norm_gradient_consistency"] = float(np.clip(norm.get("gradient_consistency", 0.0) / 5.0, 0.0, 1.0))
    norm["norm_margin_uniformity"] = float(np.clip(norm.get("margin_uniformity", 0.0), 0.0, 1.0))
    norm["norm_directional_edge_bias"] = float(np.clip(norm.get("directional_edge_bias", 0.0) / 3.0, 0.0, 1.0))

    return norm
