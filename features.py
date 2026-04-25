"""Feature extraction for patch-based document anomaly localization.

This module computes handcrafted anomaly-sensitive feature maps from a
preprocessed grayscale document image and combines them into a patch-based
anomaly score map aligned to the image resolution.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on runtime environment
    TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "variance": 0.28,
    "edge_density": 0.24,
    "laplacian": 0.22,
    "texture": 0.18,
    "stroke": 0.08,
}


def _validate_grayscale_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty grayscale image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (single channel).")


def _normalize_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a float map to [0, 1]."""
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v + eps)


def _torch_weighted_combine(feature_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine feature maps with PyTorch when available."""
    tensor_stack = torch.from_numpy(feature_stack).float()  # [N, H, W]
    tensor_weights = torch.from_numpy(weights).float().view(-1, 1, 1)

    weighted = (tensor_stack * tensor_weights).sum(dim=0)
    denom = torch.clamp(tensor_weights.sum(), min=1e-6)
    combined = weighted / denom
    return combined.cpu().numpy().astype(np.float32)


def _get_patch_positions(length: int, patch_size: int, stride: int) -> List[int]:
    """Generate robust patch start positions with end coverage."""
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def _patch_pool(score_map: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """Pool a dense score map into patch-level scores."""
    h, w = score_map.shape
    ys = _get_patch_positions(h, patch_size, stride)
    xs = _get_patch_positions(w, patch_size, stride)

    pooled = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = score_map[y : y + patch_size, x : x + patch_size]
            pooled[i, j] = float(np.mean(patch))

    return pooled


def _upsample_to_image_map(patch_map: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Upsample patch map to image resolution."""
    h, w = image_shape
    upsampled = cv2.resize(patch_map, (w, h), interpolation=cv2.INTER_CUBIC)
    return _normalize_map(upsampled)


def compute_feature_components(
    preprocessed_gray: np.ndarray,
    patch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Compute anomaly-sensitive feature component maps.

    Args:
        preprocessed_gray: Preprocessed grayscale document image.
        patch_size: Patch size used to define local neighborhood statistics.

    Returns:
        Mapping of feature name to normalized feature map in [0, 1].
    """
    _validate_grayscale_image(preprocessed_gray)

    if patch_size < 8:
        raise ValueError("patch_size must be >= 8 for stable local statistics.")

    gray = preprocessed_gray.astype(np.float32)
    kernel_size = patch_size if patch_size % 2 == 1 else patch_size + 1

    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(
        gray * gray,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )
    local_variance = np.maximum(mean_sq - mean * mean, 0.0)

    edges = cv2.Canny(preprocessed_gray, threshold1=70, threshold2=180).astype(np.float32) / 255.0
    edge_density = cv2.boxFilter(
        edges,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )

    laplacian = np.abs(cv2.Laplacian(preprocessed_gray, cv2.CV_32F, ksize=3))
    laplacian_response = cv2.boxFilter(
        laplacian,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )

    median = cv2.medianBlur(preprocessed_gray, 5).astype(np.float32)
    texture_irregularity = cv2.boxFilter(
        np.abs(gray - median),
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )

    # Black-hat emphasizes thin dark stroke inconsistencies in document regions.
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stroke_proxy = cv2.morphologyEx(preprocessed_gray, cv2.MORPH_BLACKHAT, stroke_kernel).astype(np.float32)
    stroke_proxy = cv2.boxFilter(
        stroke_proxy,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )

    features: Dict[str, np.ndarray] = {
        "variance": _normalize_map(local_variance),
        "edge_density": _normalize_map(edge_density),
        "laplacian": _normalize_map(laplacian_response),
        "texture": _normalize_map(texture_irregularity),
        "stroke": _normalize_map(stroke_proxy),
    }

    return features


def extract_patch_anomaly_map(
    preprocessed_gray: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
    feature_weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """Extract a patch-based anomaly score map aligned with the image.

    Args:
        preprocessed_gray: Preprocessed grayscale document image.
        patch_size: Sliding window size.
        stride: Sliding window stride.
        feature_weights: Optional feature weights. Missing keys fallback to defaults.

    Returns:
        2D anomaly score map in [0, 1], same size as input image.

    Raises:
        ValueError: If input image or configuration is invalid.
    """
    _validate_grayscale_image(preprocessed_gray)

    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0.")

    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights is not None:
        weights.update(feature_weights)

    components = compute_feature_components(preprocessed_gray=preprocessed_gray, patch_size=patch_size)

    feature_names = [name for name in DEFAULT_FEATURE_WEIGHTS if name in components]
    stack = np.stack([components[name] for name in feature_names], axis=0).astype(np.float32)
    weight_vector = np.array([float(weights[name]) for name in feature_names], dtype=np.float32)

    if TORCH_AVAILABLE:
        combined_dense = _torch_weighted_combine(stack, weight_vector)
    else:
        LOGGER.warning("PyTorch not available; using NumPy weighted feature fusion.")
        combined_dense = np.average(stack, axis=0, weights=weight_vector)

    combined_dense = _normalize_map(combined_dense)
    patch_scores = _patch_pool(combined_dense, patch_size=patch_size, stride=stride)
    patch_scores = _normalize_map(patch_scores)

    anomaly_map = _upsample_to_image_map(patch_scores, preprocessed_gray.shape)
    return anomaly_map
