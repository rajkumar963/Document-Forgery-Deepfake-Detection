"""Reusable keypoint and descriptor utilities for copy-move detection."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

Point2D = Tuple[float, float]
MatchPair = Tuple[int, int, float]


def extract_orb_keypoints_and_descriptors(
    gray_image: np.ndarray,
    nfeatures: int = 2500,
    scale_factor: float = 1.2,
    nlevels: int = 8,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Extract ORB keypoints and descriptors from a grayscale image.

    Args:
        gray_image: Input grayscale image.
        nfeatures: Maximum number of ORB features.
        scale_factor: Pyramid decimation ratio used by ORB.
        nlevels: Number of pyramid levels used by ORB.

    Returns:
        Tuple of keypoints and descriptors.

    Raises:
        ValueError: If input image is invalid.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Input grayscale image is empty or None.")
    if gray_image.ndim != 2:
        raise ValueError("Input image must be grayscale for ORB extraction.")

    orb = cv2.ORB_create(
        nfeatures=int(max(100, nfeatures)),
        scaleFactor=float(max(1.01, scale_factor)),
        nlevels=int(max(1, nlevels)),
    )
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    LOGGER.debug("ORB extraction produced %d keypoints.", len(keypoints))
    return keypoints, descriptors


def descriptors_are_valid(descriptors: np.ndarray | None) -> bool:
    """Return True if descriptors are available and non-empty."""
    return descriptors is not None and descriptors.size > 0 and len(descriptors.shape) == 2


def compute_keypoint_spatial_distance(
    keypoints: Sequence[cv2.KeyPoint],
    query_idx: int,
    train_idx: int,
) -> float:
    """Compute Euclidean pixel distance between two keypoints."""
    p1 = keypoints[query_idx].pt
    p2 = keypoints[train_idx].pt
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def remove_duplicate_and_mirrored_match_pairs(pairs: Iterable[MatchPair]) -> list[MatchPair]:
    """Remove duplicate and mirrored keypoint match pairs.

    A match (a, b) and mirrored (b, a) are treated as the same pair.
    """
    seen: set[tuple[int, int]] = set()
    unique: list[MatchPair] = []

    for query_idx, train_idx, distance in pairs:
        if query_idx == train_idx:
            continue
        key = (query_idx, train_idx) if query_idx < train_idx else (train_idx, query_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append((query_idx, train_idx, float(distance)))

    return unique


def matches_to_point_arrays(
    keypoints: Sequence[cv2.KeyPoint],
    pairs: Sequence[MatchPair],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert keypoint index pairs into two point arrays.

    Returns:
        src_points: Nx2 array of source points.
        dst_points: Nx2 array of destination points.
    """
    if not pairs:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    src = np.array([keypoints[q].pt for q, _, _ in pairs], dtype=np.float32)
    dst = np.array([keypoints[t].pt for _, t, _ in pairs], dtype=np.float32)
    return src, dst


def pair_displacement_vectors(
    keypoints: Sequence[cv2.KeyPoint],
    pairs: Sequence[MatchPair],
) -> np.ndarray:
    """Compute displacement vectors for each keypoint match pair."""
    if not pairs:
        return np.empty((0, 2), dtype=np.float32)

    vectors = []
    for query_idx, train_idx, _ in pairs:
        p1 = keypoints[query_idx].pt
        p2 = keypoints[train_idx].pt
        vectors.append((p2[0] - p1[0], p2[1] - p1[1]))

    return np.array(vectors, dtype=np.float32)
