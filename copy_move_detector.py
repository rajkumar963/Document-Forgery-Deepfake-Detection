"""Copy-move and copy-paste forgery detection for single document images.

This module detects suspicious duplicated regions within the same page by
matching local ORB descriptors and clustering geometrically consistent matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from bbox_utils import BoundingBox, clip_box_to_bounds, merge_overlapping_boxes, non_max_suppression
from keypoint_utils import (
    MatchPair,
    compute_keypoint_spatial_distance,
    descriptors_are_valid,
    extract_orb_keypoints_and_descriptors,
    matches_to_point_arrays,
    pair_displacement_vectors,
    remove_duplicate_and_mirrored_match_pairs,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CopyMoveConfig:
    """Configuration for copy-move detection."""

    orb_nfeatures: int = 2500
    orb_scale_factor: float = 1.2
    orb_nlevels: int = 8
    matcher_cross_check: bool = False
    ratio_test_threshold: float = 0.78
    max_descriptor_distance: int = 64
    min_spatial_distance: float = 20.0
    max_spatial_distance: float = 700.0
    min_match_count_for_clustering: int = 8
    displacement_consistency_thresh: float = 12.0
    match_draw_thickness: int = 1
    density_radius: int = 14
    density_min_hits: int = 2
    cluster_expand_pixels: int = 14
    min_cluster_area: int = 220
    merge_iou_threshold: float = 0.2
    merge_max_gap: int = 10
    nms_iou_threshold: float = 0.45


def detect_keypoints_and_descriptors(
    preprocessed_gray: np.ndarray,
    config: CopyMoveConfig,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect ORB keypoints and descriptors."""
    return extract_orb_keypoints_and_descriptors(
        gray_image=preprocessed_gray,
        nfeatures=config.orb_nfeatures,
        scale_factor=config.orb_scale_factor,
        nlevels=config.orb_nlevels,
    )


def match_descriptors_within_image(
    descriptors: np.ndarray | None,
    config: CopyMoveConfig,
) -> list[cv2.DMatch]:
    """Match descriptors against themselves and return candidate matches."""
    if not descriptors_are_valid(descriptors):
        return []

    if config.matcher_cross_check:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors, descriptors)
        return sorted(matches, key=lambda m: m.distance)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(descriptors, descriptors, k=3)

    filtered: list[cv2.DMatch] = []
    for group in knn:
        # k can be < 3 for edge cases.
        valid = [m for m in group if m.queryIdx != m.trainIdx]
        if len(valid) < 2:
            continue

        m1, m2 = valid[0], valid[1]
        if m1.distance <= config.ratio_test_threshold * max(m2.distance, 1e-6):
            filtered.append(m1)

    return sorted(filtered, key=lambda m: m.distance)


def filter_matches(
    keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    config: CopyMoveConfig,
) -> list[MatchPair]:
    """Filter matches by descriptor and spatial constraints."""
    pairs: list[MatchPair] = []

    for match in matches:
        query_idx = int(match.queryIdx)
        train_idx = int(match.trainIdx)
        descriptor_distance = float(match.distance)

        if query_idx == train_idx:
            continue
        if descriptor_distance > float(config.max_descriptor_distance):
            continue

        spatial_distance = compute_keypoint_spatial_distance(keypoints, query_idx, train_idx)
        if spatial_distance < float(config.min_spatial_distance):
            continue
        if spatial_distance > float(config.max_spatial_distance):
            continue

        pairs.append((query_idx, train_idx, descriptor_distance))

    pairs = remove_duplicate_and_mirrored_match_pairs(pairs)

    if not pairs:
        return []

    vectors = pair_displacement_vectors(keypoints, pairs)
    if vectors.shape[0] < config.min_match_count_for_clustering:
        return pairs

    robust_center = np.median(vectors, axis=0)
    residual = np.linalg.norm(vectors - robust_center, axis=1)
    keep_mask = residual <= float(config.displacement_consistency_thresh)

    consistent = [pair for pair, keep in zip(pairs, keep_mask) if keep]
    if len(consistent) >= config.min_match_count_for_clustering:
        return consistent
    return pairs


def _compute_density_score(points: np.ndarray, radius: int) -> np.ndarray:
    """Compute local density score by counting neighbors in radius."""
    if points.size == 0:
        return np.empty((0,), dtype=np.int32)

    density = np.zeros((points.shape[0],), dtype=np.int32)
    for i in range(points.shape[0]):
        diff = points - points[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        density[i] = int(np.sum(dist <= float(radius))) - 1
    return density


def _points_to_density_mask(
    points: np.ndarray,
    image_shape: tuple[int, int],
    radius: int,
    min_hits: int,
) -> np.ndarray:
    """Convert dense matched points to binary mask for contour clustering."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if points.size == 0:
        return mask

    density = _compute_density_score(points, radius=radius)
    dense_points = points[density >= int(min_hits)]
    if dense_points.size == 0:
        dense_points = points

    for x, y in dense_points:
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        cv2.circle(mask, (xi, yi), radius=max(2, radius // 2), color=255, thickness=-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, radius), max(3, radius)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def cluster_matched_regions(
    keypoints: Sequence[cv2.KeyPoint],
    filtered_pairs: Sequence[MatchPair],
    image_shape: tuple[int, int],
    config: CopyMoveConfig,
) -> list[BoundingBox]:
    """Cluster matched keypoint pairs into candidate copy-move boxes."""
    if not filtered_pairs:
        return []

    src_points, dst_points = matches_to_point_arrays(keypoints, filtered_pairs)
    all_points = np.vstack([src_points, dst_points]) if src_points.size > 0 else np.empty((0, 2), dtype=np.float32)

    density_mask = _points_to_density_mask(
        points=all_points,
        image_shape=image_shape,
        radius=config.density_radius,
        min_hits=config.density_min_hits,
    )

    expand = max(1, int(config.cluster_expand_pixels))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * expand + 1, 2 * expand + 1))
    expanded = cv2.dilate(density_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BoundingBox] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(config.min_cluster_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        clipped = clip_box_to_bounds((x, y, w, h), image_shape=image_shape)
        if clipped is not None:
            boxes.append(clipped)

    boxes = merge_overlapping_boxes(boxes, iou_threshold=config.merge_iou_threshold, max_gap=config.merge_max_gap)
    boxes = non_max_suppression(boxes, iou_threshold=config.nms_iou_threshold)
    return boxes


def detect_copy_move_regions(
    preprocessed_gray: np.ndarray,
    config: CopyMoveConfig | None = None,
) -> Dict[str, Any]:
    """Detect copy-move tampering regions within a single document image.

    Args:
        preprocessed_gray: Preprocessed grayscale document image.
        config: Optional copy-move detector configuration.

    Returns:
        Dictionary with keys:
            - boxes: list[(x, y, w, h)]
            - matches: list[(x1, y1, x2, y2, distance)]
            - num_keypoints: int
            - num_filtered_matches: int
    """
    cfg = config or CopyMoveConfig()

    if preprocessed_gray is None or preprocessed_gray.size == 0:
        raise ValueError("preprocessed_gray is empty or None.")
    if preprocessed_gray.ndim != 2:
        raise ValueError("preprocessed_gray must be a grayscale image.")

    keypoints, descriptors = detect_keypoints_and_descriptors(preprocessed_gray, config=cfg)
    num_keypoints = len(keypoints)
    LOGGER.info("Copy-move: detected %d keypoints.", num_keypoints)

    if num_keypoints < 10 or not descriptors_are_valid(descriptors):
        LOGGER.warning("Copy-move: insufficient keypoints/descriptors for matching.")
        return {
            "boxes": [],
            "matches": [],
            "num_keypoints": num_keypoints,
            "num_filtered_matches": 0,
        }

    raw_matches = match_descriptors_within_image(descriptors, config=cfg)
    LOGGER.info("Copy-move: raw candidate matches=%d", len(raw_matches))

    filtered_pairs = filter_matches(keypoints, raw_matches, config=cfg)
    LOGGER.info("Copy-move: filtered matches=%d", len(filtered_pairs))

    boxes = cluster_matched_regions(
        keypoints=keypoints,
        filtered_pairs=filtered_pairs,
        image_shape=preprocessed_gray.shape,
        config=cfg,
    )
    LOGGER.info("Copy-move: final detected regions=%d", len(boxes))

    match_metadata = []
    for query_idx, train_idx, distance in filtered_pairs:
        x1, y1 = keypoints[query_idx].pt
        x2, y2 = keypoints[train_idx].pt
        match_metadata.append((float(x1), float(y1), float(x2), float(y2), float(distance)))

    return {
        "boxes": boxes,
        "matches": match_metadata,
        "num_keypoints": num_keypoints,
        "num_filtered_matches": len(filtered_pairs),
    }
