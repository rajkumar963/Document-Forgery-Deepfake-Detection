"""Bounding box utilities for anomaly-based suspicious region localization."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

BoundingBox = Tuple[int, int, int, int]


def clip_box_to_bounds(box: BoundingBox, image_shape: tuple[int, int]) -> BoundingBox | None:
    """Clip a bounding box to image bounds.

    Args:
        box: Input bounding box as (x, y, w, h).
        image_shape: Image shape as (height, width).

    Returns:
        Clipped bounding box or None if invalid after clipping.
    """
    x, y, w, h = box
    height, width = image_shape

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None

    return (x1, y1, new_w, new_h)


def contours_to_bboxes(
    contours: Sequence[np.ndarray],
    image_shape: tuple[int, int],
    min_contour_area: int = 120,
    min_box_area: int = 250,
    min_width: int = 10,
    min_height: int = 10,
) -> List[BoundingBox]:
    """Convert contours to filtered bounding boxes."""
    boxes: List[BoundingBox] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h < min_height:
            continue
        if w * h < min_box_area:
            continue

        clipped = clip_box_to_bounds((x, y, w, h), image_shape=image_shape)
        if clipped is not None:
            boxes.append(clipped)

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _to_xyxy(box: BoundingBox) -> Tuple[int, int, int, int]:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = box
    return x, y, x + w, y + h


def _from_xyxy(box: Tuple[int, int, int, int]) -> BoundingBox:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = _to_xyxy(box_a)
    bx1, by1, bx2, by2 = _to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def _boxes_are_near(box_a: BoundingBox, box_b: BoundingBox, max_gap: int) -> bool:
    """Check whether two boxes are spatially close after margin expansion."""
    ax1, ay1, ax2, ay2 = _to_xyxy(box_a)
    bx1, by1, bx2, by2 = _to_xyxy(box_b)

    ax1 -= max_gap
    ay1 -= max_gap
    ax2 += max_gap
    ay2 += max_gap

    overlap_x = ax1 <= bx2 and bx1 <= ax2
    overlap_y = ay1 <= by2 and by1 <= ay2
    return overlap_x and overlap_y


def _merge_pair(box_a: BoundingBox, box_b: BoundingBox) -> BoundingBox:
    """Merge two boxes into their enclosing union box."""
    ax1, ay1, ax2, ay2 = _to_xyxy(box_a)
    bx1, by1, bx2, by2 = _to_xyxy(box_b)

    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)

    return _from_xyxy((x1, y1, x2, y2))


def merge_overlapping_boxes(
    boxes: Iterable[BoundingBox],
    iou_threshold: float = 0.2,
    max_gap: int = 12,
) -> List[BoundingBox]:
    """Merge overlapping or nearby boxes using iterative union."""
    pending = list(boxes)
    if not pending:
        return []

    merged_boxes: List[BoundingBox] = []

    while pending:
        current = pending.pop(0)
        changed = True

        while changed:
            changed = False
            remaining: List[BoundingBox] = []

            for candidate in pending:
                iou = compute_iou(current, candidate)
                if iou >= iou_threshold or _boxes_are_near(current, candidate, max_gap=max_gap):
                    current = _merge_pair(current, candidate)
                    changed = True
                else:
                    remaining.append(candidate)

            pending = remaining

        merged_boxes.append(current)

    merged_boxes.sort(key=lambda b: (b[1], b[0]))
    return merged_boxes


def non_max_suppression(
    boxes: Sequence[BoundingBox],
    iou_threshold: float = 0.45,
    scores: Sequence[float] | None = None,
) -> List[BoundingBox]:
    """Apply NMS to remove redundant overlapping boxes."""
    if not boxes:
        return []

    boxes_array = np.array(boxes, dtype=np.float32)
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 0] + boxes_array[:, 2]
    y2 = boxes_array[:, 1] + boxes_array[:, 3]
    areas = boxes_array[:, 2] * boxes_array[:, 3]

    if scores is None:
        order = np.argsort(areas)[::-1]
    else:
        order = np.argsort(np.array(scores, dtype=np.float32))[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection + 1e-8
        iou = intersection / union

        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    kept_boxes = [boxes[idx] for idx in keep]
    kept_boxes.sort(key=lambda b: (b[1], b[0]))
    return kept_boxes


def boxes_from_mask(
    binary_mask: np.ndarray,
    image_shape: tuple[int, int],
    min_contour_area: int = 120,
    min_box_area: int = 250,
    min_width: int = 10,
    min_height: int = 10,
    merge_iou_threshold: float = 0.2,
    merge_max_gap: int = 12,
    nms_iou_threshold: float = 0.45,
) -> List[BoundingBox]:
    """Convert a binary anomaly mask to cleaned final bounding boxes."""
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("binary_mask is empty or None.")
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array.")

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = contours_to_bboxes(
        contours=contours,
        image_shape=image_shape,
        min_contour_area=min_contour_area,
        min_box_area=min_box_area,
        min_width=min_width,
        min_height=min_height,
    )

    merged = merge_overlapping_boxes(raw_boxes, iou_threshold=merge_iou_threshold, max_gap=merge_max_gap)
    final_boxes = non_max_suppression(merged, iou_threshold=nms_iou_threshold)

    clipped: List[BoundingBox] = []
    for box in final_boxes:
        clipped_box = clip_box_to_bounds(box, image_shape=image_shape)
        if clipped_box is not None:
            clipped.append(clipped_box)

    clipped.sort(key=lambda b: (b[1], b[0]))
    return clipped
