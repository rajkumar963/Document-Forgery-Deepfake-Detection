"""Final label refinement and decision engine for document tampering detections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from confidence_utils import ConfidenceConfig, recalibrate_detection_group

LOGGER = logging.getLogger(__name__)

# Label priority for conflict resolution (higher = more specific/preferred)
# Use canonical hackathon labels where possible
LABEL_PRIORITY = {
    "copy_paste": 10,
    # Note: stamp, signature, seal are standardized to "added_content" by label_standardizer
    "added_content": 9,
    "overwritten_text": 7,
    "erased_content": 6,
    "ai_edited_region": 5,
    "irregular_spacing": 4,
    "suspicious_region": 0,
}


@dataclass
class LabelRefinerConfig:
    """Configuration for final label refinement."""

    # IoU threshold for grouping detections
    iou_threshold: float = 0.15
    
    # Distance threshold for grouping nearby detections (in pixels)
    distance_threshold: int = 30
    
    # Whether to merge boxes from grouped detections
    merge_boxes: bool = True
    
    # Suppress generic labels if specific label exists
    suppress_generic_suspicious: bool = True
    
    # Confidence config for score processing
    confidence_config: ConfidenceConfig = None

    def __post_init__(self):
        """Initialize defaults."""
        if self.confidence_config is None:
            self.confidence_config = ConfidenceConfig()


def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two boxes.

    Args:
        box_a: (x, y, w, h)
        box_b: (x, y, w, h)

    Returns:
        IoU value [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return float(inter_area / union_area)


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Get center point of bounding box.

    Args:
        box: (x, y, w, h)

    Returns:
        (center_x, center_y)
    """
    return (float(box[0] + box[2] / 2), float(box[1] + box[3] / 2))


def center_distance(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance between box centers.

    Args:
        box_a: (x, y, w, h)
        box_b: (x, y, w, h)

    Returns:
        Distance in pixels.
    """
    cx_a, cy_a = get_box_center(box_a)
    cx_b, cy_b = get_box_center(box_b)
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def should_group_detections(
    det_a: Dict,
    det_b: Dict,
    config: LabelRefinerConfig,
) -> bool:
    """Determine if two detections should be grouped together.

    Grouping criteria:
    - High IoU overlap
    - Close center distance + similar size

    Args:
        det_a: First detection.
        det_b: Second detection.
        config: Refiner configuration.

    Returns:
        True if should group.
    """
    box_a = tuple(det_a.get("bbox", (0, 0, 0, 0)))
    box_b = tuple(det_b.get("bbox", (0, 0, 0, 0)))

    iou = calculate_iou(box_a, box_b)
    if iou >= config.iou_threshold:
        return True

    distance = center_distance(box_a, box_b)
    if distance <= config.distance_threshold:
        # Also check if sizes are similar (avoid grouping tiny with huge)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        if area_a > 0 and area_b > 0:
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
            if area_ratio >= 0.3:  # Within 3x size difference
                return True

    return False


def group_overlapping_detections(
    detections: List[Dict],
    config: LabelRefinerConfig,
) -> List[List[Dict]]:
    """Group detections by overlap and proximity.

    Args:
        detections: List of detections.
        config: Refiner configuration.

    Returns:
        List of detection groups (each group is a list of detections).
    """
    if not detections:
        return []

    remaining = list(detections)
    groups: List[List[Dict]] = []

    while remaining:
        current_group = [remaining.pop(0)]
        changed = True

        while changed:
            changed = False
            keep = []

            for candidate in remaining:
                # Check if candidate should join current group
                belongs = False
                for member in current_group:
                    if should_group_detections(member, candidate, config):
                        belongs = True
                        break

                if belongs:
                    current_group.append(candidate)
                    changed = True
                else:
                    keep.append(candidate)

            remaining = keep

        groups.append(current_group)

    LOGGER.info("Grouped %d detections into %d groups", len(detections), len(groups))
    return groups


def resolve_group_label(
    detection_group: List[Dict],
) -> str:
    """Resolve final label for a group of detections.

    Priority order (see LABEL_PRIORITY):
    1. Higher priority labels win
    2. Equal priority: choose highest score
    3. Prefer specific over generic

    Args:
        detection_group: List of detections in group.

    Returns:
        Final resolved label.
    """
    if not detection_group:
        return "suspicious_region"

    # Sort by priority (descending) then by score (descending)
    sorted_dets = sorted(
        detection_group,
        key=lambda d: (
            int(LABEL_PRIORITY.get(str(d.get("label", "")), 0)),
            float(d.get("score", 0.0)),
        ),
        reverse=True,
    )

    final_label = str(sorted_dets[0].get("label", "suspicious_region"))

    # Check if any specific label exists (suppress generic if yes)
    has_specific = any(
        str(d.get("label", "")) != "suspicious_region"
        for d in detection_group
    )

    if has_specific and final_label == "suspicious_region":
        # Grab first non-generic label
        final_label = str(next(
            (d.get("label", "suspicious_region") for d in detection_group
             if str(d.get("label", "")) != "suspicious_region"),
            "suspicious_region",
        ))

    LOGGER.debug(
        "Resolved group label from %d detections: label=%s scores=%s",
        len(detection_group),
        final_label,
        [float(d.get("score", 0.0)) for d in detection_group],
    )

    return final_label


def merge_group_boxes(
    detection_group: List[Dict],
) -> Tuple[int, int, int, int]:
    """Merge bounding boxes in a detection group into single enclosing box.

    Args:
        detection_group: List of detections in group.

    Returns:
        Merged (x, y, w, h) bounding box.
    """
    if not detection_group:
        return (0, 0, 0, 0)

    boxes = [tuple(d.get("bbox", (0, 0, 0, 0))) for d in detection_group]

    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[0] + b[2] for b in boxes)
    y_max = max(b[1] + b[3] for b in boxes)

    w = x_max - x_min
    h = y_max - y_min

    return (int(x_min), int(y_min), int(w), int(h))


def clip_box_to_image(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int] | None:
    """Clip bounding box to image bounds.

    Args:
        bbox: (x, y, w, h)
        image_shape: (height, width)

    Returns:
        Clipped (x, y, w, h) or None if outside bounds.
    """
    x, y, w, h = bbox
    height, width = image_shape

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))

    new_w = x2 - x1
    new_h = y2 - y1

    if new_w <= 0 or new_h <= 0:
        return None

    return (x1, y1, new_w, new_h)


def attach_supporting_sources(
    detection_group: List[Dict],
) -> List[str]:
    """Collect supporting source modules for a group.

    Args:
        detection_group: List of detections.

    Returns:
        List of unique source modules.
    """
    sources = []
    for det in detection_group:
        source = str(det.get("source", ""))
        if source and source not in sources:
            sources.append(source)
    return sources


def refine_final_labels(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    config: LabelRefinerConfig | None = None,
) -> List[Dict]:
    """Apply final label refinement to all detections.

    Pipeline:
    1. Group overlapping detections
    2. Resolve label per group
    3. Merge boxes in group
    4. Recalibrate confidence scores
    5. Attach supporting sources
    6. Clip to image bounds
    7. Filter low-confidence detections

    Args:
        detections: List of raw detections from all modules.
        image_shape: (height, width)
        config: Refiner configuration.

    Returns:
        List of refined final detections ready for output.
    """
    config = config or LabelRefinerConfig()

    if not detections:
        LOGGER.info("No detections to refine.")
        return []

    LOGGER.info("Starting label refinement with %d raw detections", len(detections))

    # Group overlapping detections
    groups = group_overlapping_detections(detections, config)

    final_detections = []

    for group_idx, group in enumerate(groups):
        # Resolve label
        final_label = resolve_group_label(group)

        # Merge boxes
        merged_box = merge_group_boxes(group) if config.merge_boxes else tuple(group[0].get("bbox", (0, 0, 0, 0)))

        # Clip to image bounds
        clipped_box = clip_box_to_image(merged_box, image_shape)
        if clipped_box is None:
            LOGGER.debug("Group %d: merged box clipped away (outside bounds)", group_idx)
            continue

        # Recalibrate confidence
        conf_result = recalibrate_detection_group(group, final_label, config.confidence_config)

        # Attach supporting sources
        supporting_sources = attach_supporting_sources(group)

        final_detection = {
            "bbox": clipped_box,
            "label": final_label,
            "score": float(conf_result["score"]),
            "source": "final_refiner",
            "supporting_sources": supporting_sources,
            "group_size": len(group),
        }

        final_detections.append(final_detection)

    LOGGER.info("Refined %d detection groups into %d final detections", len(groups), len(final_detections))

    # Sort by position (top-left to bottom-right)
    final_detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

    return final_detections


def normalize_detection_record(
    detection: Dict,
) -> Dict:
    """Normalize detection record for consistency and serialization.

    Args:
        detection: Detection dictionary.

    Returns:
        Normalized detection record.
    """
    return {
        "bbox": tuple(int(v) for v in detection.get("bbox", (0, 0, 0, 0))),
        "label": str(detection.get("label", "suspicious_region")),
        "score": float(detection.get("score", 0.0)),
        "source": str(detection.get("source", "unknown")),
        "supporting_sources": [str(s) for s in detection.get("supporting_sources", [])],
    }
