"""Fusion utilities for multi-source document tampering detections."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

BoundingBox = Tuple[int, int, int, int]

LABEL_PRIORITY = {
    "copy_paste": 4,
    "stamp": 3,
    "signature": 3,
    "seal": 3,
    "added_mark": 3,
    "ai_edited_region": 3,
    "overwritten_text": 2,
    "irregular_spacing": 2,
    "erased_content": 1,
    "suspicious_region": 0,
}


def _to_xyxy(box: BoundingBox) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def _from_xyxy(box: Tuple[float, float, float, float]) -> BoundingBox:
    x1, y1, x2, y2 = box
    return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = _to_xyxy(box_a)
    bx1, by1, bx2, by2 = _to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area

    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def resolve_label_priority(label_a: str, label_b: str) -> str:
    """Resolve two labels according to priority rules."""
    pa = int(LABEL_PRIORITY.get(label_a, 0))
    pb = int(LABEL_PRIORITY.get(label_b, 0))
    if pa >= pb:
        return label_a
    return label_b


def attach_refined_labels_to_regions(
    base_detections: Sequence[Dict],
    refined_detections: Sequence[Dict],
    iou_threshold: float = 0.2,
) -> List[Dict]:
    """Attach Day 6 refined labels onto generic suspicious regions.

    If a base detection labeled suspicious_region overlaps with a refined region,
    replace its label/score/source while preserving bbox and metadata.
    """
    refined = [_normalize_detection(d) for d in refined_detections]
    output: List[Dict] = []

    for det in base_detections:
        current = dict(det)
        current_norm = _normalize_detection(current)

        if current_norm["label"] != "suspicious_region":
            output.append(current)
            continue

        best_idx = -1
        best_iou = 0.0
        for idx, candidate in enumerate(refined):
            iou = calculate_iou(current_norm["bbox"], candidate["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            match = refined[best_idx]
            current["label"] = match["label"]
            current["score"] = max(float(current.get("score", 0.0)), float(match.get("score", 0.0)))
            current["source"] = str(match.get("source", "region_classifier"))
            if "features" in refined_detections[best_idx]:
                current["features"] = refined_detections[best_idx]["features"]
            if "erasure_score" in refined_detections[best_idx]:
                current["erasure_score"] = refined_detections[best_idx]["erasure_score"]
            if "overwrite_score" in refined_detections[best_idx]:
                current["overwrite_score"] = refined_detections[best_idx]["overwrite_score"]

        output.append(current)

    return output


def _merge_boxes(box_a: BoundingBox, box_b: BoundingBox) -> BoundingBox:
    """Merge two boxes into a single enclosing box."""
    ax1, ay1, ax2, ay2 = _to_xyxy(box_a)
    bx1, by1, bx2, by2 = _to_xyxy(box_b)
    return _from_xyxy((min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)))


def _normalize_detection(det: Dict) -> Dict:
    """Normalize one detection dictionary."""
    bbox = tuple(det.get("bbox", (0, 0, 0, 0)))
    return {
        "bbox": bbox,
        "label": str(det.get("label", "suspicious_region")),
        "score": float(det.get("score", 0.0)),
        "source": str(det.get("source", "unknown")),
    }


def merge_overlapping_detections(
    detections: Sequence[Dict],
    iou_threshold: float = 0.3,
) -> List[Dict]:
    """Merge overlapping detections while preserving label/source metadata.

    If two detections overlap strongly, their boxes are merged and label priority
    is resolved using resolve_label_priority.
    """
    pending = [_normalize_detection(d) for d in detections]
    merged: List[Dict] = []

    while pending:
        current = pending.pop(0)
        changed = True

        while changed:
            changed = False
            keep: List[Dict] = []
            for candidate in pending:
                iou = calculate_iou(current["bbox"], candidate["bbox"])
                if iou >= iou_threshold:
                    current["bbox"] = _merge_boxes(current["bbox"], candidate["bbox"])
                    current["label"] = resolve_label_priority(current["label"], candidate["label"])
                    current["score"] = max(float(current["score"]), float(candidate["score"]))

                    # Prefer source of highest-priority label; fallback to higher score source.
                    if current["label"] == candidate["label"]:
                        if float(candidate["score"]) > float(current["score"]):
                            current["source"] = candidate["source"]
                    else:
                        if resolve_label_priority(current["label"], candidate["label"]) == candidate["label"]:
                            current["source"] = candidate["source"]

                    changed = True
                else:
                    keep.append(candidate)
            pending = keep

        merged.append(current)

    merged.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return merged


def deduplicate_labeled_boxes(
    detections: Sequence[Dict],
    iou_threshold: float = 0.85,
) -> List[Dict]:
    """Remove near-duplicate labeled boxes with score/priority preference."""
    ordered = sorted(
        [_normalize_detection(d) for d in detections],
        key=lambda d: (int(LABEL_PRIORITY.get(d["label"], 0)), float(d["score"])),
        reverse=True,
    )

    kept: List[Dict] = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if calculate_iou(candidate["bbox"], existing["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return kept


def merge_refined_detections(
    detections: Sequence[Dict],
    merge_iou_threshold: float = 0.3,
    dedupe_iou_threshold: float = 0.85,
) -> List[Dict]:
    """Merge and deduplicate refined detections for final reporting."""
    merged = merge_overlapping_detections(detections, iou_threshold=merge_iou_threshold)
    return deduplicate_labeled_boxes(merged, iou_threshold=dedupe_iou_threshold)


def fuse_document_level_result(
    document_level_result: Dict,
) -> Dict:
    """Format document-level AI-generation result for final report.

    Args:
        document_level_result: Output from detect_fully_ai_generated_document.

    Returns:
        Formatted dict with label, score, and features for final report.
    """
    return {
        "label": str(document_level_result.get("label", "authentic_or_unknown")),
        "score": float(document_level_result.get("score", 0.0)),
        "features": document_level_result.get("features", {}),
    }


def fuse_region_level_results(
    ai_edit_detections: Sequence[Dict],
    other_detections: Sequence[Dict],
    merge_iou_threshold: float = 0.3,
) -> List[Dict]:
    """Fuse region-level AI-edit detections with other anomalies.

    AI-edited regions are label-prioritized between added_mark and erased_content.

    Args:
        ai_edit_detections: List from detect_ai_edited_regions_from_boxes.
        other_detections: Other anomaly detections (anomaly, yolo, ocr, etc.).
        merge_iou_threshold: IoU threshold for merging overlaps.

    Returns:
        Merged detections with ai_edited_region prioritized.
    """
    # Convert AI-edit detections to standard dict format
    # Only include regions with sufficient confidence
    converted = []
    ai_edit_threshold = 0.50  # Only convert "not_ai_edited" if score >= threshold
    
    for det in ai_edit_detections:
        if "skip_reason" in det:
            continue  # Skip too-small regions
        
        label = str(det.get("label", "not_ai_edited"))
        score = float(det.get("score", 0.0))
        
        # For "not_ai_edited", only include if it's actually confident
        # (i.e., high confidence that region is NOT AI-edited = probably authentic)
        if label == "not_ai_edited" and score < ai_edit_threshold:
            continue  # Skip low-confidence non-AI-edits
        
        # Convert "not_ai_edited" to "suspicious_region" only if moderate-high confidence
        if label == "not_ai_edited":
            label = "suspicious_region"
        
        converted.append({
            "bbox": det.get("bbox", (0, 0, 0, 0)),
            "label": label,
            "score": score,
            "source": "ai_edit_region_detector",
            "context_contrast": float(det.get("context_contrast", 0.0)),
        })

    # Combine all detections
    all_dets = list(other_detections) + converted

    # Merge with priority resolution
    merged = merge_refined_detections(
        all_dets,
        merge_iou_threshold=merge_iou_threshold,
    )

    return merged


def build_detection_groups(
    detections: Sequence[Dict],
    iou_threshold: float = 0.15,
    distance_threshold: int = 30,
) -> List[List[Dict]]:
    """Build groups of overlapping or nearby detections.

    Detections are grouped if:
    - They have high IoU overlap
    - Their centers are close and sizes are similar

    Args:
        detections: List of detections.
        iou_threshold: IoU threshold for grouping.
        distance_threshold: Max distance between centers for grouping (pixels).

    Returns:
        List of detection groups.
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
            keep: List[Dict] = []

            for candidate in remaining:
                belongs = False

                for member in current_group:
                    member_box = tuple(member.get("bbox", (0, 0, 0, 0)))
                    cand_box = tuple(candidate.get("bbox", (0, 0, 0, 0)))

                    iou = calculate_iou(member_box, cand_box)
                    if iou >= iou_threshold:
                        belongs = True
                        break

                    if distance_threshold > 0:
                        # Check center distance
                        cx_m = member_box[0] + member_box[2] / 2
                        cy_m = member_box[1] + member_box[3] / 2
                        cx_c = cand_box[0] + cand_box[2] / 2
                        cy_c = cand_box[1] + cand_box[3] / 2
                        dist = ((cx_m - cx_c) ** 2 + (cy_m - cy_c) ** 2) ** 0.5

                        if dist <= distance_threshold:
                            area_m = member_box[2] * member_box[3]
                            area_c = cand_box[2] * cand_box[3]
                            if area_m > 0 and area_c > 0:
                                area_ratio = min(area_m, area_c) / max(area_m, area_c)
                                if area_ratio >= 0.3:
                                    belongs = True
                                    break

                if belongs:
                    current_group.append(candidate)
                    changed = True
                else:
                    keep.append(candidate)

            remaining = keep

        groups.append(current_group)

    return groups


def merge_boxes_in_group(
    detection_group: Sequence[Dict],
) -> BoundingBox:
    """Merge all boxes in a group into single enclosing box.

    Args:
        detection_group: List of detections.

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


def collect_group_metadata(
    detection_group: Sequence[Dict],
) -> Dict[str, any]:
    """Collect metadata for a detection group.

    Args:
        detection_group: List of detections in group.

    Returns:
        Metadata dictionary with sources, scores, labels, etc.
    """
    if not detection_group:
        return {}

    sources = []
    labels = []
    scores = []

    for det in detection_group:
        source = str(det.get("source", ""))
        if source and source not in sources:
            sources.append(source)

        label = str(det.get("label", ""))
        if label not in labels:
            labels.append(label)

        score = float(det.get("score", 0.0))
        scores.append(score)

    return {
        "group_size": len(detection_group),
        "sources": sources,
        "labels": labels,
        "scores": scores,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "max_score": float(np.max(scores)) if scores else 0.0,
        "min_score": float(np.min(scores)) if scores else 0.0,
    }

