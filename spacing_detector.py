"""High-level OCR spacing detector for text-level tampering localization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ocr_engine import OCREngineConfig, extract_ocr_words
from text_analysis import detect_font_inconsistency, detect_irregular_spacing, group_words_by_line

LOGGER = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]


@dataclass
class SpacingDetectorConfig:
    """Configuration for OCR spacing anomaly detector."""

    ocr_min_confidence: float = 45.0
    spacing_z_threshold: float = 2.2
    spacing_min_gap: float = 7.0
    spacing_max_relative_gap: float = 2.4
    height_z_threshold: float = 1.8
    min_words_per_line: int = 4
    box_expand: int = 6
    merge_iou_threshold: float = 0.20
    merge_gap: int = 10


def _expand_box(box: BoundingBox, expand: int, image_shape: tuple[int, int]) -> BoundingBox:
    """Expand bounding box with clipping to image bounds."""
    x, y, w, h = box
    height, width = image_shape

    x1 = max(0, x - expand)
    y1 = max(0, y - expand)
    x2 = min(width, x + w + expand)
    y2 = min(height, y + h + expand)
    return (x1, y1, x2 - x1, y2 - y1)


def _to_xyxy(box: BoundingBox) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _from_xyxy(box: tuple[int, int, int, int]) -> BoundingBox:
    x1, y1, x2, y2 = box
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _nearby(a: BoundingBox, b: BoundingBox, gap: int) -> bool:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ax1 -= gap
    ay1 -= gap
    ax2 += gap
    ay2 += gap
    return (ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2)


def _merge_box(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    return _from_xyxy((min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)))


def _merge_suspicious_regions(
    regions: List[Dict[str, Any]],
    iou_threshold: float,
    nearby_gap: int,
) -> List[Dict[str, Any]]:
    """Merge overlapping/nearby OCR suspicious regions."""
    pending = [dict(r) for r in regions]
    merged: List[Dict[str, Any]] = []

    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            keep = []
            for candidate in pending:
                iou = _compute_iou(current["bbox"], candidate["bbox"])
                if iou >= iou_threshold or _nearby(current["bbox"], candidate["bbox"], nearby_gap):
                    current["bbox"] = _merge_box(current["bbox"], candidate["bbox"])
                    current["score"] = max(float(current["score"]), float(candidate["score"]))
                    changed = True
                else:
                    keep.append(candidate)
            pending = keep
        merged.append(current)

    merged.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return merged


def detect_irregular_text_regions(
    preprocessed_gray,
    config: SpacingDetectorConfig | None = None,
) -> List[Dict[str, Any]]:
    """Detect irregular spacing and font anomalies using OCR analysis.

    Returns:
        List of regions in format:
            {"bbox": (x, y, w, h), "label": "irregular_spacing", "score": float}
    """
    cfg = config or SpacingDetectorConfig()

    ocr_words = extract_ocr_words(
        preprocessed_gray,
        config=OCREngineConfig(min_confidence=cfg.ocr_min_confidence),
    )

    if not ocr_words:
        LOGGER.warning("Spacing detector: no OCR words available.")
        return []

    grouped = group_words_by_line(ocr_words)

    spacing_anomalies = detect_irregular_spacing(
        grouped,
        z_threshold=cfg.spacing_z_threshold,
        min_absolute_gap=cfg.spacing_min_gap,
        max_relative_gap=cfg.spacing_max_relative_gap,
        min_words_per_line=cfg.min_words_per_line,
    )

    font_anomalies = detect_font_inconsistency(
        grouped,
        height_z_threshold=cfg.height_z_threshold,
        min_words_per_line=cfg.min_words_per_line,
    )

    height, width = preprocessed_gray.shape[:2]
    raw_regions: List[Dict[str, Any]] = []

    for item in spacing_anomalies + font_anomalies:
        expanded = _expand_box(item["bbox"], cfg.box_expand, (height, width))
        raw_regions.append(
            {
                "bbox": expanded,
                "label": "irregular_spacing",
                "score": float(item.get("score", 0.6)),
            }
        )

    merged = _merge_suspicious_regions(
        raw_regions,
        iou_threshold=cfg.merge_iou_threshold,
        nearby_gap=cfg.merge_gap,
    )

    LOGGER.info(
        "Spacing detector: words=%d, spacing_anomalies=%d, font_anomalies=%d, final_regions=%d",
        len(ocr_words),
        len(spacing_anomalies),
        len(font_anomalies),
        len(merged),
    )

    return merged
