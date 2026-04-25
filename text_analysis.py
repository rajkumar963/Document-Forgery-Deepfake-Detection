"""Text-level spacing and font consistency analysis for forgery detection."""

from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]


def _merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """Merge multiple boxes into one enclosing box."""
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return (x1, y1, x2 - x1, y2 - y1)


def group_words_by_line(words: List[Dict[str, Any]]) -> Dict[Tuple[int, int, int], List[Dict[str, Any]]]:
    """Group OCR words by (block_num, par_num, line_num).

    Args:
        words: OCR word entries.

    Returns:
        Dictionary mapping line key to words sorted left-to-right.
    """
    grouped: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

    for word in words:
        key = (
            int(word.get("block_num", 0)),
            int(word.get("par_num", 0)),
            int(word.get("line_num", 0)),
        )
        grouped.setdefault(key, []).append(word)

    for key in grouped:
        grouped[key].sort(key=lambda w: int(w["bbox"][0]))

    return grouped


def compute_spacing_metrics(line_words: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute spacing and height statistics for one text line."""
    if len(line_words) < 2:
        return {
            "avg_spacing": 0.0,
            "std_spacing": 0.0,
            "avg_height": float(line_words[0]["bbox"][3]) if line_words else 0.0,
            "std_height": 0.0,
            "num_words": float(len(line_words)),
        }

    spacings: List[float] = []
    heights: List[float] = []

    for idx, word in enumerate(line_words):
        x, _, w, h = word["bbox"]
        heights.append(float(h))

        if idx < len(line_words) - 1:
            next_x = int(line_words[idx + 1]["bbox"][0])
            spacing = float(next_x - (x + w))
            if spacing >= 0:
                spacings.append(spacing)

    if not spacings:
        spacings = [0.0]

    return {
        "avg_spacing": float(mean(spacings)),
        "std_spacing": float(pstdev(spacings) if len(spacings) > 1 else 0.0),
        "avg_height": float(mean(heights)) if heights else 0.0,
        "std_height": float(pstdev(heights) if len(heights) > 1 else 0.0),
        "num_words": float(len(line_words)),
    }


def detect_irregular_spacing(
    grouped_lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]],
    z_threshold: float = 2.2,
    min_absolute_gap: float = 7.0,
    max_relative_gap: float = 2.4,
    min_words_per_line: int = 4,
) -> List[Dict[str, Any]]:
    """Detect suspicious spacing anomalies between consecutive words.

    Uses per-line spacing distribution and flags outlier gaps.
    """
    anomalies: List[Dict[str, Any]] = []

    for line_key, line_words in grouped_lines.items():
        if len(line_words) < min_words_per_line:
            continue

        metrics = compute_spacing_metrics(line_words)
        avg_gap = metrics["avg_spacing"]
        std_gap = max(metrics["std_spacing"], 1.0)

        # Skip likely justified text lines with uniformly large spacing variance.
        if std_gap > avg_gap * 1.2 and avg_gap > 10.0:
            continue

        for idx in range(len(line_words) - 1):
            left_word = line_words[idx]
            right_word = line_words[idx + 1]

            x1, y1, w1, h1 = left_word["bbox"]
            x2, y2, w2, h2 = right_word["bbox"]

            gap = float(x2 - (x1 + w1))
            if gap < 0:
                continue

            z_score = (gap - avg_gap) / std_gap
            relative = gap / max(avg_gap, 1.0)

            if gap >= min_absolute_gap and z_score >= z_threshold and relative <= max_relative_gap * 3:
                bbox = _merge_boxes([(x1, y1, w1, h1), (x2, y2, w2, h2)])
                score = float(min(0.99, max(0.5, 0.5 + (z_score / 6.0))))
                anomalies.append(
                    {
                        "bbox": bbox,
                        "label": "irregular_spacing",
                        "score": score,
                        "line_key": line_key,
                        "gap": gap,
                        "z_score": z_score,
                        "left_text": left_word.get("text", ""),
                        "right_text": right_word.get("text", ""),
                    }
                )

    LOGGER.info("Text analysis found %d spacing anomalies.", len(anomalies))
    return anomalies


def detect_font_inconsistency(
    grouped_lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]],
    height_z_threshold: float = 1.8,
    min_words_per_line: int = 4,
) -> List[Dict[str, Any]]:
    """Detect suspicious font-size inconsistency inside lines."""
    anomalies: List[Dict[str, Any]] = []

    for line_key, line_words in grouped_lines.items():
        if len(line_words) < min_words_per_line:
            continue

        heights = np.array([float(word["bbox"][3]) for word in line_words], dtype=np.float32)
        mean_h = float(np.mean(heights))
        std_h = float(np.std(heights))
        if std_h < 1.0:
            continue

        for word in line_words:
            x, y, w, h = word["bbox"]
            z = (float(h) - mean_h) / std_h
            if abs(z) >= height_z_threshold:
                score = float(min(0.95, 0.45 + abs(z) / 5.0))
                anomalies.append(
                    {
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "label": "irregular_spacing",
                        "score": score,
                        "line_key": line_key,
                        "height_z": float(z),
                        "text": word.get("text", ""),
                    }
                )

    LOGGER.info("Text analysis found %d font inconsistency anomalies.", len(anomalies))
    return anomalies
