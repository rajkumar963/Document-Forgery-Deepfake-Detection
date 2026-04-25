"""Visualization helpers for multi-label document forgery detections."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

BoundingBox = Tuple[int, int, int, int]

ADDED_CONTENT_LABELS = {"stamp", "signature", "seal", "added_mark"}


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Get BGR color for a detection label.

    Color mapping:
        suspicious_region -> red
        copy_paste -> blue
        irregular_spacing -> green
        stamp/signature/seal/added_mark -> orange
        overwritten_text -> magenta
        erased_content -> cyan
        ai_edited_region -> purple
    """
    if label == "copy_paste":
        return (255, 0, 0)
    if label == "irregular_spacing":
        return (0, 255, 0)
    if label == "overwritten_text":
        return (255, 0, 255)
    if label == "erased_content":
        return (255, 255, 0)
    if label == "ai_edited_region":
        return (200, 0, 200)
    if label in ADDED_CONTENT_LABELS:
        return (0, 165, 255)
    return (0, 0, 255)


def _clip_box(box: BoundingBox, image_shape: tuple[int, int]) -> BoundingBox | None:
    """Clip bounding box to image dimensions."""
    x, y, w, h = box
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


def draw_labeled_boxes(
    image: np.ndarray,
    detections: Iterable[Dict],
    thickness: int = 2,
) -> np.ndarray:
    """Draw labeled boxes with score and source metadata."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    canvas = image.copy()
    height, width = canvas.shape[:2]

    for det in detections:
        if "bbox" not in det or "label" not in det:
            continue

        clipped = _clip_box(tuple(det["bbox"]), image_shape=(height, width))
        if clipped is None:
            continue

        x, y, w, h = clipped
        label = str(det.get("label", "unknown"))
        score = float(det.get("score", 0.0))
        source = str(det.get("source", ""))

        color = get_color_for_label(label)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        caption = f"{label}:{score:.2f}"
        if source:
            caption = f"{caption} [{source}]"

        text_y = y - 8
        if text_y < 12:
            text_y = min(height - 6, y + 14)

        cv2.putText(
            canvas,
            caption,
            (max(0, x), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_detection_visualization(
    image: np.ndarray,
    detections: Iterable[Dict],
    output_path: str | Path,
    yolo_only: bool = False,
) -> Path:
    """Save detection visualization image.

    Args:
        image: Input BGR image.
        detections: Detection dictionaries.
        output_path: Destination image path.
        yolo_only: If True, draw only added-content labels.
    """
    if yolo_only:
        selected = [det for det in detections if str(det.get("label", "")) in ADDED_CONTENT_LABELS]
    else:
        selected = list(detections)

    drawn = draw_labeled_boxes(image, selected)

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(out), drawn)
    if not success:
        raise IOError(f"Failed to save visualization image: {out}")

    return out


def draw_final_detections(
    image: np.ndarray,
    final_detections: Iterable[Dict],
    show_supporting_sources: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw final refined detections with supporting source info.

    Args:
        image: Input BGR image.
        final_detections: Final refined detections (from label_refiner).
        show_supporting_sources: If True, show supporting sources below label.
        thickness: Thickness of bounding box lines.

    Returns:
        Annotated image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    canvas = image.copy()
    height, width = canvas.shape[:2]

    for det in final_detections:
        if "bbox" not in det or "label" not in det:
            continue

        clipped = _clip_box(tuple(det["bbox"]), image_shape=(height, width))
        if clipped is None:
            continue

        x, y, w, h = clipped
        label = str(det.get("label", "unknown"))
        score = float(det.get("score", 0.0))
        supporting_sources = det.get("supporting_sources", [])

        color = get_color_for_label(label)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        # Main label with score
        caption = f"{label}: {score:.3f}"
        text_y = y - 20

        if text_y < 12:
            text_y = min(height - 6, y + 14)

        cv2.putText(
            canvas,
            caption,
            (max(0, x), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        # Supporting sources (optional)
        if show_supporting_sources and supporting_sources:
            sources_str = ", ".join(supporting_sources)
            src_text = f"src: {sources_str}"
            src_y = text_y + 16

            if src_y < height - 4:
                cv2.putText(
                    canvas,
                    src_text,
                    (max(0, x), src_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    return canvas


def draw_intermediate_detections(
    image: np.ndarray,
    intermediate_detections: Iterable[Dict],
    thickness: int = 1,
    alpha: float = 0.6,
) -> np.ndarray:
    """Draw intermediate detections (before refinement) with transparency.

    Args:
        image: Input BGR image.
        intermediate_detections: Detections before refinement (for debugging).
        thickness: Thickness of bounding box lines.
        alpha: Transparency factor [0, 1].

    Returns:
        Annotated image with transparent overlay.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    overlay = image.copy()
    height, width = overlay.shape[:2]

    for det in intermediate_detections:
        if "bbox" not in det or "label" not in det:
            continue

        clipped = _clip_box(tuple(det["bbox"]), image_shape=(height, width))
        if clipped is None:
            continue

        x, y, w, h = clipped
        label = str(det.get("label", "unknown"))
        score = float(det.get("score", 0.0))

        color = get_color_for_label(label)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

        caption = f"{label}:{score:.2f}"
        text_y = y - 8 if (y - 8) >= 12 else min(height - 6, y + 14)

        cv2.putText(
            overlay,
            caption,
            (max(0, x), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def save_pipeline_visualizations(
    image: np.ndarray,
    intermediate_detections: List[Dict],
    final_detections: List[Dict],
    output_dir: str | Path,
    base_filename: str = "detection",
) -> Dict[str, Path]:
    """Save both intermediate and final detection visualizations.

    Args:
        image: Processed BGR image.
        intermediate_detections: Detections before refinement.
        final_detections: Refined final detections.
        output_dir: Output directory path.
        base_filename: Base name for output files (without extension).

    Returns:
        Dictionary mapping {"intermediate": Path, "final": Path}
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Save intermediate visualization (for debugging)
    if intermediate_detections:
        intermediate_vis = draw_intermediate_detections(image, intermediate_detections)
        intermediate_path = out_dir / f"{base_filename}_intermediate.png"
        cv2.imwrite(str(intermediate_path), intermediate_vis)
        results["intermediate"] = intermediate_path

    # Save final visualization (main output)
    final_vis = draw_final_detections(image, final_detections, show_supporting_sources=True)
    final_path = out_dir / f"{base_filename}_final.png"
    cv2.imwrite(str(final_path), final_vis)
    results["final"] = final_path

    return results
