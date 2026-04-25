"""YOLO-based added content detector for document forgery analysis.

Detects stamp, signature, seal, and added_mark regions using Ultralytics YOLOv8.
Also provides a lightweight training helper for custom dataset fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

BoundingBox = Tuple[int, int, int, int]
ADDED_CONTENT_CLASSES = ["stamp", "signature", "seal", "added_mark"]


@dataclass
class AddedContentConfig:
    """Configuration for YOLO added-content inference."""

    model_path: str | None = None
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str | None = None


def _clip_box_to_bounds(box: BoundingBox, image_shape: tuple[int, int]) -> BoundingBox | None:
    """Clip a box to image bounds and return None if invalid."""
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


def load_yolo_model(model_path: str | Path | None):
    """Load an Ultralytics YOLO model from a path.

    Returns:
        Loaded YOLO model instance, or None if unavailable.
    """
    if not ULTRALYTICS_AVAILABLE:
        LOGGER.warning("Ultralytics is not installed. Added-content detection is disabled.")
        return None

    if not model_path:
        LOGGER.warning("No YOLO model path provided. Added-content detection is skipped.")
        return None

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        LOGGER.warning("YOLO model file not found at: %s", path)
        return None

    try:
        model = YOLO(str(path))
        LOGGER.info("Loaded YOLO model from: %s", path)
        return model
    except Exception as exc:
        LOGGER.error("Failed to load YOLO model from %s: %s", path, exc)
        return None


def filter_low_confidence_detections(
    detections: List[Dict[str, Any]],
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """Filter detections by confidence threshold."""
    threshold = float(max(0.0, min(1.0, confidence_threshold)))
    return [det for det in detections if float(det.get("score", 0.0)) >= threshold]


def convert_yolo_results_to_boxes(
    results: Any,
    image_shape: tuple[int, int],
) -> List[Dict[str, Any]]:
    """Convert Ultralytics YOLO results to normalized detection dictionaries."""
    detections: List[Dict[str, Any]] = []

    if results is None:
        return detections

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        names = getattr(result, "names", {})
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []

        for box_arr, score, cls_id in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = [float(v) for v in box_arr]
            label = str(names.get(int(cls_id), f"class_{int(cls_id)}"))

            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            raw_box = (int(round(x1)), int(round(y1)), w, h)
            clipped = _clip_box_to_bounds(raw_box, image_shape=image_shape)
            if clipped is None:
                continue

            detections.append(
                {
                    "bbox": clipped,
                    "label": label,
                    "score": float(score),
                    "source": "yolo",
                }
            )

    return detections


def detect_added_content(
    image: np.ndarray,
    model_path: str | Path | None,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    device: str | None = None,
) -> List[Dict[str, Any]]:
    """Run YOLO inference to detect stamps/signatures/seals/added marks.

    Args:
        image: Input BGR image.
        model_path: Path to trained YOLO model weights.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold for YOLO predictor.
        device: Optional inference device string, e.g. "cpu" or "0".

    Returns:
        Detection dictionaries with bbox, label, score, and source.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image for added-content detection is empty or None.")

    model = load_yolo_model(model_path)
    if model is None:
        return []

    try:
        kwargs: Dict[str, Any] = {
            "conf": float(confidence_threshold),
            "iou": float(iou_threshold),
            "verbose": False,
        }
        if device:
            kwargs["device"] = device

        results = model.predict(source=image, **kwargs)
        detections = convert_yolo_results_to_boxes(results, image_shape=image.shape[:2])
        detections = filter_low_confidence_detections(detections, confidence_threshold=confidence_threshold)

        # Keep expected classes only and discard unknown labels.
        allowed = set(ADDED_CONTENT_CLASSES)
        detections = [det for det in detections if det["label"] in allowed]

        class_counts: Dict[str, int] = {}
        for det in detections:
            class_counts[det["label"]] = class_counts.get(det["label"], 0) + 1

        LOGGER.info("YOLO added-content detections: %d", len(detections))
        if class_counts:
            LOGGER.info("YOLO class counts: %s", class_counts)

        return detections
    except Exception as exc:
        LOGGER.error("YOLO inference failed: %s", exc)
        return []


def train_added_content_detector(
    data_yaml_path: str | Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/added_content",
    name: str = "day5_train",
    device: str | None = None,
) -> Path | None:
    """Train a YOLOv8 model for added-content classes.

    Args:
        data_yaml_path: Path to dataset YAML file.
        model_name: Base model weights or config.
        epochs: Number of training epochs.
        imgsz: Training image size.
        batch: Batch size.
        project: Ultralytics project output directory.
        name: Run name.
        device: Optional device string.

    Returns:
        Path to best weights if training succeeds, else None.
    """
    if not ULTRALYTICS_AVAILABLE:
        LOGGER.error("Ultralytics is not installed. Cannot train YOLO model.")
        return None

    data_path = Path(data_yaml_path).expanduser().resolve()
    if not data_path.exists():
        LOGGER.error("Dataset yaml not found: %s", data_path)
        return None

    try:
        model = YOLO(model_name)
        train_kwargs: Dict[str, Any] = {
            "data": str(data_path),
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "project": str(project),
            "name": str(name),
            "verbose": True,
        }
        if device:
            train_kwargs["device"] = device

        train_result = model.train(**train_kwargs)
        best_path = getattr(train_result, "best", None)

        if best_path is None:
            LOGGER.warning("Training finished but best weights path was not returned.")
            return None

        best_weights = Path(best_path).resolve()
        LOGGER.info("Training complete. Best model: %s", best_weights)
        return best_weights
    except Exception as exc:
        LOGGER.error("YOLO training failed: %s", exc)
        return None
