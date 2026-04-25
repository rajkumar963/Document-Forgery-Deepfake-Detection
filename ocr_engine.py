"""OCR extraction engine for document text analysis.

Uses pytesseract to extract word-level text, bounding boxes, confidence,
and line metadata for downstream tampering analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

LOGGER = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]


@dataclass
class OCREngineConfig:
    """Configuration for OCR extraction."""

    min_confidence: float = 45.0
    language: str = "eng"
    psm: int = 6
    oem: int = 3


@dataclass
class OCRWord:
    """Normalized word-level OCR output."""

    text: str
    bbox: BoundingBox
    conf: float
    line_num: int
    word_num: int
    block_num: int
    par_num: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert OCRWord to dictionary output format."""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "conf": float(self.conf),
            "line_num": int(self.line_num),
            "word_num": int(self.word_num),
            "block_num": int(self.block_num),
            "par_num": int(self.par_num),
        }


def _prepare_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Prepare image for OCR while preserving document details."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Input image must be a grayscale or BGR image.")

    # Mild denoising and contrast enhancement improves OCR robustness.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def extract_ocr_words(
    image: np.ndarray,
    config: OCREngineConfig | None = None,
) -> List[Dict[str, Any]]:
    """Extract word-level OCR entries using pytesseract.image_to_data.

    Args:
        image: Input grayscale or BGR image.
        config: Optional OCR configuration.

    Returns:
        List of dictionaries in format:
            {
                "text": str,
                "bbox": (x, y, w, h),
                "conf": float,
                "line_num": int,
                "word_num": int,
                "block_num": int,
                "par_num": int,
            }

    Raises:
        RuntimeError: If Tesseract OCR engine is unavailable.
        ValueError: If image is invalid.
    """
    cfg = config or OCREngineConfig()
    prepared = _prepare_image_for_ocr(image)

    tess_config = f"--oem {int(cfg.oem)} --psm {int(cfg.psm)}"

    try:
        data = pytesseract.image_to_data(
            prepared,
            lang=cfg.language,
            config=tess_config,
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract executable was not found. Install Tesseract OCR and add it to PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc

    n_items = len(data.get("text", []))
    words: List[OCRWord] = []

    for idx in range(n_items):
        raw_text = str(data["text"][idx]).strip()
        if not raw_text:
            continue

        try:
            conf = float(data["conf"][idx])
        except (ValueError, TypeError):
            conf = -1.0

        if conf < cfg.min_confidence:
            continue

        x = int(data["left"][idx])
        y = int(data["top"][idx])
        w = int(data["width"][idx])
        h = int(data["height"][idx])

        if w <= 0 or h <= 0:
            continue

        words.append(
            OCRWord(
                text=raw_text,
                bbox=(x, y, w, h),
                conf=conf,
                line_num=int(data.get("line_num", [0])[idx]),
                word_num=int(data.get("word_num", [0])[idx]),
                block_num=int(data.get("block_num", [0])[idx]),
                par_num=int(data.get("par_num", [0])[idx]),
            )
        )

    LOGGER.info("OCR extracted %d high-confidence words (raw=%d).", len(words), n_items)
    return [word.to_dict() for word in words]
