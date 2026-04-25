"""Input loading utilities for document forgery detection.

Supports image files and PDFs. For PDFs, the first page is converted to a
NumPy BGR image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Optional PDF backends. We try PyMuPDF first, then pdf2image.
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - depends on environment
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - depends on environment
    convert_from_path = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _validate_input_path(file_path: str | Path) -> Path:
    """Validate and normalize an input file path.

    Args:
        file_path: Path to an image or PDF file.

    Returns:
        Normalized ``Path`` object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the input path is not a file.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    return path


def _load_image(image_path: Path) -> np.ndarray:
    """Load an image file into a BGR NumPy array.

    Args:
        image_path: Path to an image file.

    Returns:
        Loaded image as ``np.ndarray`` in BGR channel order.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Failed to load image or image is empty: {image_path}")
    return image


def _load_pdf_first_page_pymupdf(pdf_path: Path) -> np.ndarray:
    """Load the first page of a PDF as a BGR image using PyMuPDF.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        First page as a BGR NumPy image.

    Raises:
        RuntimeError: If PyMuPDF is unavailable or conversion fails.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed.")

    try:
        document = fitz.open(str(pdf_path))
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        page = document.load_page(0)
        pix = page.get_pixmap(dpi=200)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Handle RGB or RGBA output.
        if pix.n == 4:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return bgr
    except Exception as exc:  # pragma: no cover - depends on input/backend
        raise RuntimeError(f"Failed to read PDF via PyMuPDF: {exc}") from exc


def _load_pdf_first_page_pdf2image(pdf_path: Path) -> np.ndarray:
    """Load the first page of a PDF as a BGR image using pdf2image.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        First page as a BGR NumPy image.

    Raises:
        RuntimeError: If pdf2image is unavailable or conversion fails.
    """
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed.")

    try:
        pages = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
        if not pages:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        pil_image = pages[0]
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as exc:  # pragma: no cover - depends on input/backend
        raise RuntimeError(f"Failed to read PDF via pdf2image: {exc}") from exc


def _load_pdf_first_page(pdf_path: Path) -> np.ndarray:
    """Load first PDF page to BGR image, trying available backends.

    Backend preference:
        1. PyMuPDF (fitz)
        2. pdf2image

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        First page as BGR NumPy array.

    Raises:
        RuntimeError: If no backend works.
    """
    errors: list[str] = []

    try:
        return _load_pdf_first_page_pymupdf(pdf_path)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(str(exc))

    try:
        return _load_pdf_first_page_pdf2image(pdf_path)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(str(exc))

    raise RuntimeError(
        "Unable to load PDF first page. Install either PyMuPDF or pdf2image "
        f"and verify dependencies. Backend errors: {errors}"
    )


def load_document(file_path: str | Path) -> Tuple[np.ndarray, Path]:
    """Load an input document (image or PDF) as a BGR NumPy array.

    Args:
        file_path: Path to input image or PDF.

    Returns:
        Tuple of:
            - image: Loaded BGR image as ``np.ndarray``
            - resolved_path: Resolved absolute ``Path`` to input

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file type is unsupported or image is invalid.
        RuntimeError: If PDF conversion fails.
    """
    path = _validate_input_path(file_path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        image = _load_image(path)
        return image, path

    if suffix in PDF_EXTENSIONS:
        image = _load_pdf_first_page(path)
        if image is None or image.size == 0:
            raise ValueError(f"Converted PDF image is empty: {path}")
        return image, path

    supported = sorted(IMAGE_EXTENSIONS.union(PDF_EXTENSIONS))
    raise ValueError(f"Unsupported file extension '{suffix}'. Supported: {supported}")
