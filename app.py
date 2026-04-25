"""Streamlit app for document forgery detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure the project virtual environment packages are importable even when
# Streamlit is launched from a different Python entrypoint.
_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE_PACKAGES.exists() and str(_VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE_PACKAGES))

import cv2
import numpy as np
import streamlit as st

from anomaly_detector import detect_suspicious_regions
from pdf_loader import load_document
from preprocess import preprocess_image
from visualize import draw_bounding_boxes

BoundingBox = Tuple[int, int, int, int]


ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for Streamlit display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_uploaded_to_temp(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Persist uploaded file to a temporary path and return that path.

    The returned file should be deleted by the caller after processing.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def run_detection_pipeline(input_path: Path) -> Tuple[np.ndarray, np.ndarray, List[BoundingBox]]:
    """Run the forgery detection pipeline on an input file path.

    Returns:
        original_bgr: Original loaded image in BGR.
        output_bgr: Image with suspicious region bounding boxes in BGR.
        boxes: Detected suspicious bounding boxes.
    """
    original_bgr, _ = load_document(input_path)
    processed_gray, resized_bgr, _ = preprocess_image(original_bgr, target_width=1024)
    boxes = detect_suspicious_regions(processed_gray)
    output_bgr = draw_bounding_boxes(resized_bgr, boxes)
    return original_bgr, output_bgr, boxes


def render_regions(boxes: List[BoundingBox]) -> None:
    """Render detected suspicious regions in the UI."""
    st.subheader("Detected Regions")

    if not boxes:
        st.info("No suspicious regions detected.")
        return

    region_rows = [
        {
            "Region": idx,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": w * h,
        }
        for idx, (x, y, w, h) in enumerate(boxes, start=1)
    ]
    st.dataframe(region_rows, use_container_width=True)


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="Document Forgery Detection System", layout="wide")

    st.title("Document Forgery Detection System")
    st.write("Upload a PDF or image to detect suspicious/tampered regions.")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Please upload a file to begin.")
        return

    temp_path: Path | None = None
    try:
        with st.spinner("Processing document..."):
            temp_path = save_uploaded_to_temp(uploaded_file)
            original_bgr, output_bgr, boxes = run_detection_pipeline(temp_path)

        st.success("Processing complete.")
        st.metric("Suspicious Regions Detected", len(boxes))

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Original Image")
            st.image(bgr_to_rgb(original_bgr), use_container_width=True)

        with col_right:
            st.subheader("Output with Bounding Boxes")
            st.image(bgr_to_rgb(output_bgr), use_container_width=True)

        render_regions(boxes)

    except Exception as exc:
        st.error(f"Failed to process file: {exc}")
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                # Non-fatal cleanup issue; no need to block the app.
                pass


if __name__ == "__main__":
    main()
 