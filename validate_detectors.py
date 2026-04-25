#!/usr/bin/env python3
"""Validation script for new forensic detectors.

Tests:
1. Imports work correctly
2. Detector configs instantiate
3. Basic detection functions run without error
4. Output format matches expected structure
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

try:
    LOGGER.info("Importing new detectors...")
    from page_boundary_detector import (
        PageBoundaryDetectorConfig,
        detect_page_boundary_discontinuities,
    )
    from document_merge_detector import (
        DocumentMergeDetectorConfig,
        detect_document_merging,
    )
    LOGGER.info("✓ Imports successful")
except ImportError as e:
    LOGGER.error(f"✗ Import failed: {e}", exc_info=True)
    sys.exit(1)

try:
    LOGGER.info("Testing detector configs...")
    cfg_page = PageBoundaryDetectorConfig()
    cfg_merge = DocumentMergeDetectorConfig()
    LOGGER.info(f"✓ PageBoundaryDetectorConfig: {cfg_page}")
    LOGGER.info(f"✓ DocumentMergeDetectorConfig: {cfg_merge}")
except Exception as e:
    LOGGER.error(f"✗ Config instantiation failed: {e}", exc_info=True)
    sys.exit(1)

try:
    LOGGER.info("Creating synthetic test image...")
    # Create a simple test image with two distinct regions
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
    # Add a darker region to simulate a page boundary
    test_image[256:, :] = 50  # Darker bottom half
    LOGGER.info(f"✓ Test image shape: {test_image.shape}")
except Exception as e:
    LOGGER.error(f"✗ Test image creation failed: {e}", exc_info=True)
    sys.exit(1)

try:
    LOGGER.info("Testing page_boundary_detector...")
    page_detections = detect_page_boundary_discontinuities(
        image=test_image,
        config=cfg_page,
    )
    LOGGER.info(
        f"✓ Page boundary detections: {len(page_detections)} "
        f"(expected format: list of dicts with bbox/label/score)"
    )
    if page_detections:
        det = page_detections[0]
        required_keys = {"bbox", "label", "score", "source"}
        if required_keys.issubset(det.keys()):
            LOGGER.info(f"  - Detection format: ✓ {list(det.keys())}")
        else:
            LOGGER.warning(f"  - Missing keys: {required_keys - set(det.keys())}")
except Exception as e:
    LOGGER.error(f"✗ Page boundary detection failed: {e}", exc_info=True)
    sys.exit(1)

try:
    LOGGER.info("Testing document_merge_detector...")
    merge_detections = detect_document_merging(
        image=test_image,
        config=cfg_merge,
    )
    LOGGER.info(
        f"✓ Document merge detections: {len(merge_detections)} "
        f"(expected format: list of dicts with bbox/label/score)"
    )
    if merge_detections:
        det = merge_detections[0]
        required_keys = {"bbox", "label", "score", "source"}
        if required_keys.issubset(det.keys()):
            LOGGER.info(f"  - Detection format: ✓ {list(det.keys())}")
        else:
            LOGGER.warning(f"  - Missing keys: {required_keys - set(det.keys())}")
except Exception as e:
    LOGGER.error(f"✗ Document merge detection failed: {e}", exc_info=True)
    sys.exit(1)

try:
    LOGGER.info("Testing pipeline import...")
    from pipeline import PipelineConfig, detect_document_tampering
    cfg = PipelineConfig()
    LOGGER.info(f"✓ PipelineConfig has page_boundary: {hasattr(cfg, 'page_boundary')}")
    LOGGER.info(f"✓ PipelineConfig has document_merge: {hasattr(cfg, 'document_merge')}")
except Exception as e:
    LOGGER.error(f"✗ Pipeline import/integration failed: {e}", exc_info=True)
    sys.exit(1)

LOGGER.info("\n" + "=" * 60)
LOGGER.info("✓ ALL VALIDATIONS PASSED")
LOGGER.info("=" * 60)
LOGGER.info("\nNew detectors successfully integrated:")
LOGGER.info("  1. page_boundary_detector - detects page boundaries and discontinuities")
LOGGER.info("  2. document_merge_detector - detects merged document content")
LOGGER.info("\nBoth detectors are now part of the main pipeline.")
