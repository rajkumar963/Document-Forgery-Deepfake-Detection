"""Unified Day 8 tampering detection pipeline with final label refinement.

Integrates:
- anomaly-based suspicious regions
- copy-move regions
- OCR spacing irregularity regions
- YOLO added-content detections (stamp/signature/seal/added_mark)
- AI-generation document-level detection
- AI-editing region-level detection
- Page boundary discontinuity detection
- Document merging detection
- Final label refinement and decision layer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from added_content_detector import AddedContentConfig, detect_added_content
from ai_edited_region_detector import RegionAIEditConfig, detect_ai_edited_regions_from_boxes
from ai_generated_detector import DocumentClassifierConfig, detect_fully_ai_generated_document
from anomaly_detector import detect_suspicious_regions
from anomaly_map import ThresholdStrategy, build_anomaly_outputs
from bbox_utils import BoundingBox, boxes_from_mask, merge_overlapping_boxes, non_max_suppression
from confidence_utils import ConfidenceConfig, filter_detections_by_score
from copy_move_detector import CopyMoveConfig, detect_copy_move_regions
from document_merge_detector import DocumentMergeDetectorConfig, detect_document_merging
from erased_overwritten_classifier import RegionClassifierConfig, classify_suspicious_regions
from features import DEFAULT_FEATURE_WEIGHTS, extract_patch_anomaly_map
from label_refiner import LabelRefinerConfig, refine_final_labels
from label_standardizer import standardize_pipeline_output
from page_boundary_detector import PageBoundaryDetectorConfig, detect_page_boundary_discontinuities
from pdf_loader import load_document
from preprocess import preprocess_image
from region_fusion import (
    attach_refined_labels_to_regions,
    fuse_document_level_result,
    fuse_region_level_results,
    merge_refined_detections,
)
from spacing_detector import SpacingDetectorConfig, detect_irregular_text_regions

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for anomaly feature extraction."""

    patch_size: int = 32
    stride: int = 16
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS))


@dataclass
class AnomalyMapConfig:
    """Configuration for anomaly map post-processing."""

    threshold_strategy: ThresholdStrategy = "percentile"
    threshold_value: float = 0.60
    percentile: float = 90.0
    smoothing_ksize: int = 7
    adaptive_block_size: int = 35
    adaptive_c: float = -2.0
    min_region_area: int = 200
    morph_kernel_size: int = 5


@dataclass
class BoxConfig:
    """Configuration for anomaly boxes."""

    min_contour_area: int = 120
    min_box_area: int = 250
    min_width: int = 10
    min_height: int = 10
    merge_iou_threshold: float = 0.20
    merge_max_gap: int = 12
    nms_iou_threshold: float = 0.45


@dataclass
class FusionConfig:
    """Configuration for label-aware detection fusion."""

    merge_iou_threshold: float = 0.30
    dedupe_iou_threshold: float = 0.85
    attach_refined_iou_threshold: float = 0.20


@dataclass
class PipelineConfig:
    """Top-level Day 8 pipeline configuration with final refinement."""

    target_width: int = 1024
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    anomaly_map: AnomalyMapConfig = field(default_factory=AnomalyMapConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    copy_move: CopyMoveConfig = field(default_factory=CopyMoveConfig)
    spacing: SpacingDetectorConfig = field(default_factory=SpacingDetectorConfig)
    added_content: AddedContentConfig = field(default_factory=AddedContentConfig)
    region_classifier: RegionClassifierConfig = field(default_factory=RegionClassifierConfig)
    document_ai_classifier: DocumentClassifierConfig = field(default_factory=DocumentClassifierConfig)
    region_ai_edit_classifier: RegionAIEditConfig = field(default_factory=RegionAIEditConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    label_refiner: LabelRefinerConfig = field(default_factory=LabelRefinerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    page_boundary: PageBoundaryDetectorConfig = field(default_factory=PageBoundaryDetectorConfig)
    document_merge: DocumentMergeDetectorConfig = field(default_factory=DocumentMergeDetectorConfig)
    combine_day1_contours: bool = True
    day1_min_area: int = 500


def _run_anomaly_stage(
    preprocessed_gray: np.ndarray,
    processed_bgr: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, list[BoundingBox]]:
    """Run anomaly pipeline stage and return map, mask, and boxes."""
    feature_map = extract_patch_anomaly_map(
        preprocessed_gray=preprocessed_gray,
        patch_size=cfg.feature.patch_size,
        stride=cfg.feature.stride,
        feature_weights=cfg.feature.weights,
    )

    anomaly_map, binary_mask, _ = build_anomaly_outputs(
        feature_map=feature_map,
        threshold_strategy=cfg.anomaly_map.threshold_strategy,
        threshold_value=cfg.anomaly_map.threshold_value,
        percentile=cfg.anomaly_map.percentile,
        smoothing_ksize=cfg.anomaly_map.smoothing_ksize,
        adaptive_block_size=cfg.anomaly_map.adaptive_block_size,
        adaptive_c=cfg.anomaly_map.adaptive_c,
        min_region_area=cfg.anomaly_map.min_region_area,
        morph_kernel_size=cfg.anomaly_map.morph_kernel_size,
    )

    anomaly_boxes = boxes_from_mask(
        binary_mask=binary_mask,
        image_shape=processed_bgr.shape[:2],
        min_contour_area=cfg.box.min_contour_area,
        min_box_area=cfg.box.min_box_area,
        min_width=cfg.box.min_width,
        min_height=cfg.box.min_height,
        merge_iou_threshold=cfg.box.merge_iou_threshold,
        merge_max_gap=cfg.box.merge_max_gap,
        nms_iou_threshold=cfg.box.nms_iou_threshold,
    )

    if cfg.combine_day1_contours:
        day1_boxes = detect_suspicious_regions(preprocessed_gray, min_area=cfg.day1_min_area)
        anomaly_boxes = merge_overlapping_boxes(
            anomaly_boxes + day1_boxes,
            iou_threshold=cfg.box.merge_iou_threshold,
            max_gap=cfg.box.merge_max_gap,
        )
        anomaly_boxes = non_max_suppression(anomaly_boxes, iou_threshold=cfg.box.nms_iou_threshold)

    return anomaly_map, binary_mask, anomaly_boxes


def detect_document_tampering(
    input_path: str | Path,
    config: PipelineConfig | None = None,
) -> Dict:
    """Run full Day 8 tampering detection pipeline with final refinement.

    Pipeline stages:
    1. Load and preprocess image
    2. Run document-level AI detection
    3. Run anomaly detector
    4. Run copy-move detector
    5. Run OCR spacing detector
    6. Run YOLO added-content detector
    7. Run erased/overwritten classifier
    8. Run region-level AI-edit detector
    9. Fuse intermediate detections
    10. Apply final label refinement

    Returns:
        {
            "processed_image": np.ndarray,
            "anomaly_map": np.ndarray,
            "binary_mask": np.ndarray,
            "document_label": {
                "label": "fully_ai_generated_document" or "authentic_or_unknown",
                "score": float
            },
            "intermediate_detections": [...],
            "final_detections": [
                {
                    "bbox": (x,y,w,h),
                    "label": str,
                    "score": float,
                    "source": "final_refiner",
                    "supporting_sources": [str, ...]
                }
            ],
            "source_path": Path,
        }
    """
    cfg = config or PipelineConfig()

    try:
        LOGGER.info("Loading input document...")
        image_bgr, resolved_path = load_document(input_path)

        LOGGER.info("Preprocessing image...")
        preprocessed_gray, processed_bgr, _ = preprocess_image(
            image_bgr,
            target_width=cfg.target_width,
        )

        LOGGER.info("Running document-level AI-generation detector...")
        document_ai_result = detect_fully_ai_generated_document(
            preprocessed_gray,
            config=cfg.document_ai_classifier,
        )

        LOGGER.info("Running anomaly detector...")
        anomaly_map, binary_mask, anomaly_boxes = _run_anomaly_stage(
            preprocessed_gray=preprocessed_gray,
            processed_bgr=processed_bgr,
            cfg=cfg,
        )

        LOGGER.info("Running copy-move detector...")
        copy_move_result = detect_copy_move_regions(preprocessed_gray, config=cfg.copy_move)
        copy_move_boxes = list(copy_move_result.get("boxes", []))

        LOGGER.info("Running OCR spacing detector...")
        try:
            spacing_regions = detect_irregular_text_regions(preprocessed_gray, config=cfg.spacing)
        except RuntimeError as exc:
            if "Tesseract" in str(exc) or "tesseract" in str(exc):
                LOGGER.warning("OCR spacing detector skipped: %s", exc)
                spacing_regions = []
            else:
                raise

        LOGGER.info("Running YOLO added-content detector...")
        yolo_detections = detect_added_content(
            image=processed_bgr,
            model_path=cfg.added_content.model_path,
            confidence_threshold=cfg.added_content.confidence_threshold,
            iou_threshold=cfg.added_content.iou_threshold,
            device=cfg.added_content.device,
        )

        LOGGER.info("Running erased/overwritten region classifier...")
        classified_regions = classify_suspicious_regions(
            original_image=processed_bgr,
            preprocessed_gray=preprocessed_gray,
            candidate_boxes=anomaly_boxes,
            config=cfg.region_classifier,
        )

        LOGGER.info("Running region-level AI-edit detector...")
        ai_edit_detections = detect_ai_edited_regions_from_boxes(
            gray_image=preprocessed_gray,
            suspicious_boxes=anomaly_boxes,
            config=cfg.region_ai_edit_classifier,
        )

        LOGGER.info("Running page boundary discontinuity detector...")
        page_boundary_detections = detect_page_boundary_discontinuities(
            image=processed_bgr,
            config=cfg.page_boundary,
        )

        LOGGER.info("Running document merging detector...")
        document_merge_detections = detect_document_merging(
            image=processed_bgr,
            config=cfg.document_merge,
        )

        anomaly_detections = [
            {
                "bbox": box,
                "label": "suspicious_region",
                "score": 0.67,
                "source": "anomaly",
            }
            for box in anomaly_boxes
        ]

        copy_move_detections = [
            {
                "bbox": box,
                "label": "copy_paste",
                "score": 0.89,
                "source": "copy_move",
            }
            for box in copy_move_boxes
        ]

        spacing_detections = [
            {
                "bbox": tuple(det["bbox"]),
                "label": "irregular_spacing",
                "score": float(det.get("score", 0.72)),
                "source": "ocr_spacing",
            }
            for det in spacing_regions
        ]

        refined_anomaly_detections = attach_refined_labels_to_regions(
            base_detections=anomaly_detections,
            refined_detections=classified_regions,
            iou_threshold=cfg.fusion.attach_refined_iou_threshold,
        )

        # Combine AI-edit detections with other detections
        fused_with_ai = fuse_region_level_results(
            ai_edit_detections=ai_edit_detections,
            other_detections=refined_anomaly_detections + copy_move_detections + spacing_detections + yolo_detections + page_boundary_detections + document_merge_detections,
            merge_iou_threshold=cfg.fusion.merge_iou_threshold,
        )

        LOGGER.info(
            "Intermediate detections: anomaly=%d, erased/overwritten=%d, copy_move=%d, ocr=%d, yolo=%d, ai_edit=%d, page_boundary=%d, doc_merge=%d",
            len(anomaly_detections),
            len(classified_regions),
            len(copy_move_detections),
            len(spacing_detections),
            len(yolo_detections),
            sum(1 for d in ai_edit_detections if "skip_reason" not in d),
            len(page_boundary_detections),
            len(document_merge_detections),
        )

        LOGGER.info("Detections before refinement: %d", len(fused_with_ai))

        # Apply final label refinement
        LOGGER.info("Applying final label refinement and decision layer...")
        image_height, image_width = processed_bgr.shape[:2]
        final_detections = refine_final_labels(
            detections=fused_with_ai,
            image_shape=(image_height, image_width),
            config=cfg.label_refiner,
        )

        # Filter by confidence thresholds
        final_detections = filter_detections_by_score(
            detections=final_detections,
            config=cfg.confidence,
        )

        LOGGER.info(
            "Document-level AI-generation: label=%s score=%.3f",
            document_ai_result.get("label"),
            document_ai_result.get("score", 0.0),
        )
        LOGGER.info("Final refined detections after filtering: %d", len(final_detections))

        raw_output = {
            "processed_image": processed_bgr,
            "anomaly_map": anomaly_map,
            "binary_mask": binary_mask,
            "document_label": {
                "label": str(document_ai_result.get("label", "authentic_or_unknown")),
                "score": float(document_ai_result.get("score", 0.0)),
            },
            "intermediate_detections": fused_with_ai,
            "final_detections": final_detections,
            "source_path": resolved_path,
        }
        
        # Standardize all labels to canonical hackathon format
        standardized_output = standardize_pipeline_output(raw_output)
        
        return standardized_output
    except Exception as exc:
        LOGGER.exception("Day 8 tampering pipeline failed.")
        raise RuntimeError(f"Failed to detect tampering for '{input_path}': {exc}") from exc
