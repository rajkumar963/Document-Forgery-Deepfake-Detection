"""Day 8 command-line entrypoint for document forgery detection with final refinement."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from added_content_detector import AddedContentConfig
from ai_edited_region_detector import RegionAIEditConfig
from ai_generated_detector import DocumentClassifierConfig
from confidence_utils import ConfidenceConfig
from copy_move_detector import CopyMoveConfig
from erased_overwritten_classifier import RegionClassifierConfig
from label_refiner import LabelRefinerConfig
from pipeline import AnomalyMapConfig, BoxConfig, FeatureConfig, FusionConfig, PipelineConfig, detect_document_tampering
from spacing_detector import SpacingDetectorConfig
from visualization import draw_final_detections, save_pipeline_visualizations, save_detection_visualization

LOGGER = logging.getLogger(__name__)


def _configure_logging(log_level: str = "INFO") -> None:
    """Configure logging for CLI execution."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _save_anomaly_heatmap(anomaly_map: np.ndarray, output_path: Path) -> Path:
    """Save anomaly heatmap as colormap image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heat_u8 = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    success = cv2.imwrite(str(output_path), heat_bgr)
    if not success:
        raise IOError(f"Failed to save anomaly heatmap: {output_path}")

    return output_path.resolve()


def _save_binary_mask(binary_mask: np.ndarray, output_path: Path) -> Path:
    """Save binary suspicious mask image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), binary_mask)
    if not success:
        raise IOError(f"Failed to save binary mask: {output_path}")

    return output_path.resolve()


def _clip_bbox_to_image(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Clip bbox to image bounds and return None if invalid."""
    x, y, w, h = bbox
    height, width = image_shape

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _save_debug_classified_crops(
    image: np.ndarray,
    classified_regions: List[Dict],
    output_dir: Path,
) -> int:
    """Save debug crops for erased/overwritten classified regions."""
    debug_dir = output_dir / "debug_classified_regions"
    debug_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, det in enumerate(classified_regions, start=1):
        label = str(det.get("label", ""))
        if label not in {"erased_content", "overwritten_text"}:
            continue

        clipped = _clip_bbox_to_image(tuple(det.get("bbox", (0, 0, 0, 0))), image.shape[:2])
        if clipped is None:
            continue

        x, y, w, h = clipped
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        score = float(det.get("score", 0.0))
        out_path = debug_dir / f"{idx:03d}_{label}_{score:.3f}.png"
        if cv2.imwrite(str(out_path), crop):
            saved += 1

    return saved


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline config object from CLI args."""
    return PipelineConfig(
        target_width=args.target_width,
        feature=FeatureConfig(patch_size=args.patch_size, stride=args.stride),
        anomaly_map=AnomalyMapConfig(
            threshold_strategy=args.threshold_strategy,
            threshold_value=args.threshold_value,
            percentile=args.percentile,
            smoothing_ksize=args.smoothing_ksize,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            min_region_area=args.min_region_area,
            morph_kernel_size=args.morph_kernel_size,
        ),
        box=BoxConfig(
            min_contour_area=args.min_contour_area,
            min_box_area=args.min_box_area,
            min_width=args.min_width,
            min_height=args.min_height,
            merge_iou_threshold=args.merge_iou_threshold,
            merge_max_gap=args.merge_max_gap,
            nms_iou_threshold=args.nms_iou_threshold,
        ),
        copy_move=CopyMoveConfig(
            orb_nfeatures=args.orb_nfeatures,
            ratio_test_threshold=args.ratio_test_threshold,
            max_descriptor_distance=args.max_descriptor_distance,
            min_spatial_distance=args.min_spatial_distance,
            max_spatial_distance=args.max_spatial_distance,
            min_match_count_for_clustering=args.min_match_count,
        ),
        spacing=SpacingDetectorConfig(
            ocr_min_confidence=args.ocr_min_confidence,
            spacing_z_threshold=args.spacing_z_threshold,
            spacing_min_gap=args.spacing_min_gap,
            height_z_threshold=args.height_z_threshold,
            min_words_per_line=args.min_words_per_line,
        ),
        region_classifier=RegionClassifierConfig(
            crop_expand_px=args.classifier_crop_expand,
            min_region_area=args.classifier_min_region_area,
            min_width=args.classifier_min_width,
            min_height=args.classifier_min_height,
            erasure_score_threshold=args.erasure_score_threshold,
            overwrite_score_threshold=args.overwrite_score_threshold,
            decision_margin=args.classifier_decision_margin,
        ),
        document_ai_classifier=DocumentClassifierConfig(
            ai_generated_score_threshold=args.doc_ai_gen_threshold,
            enable_shallow_model=args.doc_ai_enable_model,
            model_path=args.doc_ai_model_path,
            use_margin_uniformity=True,
            use_fft_analysis=True,
            use_noise_consistency=True,
        ),
        region_ai_edit_classifier=RegionAIEditConfig(
            ai_edit_region_threshold=args.region_ai_edit_threshold,
            min_region_area=args.region_ai_edit_min_area,
            context_margin=args.region_ai_edit_context_margin,
            use_context_comparison=args.region_ai_edit_use_context,
        ),
        added_content=AddedContentConfig(
            model_path=args.yolo_model,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.yolo_iou,
            device=args.device,
        ),
        confidence=ConfidenceConfig(
            agreement_boost_factor=args.conf_agreement_boost,
            weak_isolated_penalty=args.conf_weak_penalty,
            min_score_threshold=args.conf_min_threshold,
        ),
        label_refiner=LabelRefinerConfig(
            iou_threshold=args.refiner_iou,
            distance_threshold=args.refiner_distance,
            merge_boxes=args.refiner_merge_boxes,
            suppress_generic_suspicious=True,
        ),
        fusion=FusionConfig(
            merge_iou_threshold=args.fusion_iou,
            dedupe_iou_threshold=args.fusion_dedupe_iou,
            attach_refined_iou_threshold=args.attach_refined_iou,
        ),
        combine_day1_contours=not args.disable_day1_merge,
        day1_min_area=args.day1_min_area,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Day 8 document forgery detection pipeline with final label refinement.")

    parser.add_argument("--input", required=True, type=str, help="Path to input image or PDF.")
    parser.add_argument("--yolo-model", type=str, default="", help="Path to YOLO stamp/signature model.")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--confidence-threshold", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--device", type=str, default=None, help="YOLO inference device, e.g. cpu or 0.")

    parser.add_argument("--target-width", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold-strategy", choices=["adaptive", "fixed", "otsu", "percentile"], default="percentile")
    parser.add_argument("--threshold-value", type=float, default=0.60)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--smoothing-ksize", type=int, default=7)
    parser.add_argument("--adaptive-block-size", type=int, default=35)
    parser.add_argument("--adaptive-c", type=float, default=-2.0)
    parser.add_argument("--min-region-area", type=int, default=200)
    parser.add_argument("--morph-kernel-size", type=int, default=5)

    parser.add_argument("--min-contour-area", type=int, default=120)
    parser.add_argument("--min-box-area", type=int, default=250)
    parser.add_argument("--min-width", type=int, default=10)
    parser.add_argument("--min-height", type=int, default=10)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.20)
    parser.add_argument("--merge-max-gap", type=int, default=12)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--disable-day1-merge", action="store_true")
    parser.add_argument("--day1-min-area", type=int, default=500)

    parser.add_argument("--orb-nfeatures", type=int, default=2500)
    parser.add_argument("--ratio-test-threshold", type=float, default=0.78)
    parser.add_argument("--max-descriptor-distance", type=int, default=64)
    parser.add_argument("--min-spatial-distance", type=float, default=20.0)
    parser.add_argument("--max-spatial-distance", type=float, default=700.0)
    parser.add_argument("--min-match-count", type=int, default=8)

    parser.add_argument("--ocr-min-confidence", type=float, default=45.0)
    parser.add_argument("--spacing-z-threshold", type=float, default=2.2)
    parser.add_argument("--spacing-min-gap", type=float, default=7.0)
    parser.add_argument("--height-z-threshold", type=float, default=1.8)
    parser.add_argument("--min-words-per-line", type=int, default=4)

    parser.add_argument("--fusion-iou", type=float, default=0.30)
    parser.add_argument("--fusion-dedupe-iou", type=float, default=0.85)
    parser.add_argument("--attach-refined-iou", type=float, default=0.20)

    parser.add_argument("--classifier-crop-expand", type=int, default=8)
    parser.add_argument("--classifier-min-region-area", type=int, default=64)
    parser.add_argument("--classifier-min-width", type=int, default=8)
    parser.add_argument("--classifier-min-height", type=int, default=8)
    parser.add_argument("--erasure-score-threshold", type=float, default=0.52)
    parser.add_argument("--overwrite-score-threshold", type=float, default=0.52)
    parser.add_argument("--classifier-decision-margin", type=float, default=0.05)
    parser.add_argument("--save-debug-crops", action="store_true", help="Save debug crops for erased/overwritten classifications.")

    # Document-level AI generation detection
    parser.add_argument("--doc-ai-gen-threshold", type=float, default=0.70, help="Threshold for full-page AI-generation score.")
    parser.add_argument("--doc-ai-enable-model", action="store_true", help="Enable optional shallow ML classifier for document AI detection.")
    parser.add_argument("--doc-ai-model-path", type=str, default=None, help="Path to shallow ML model for document AI classification.")

    # Region-level AI editing detection
    parser.add_argument("--region-ai-edit-threshold", type=float, default=0.65, help="Threshold for region-level AI-edit score.")
    parser.add_argument("--region-ai-edit-min-area", type=int, default=100, help="Minimum region area for AI-edit analysis.")
    parser.add_argument("--region-ai-edit-context-margin", type=int, default=20, help="Context margin for comparing regions to surroundings.")
    parser.add_argument("--region-ai-edit-use-context", action="store_true", default=True, help="Use context comparison for regional AI detection.")

    # Day 8: Confidence score calibration
    parser.add_argument("--conf-agreement-boost", type=float, default=1.15, help="Confidence boost factor when multiple modules agree.")
    parser.add_argument("--conf-weak-penalty", type=float, default=0.75, help="Penalty factor for weak isolated detections.")
    parser.add_argument("--conf-min-threshold", type=float, default=0.50, help="Minimum confidence threshold for score processing.")

    # Day 8: Label refinement
    parser.add_argument("--refiner-iou", type=float, default=0.15, help="IoU threshold for grouping detections.")
    parser.add_argument("--refiner-distance", type=int, default=30, help="Center distance threshold for grouping (pixels).")
    parser.add_argument("--refiner-merge-boxes", action="store_true", default=True, help="Merge boxes in grouped detections.")

    # Output options
    parser.add_argument("--save-json-report", action="store_true", help="Save detailed JSON report of all detections.")
    parser.add_argument("--save-intermediate-vis", action="store_true", help="Save intermediate detections visualization for debugging.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--show", action="store_true", help="Show visual previews with matplotlib.")

    return parser.parse_args()


def main() -> None:
    """Execute Day 8 full detection flow with final refinement and save artifacts."""
    args = parse_args()
    _configure_logging(args.log_level)

    try:
        config = _build_pipeline_config(args)
        result = detect_document_tampering(args.input, config=config)

        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = result["source_path"]
        base = source_path.stem

        # Save diagnostic artifacts
        LOGGER.info("Saving diagnostic artifacts...")
        anomaly_heatmap_path = _save_anomaly_heatmap(result["anomaly_map"], output_dir / f"{base}_anomaly_heatmap.png")
        binary_mask_path = _save_binary_mask(result["binary_mask"], output_dir / f"{base}_binary_mask.png")
        LOGGER.info("Saved anomaly heatmap: %s", anomaly_heatmap_path)
        LOGGER.info("Saved binary mask: %s", binary_mask_path)

        # Save visualizations
        LOGGER.info("Saving final detection visualizations...")
        processed_image = result["processed_image"]
        intermediate_detections = result.get("intermediate_detections", [])
        final_detections = result.get("final_detections", [])

        vis_paths = save_pipeline_visualizations(
            image=processed_image,
            intermediate_detections=intermediate_detections,
            final_detections=final_detections,
            output_dir=output_dir,
            base_filename=base,
        )

        if "intermediate" in vis_paths:
            LOGGER.info("Saved intermediate detections visualization: %s", vis_paths["intermediate"])
        LOGGER.info("Saved final detections visualization: %s", vis_paths["final"])

        # Save optional intermediate visualization separately if requested
        if args.save_intermediate_vis and intermediate_detections:
            inter_path = output_dir / f"{base}_intermediate_only.png"
            from visualization import draw_intermediate_detections
            inter_vis = draw_intermediate_detections(processed_image, intermediate_detections)
            cv2.imwrite(str(inter_path), inter_vis)
            LOGGER.info("Saved intermediate-only visualization: %s", inter_path)

        # Save debug crops if requested
        if args.save_debug_crops:
            saved_crops = _save_debug_classified_crops(
                image=processed_image,
                classified_regions=result.get("classified_regions", []),
                output_dir=output_dir,
            )
            LOGGER.info("Saved debug classified crops: %d", saved_crops)

        # Save JSON report if requested
        if args.save_json_report:
            doc_label_info = result.get("document_label", {})
            json_report = {
                "source_file": str(source_path),
                "document_label": {
                    "label": doc_label_info.get("label", "unknown"),
                    "score": float(doc_label_info.get("score", 0.0)),
                },
                "intermediate_detection_count": len(intermediate_detections),
                "final_detection_count": len(final_detections),
                "final_detections": [
                    {
                        "bbox": det["bbox"],
                        "label": det["label"],
                        "score": float(det["score"]),
                        "source": det["source"],
                        "supporting_sources": det.get("supporting_sources", []),
                    }
                    for det in final_detections
                ],
            }
            json_path = output_dir / f"{base}_final_report.json"
            with open(json_path, "w") as f:
                json.dump(json_report, f, indent=2)
            LOGGER.info("Saved final JSON report: %s", json_path)

        # Print final summary
        print("\n" + "=" * 70)
        print("DAY 8: FINAL LABEL REFINEMENT DETECTION SUMMARY")
        print("=" * 70)

        doc_label_info = result.get("document_label", {})
        print(f"\nDocument-Level Analysis:")
        print(f"  Label: {doc_label_info.get('label', 'unknown')}")
        print(f"  Score: {doc_label_info.get('score', 0.0):.3f}")

        print(f"\nDetection Statistics:")
        print(f"  Intermediate detections (before refinement): {len(intermediate_detections)}")
        print(f"  Final detections (after refinement):        {len(final_detections)}")

        if final_detections:
            label_counts = Counter(det["label"] for det in final_detections)
            print(f"\nFinal Detection Breakdown:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")

            print(f"\nDetailed Final Detections:")
            print("-" * 70)
            for idx, det in enumerate(final_detections, start=1):
                x, y, w, h = det["bbox"]
                score = float(det.get("score", 0.0))
                label = det.get("label", "unknown")
                sources = det.get("supporting_sources", [])
                sources_str = " + ".join(sources) if sources else "unknown"

                print(f"{idx}. [{label}] score={score:.3f} | bbox=({x},{y},{w},{h}) | src=[{sources_str}]")
        else:
            print(f"\nNo detections found after refinement.")

        print("=" * 70 + "\n")

        # Show visualizations if requested
        if args.show:
            LOGGER.info("Displaying visualizations...")
            final_vis = draw_final_detections(processed_image, final_detections, show_supporting_sources=True)

            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(result["anomaly_map"], cmap="jet")
            plt.title("Anomaly Heatmap")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            if intermediate_detections:
                from visualization import draw_intermediate_detections
                inter_vis = draw_intermediate_detections(processed_image, intermediate_detections)
                plt.imshow(cv2.cvtColor(inter_vis, cv2.COLOR_BGR2RGB))
                plt.title(f"Intermediate ({len(intermediate_detections)} detections)")
            else:
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                plt.title("Intermediate (none)")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Final Refined ({len(final_detections)} detections)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    except Exception:
        LOGGER.exception("Day 8 pipeline execution failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
