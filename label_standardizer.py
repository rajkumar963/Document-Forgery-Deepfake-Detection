"""Label standardization to hackathon problem statement format.

Maps internal detector labels to canonical hackathon categories:
- stamp, signature, seal, added_mark → added_content
- erased_content, overwritten_text → unchanged
- copy_paste → copy_paste  
- irregular_spacing → irregular_spacing
- ai_edited_region → ai_edited_region
- suspicious_region → suspicious_region (fallback)
- fully_ai_generated_document → fully_ai_generated_document (doc-level)
- authentic_or_unknown → authentic_or_unknown (doc-level)

NOT IN CURRENT IMPLEMENTATION (to be added):
- watermark_removal
- merged_document_content
"""

from __future__ import annotations

from typing import Any, Dict, List


# Mapping from internal labels to canonical hackathon labels
LABEL_CANONICAL_MAP = {
    # Added content - YOLO classes consolidated
    "stamp": "added_content",
    "signature": "added_content",
    "seal": "added_content",
    "added_mark": "added_content",
    
    # Direct mappings - already correct
    "erased_content": "erased_content",
    "overwritten_text": "overwritten_text",
    "copy_paste": "copy_paste",
    "irregular_spacing": "irregular_spacing",
    "ai_edited_region": "ai_edited_region",
    
    # Fallback - generic
    "suspicious_region": "suspicious_region",
    
    # Document-level
    "fully_ai_generated_document": "fully_ai_generated_document",
    "authentic_or_unknown": "authentic_or_unknown",
}

# Canonical labels for external output
CANONICAL_LABELS = {
    "copy_paste",
    "overwritten_text", 
    "erased_content",
    "added_content",
    "merged_document_content",
    "watermark_removal",
    "irregular_spacing",
    "ai_edited_region",
    "fully_ai_generated_document",
    "authentic_or_unknown",
    "suspicious_region",  # fallback
}


def standardize_label(internal_label: str) -> str:
    """Convert internal detector label to canonical hackathon label.
    
    Args:
        internal_label: Label from detector (e.g., "stamp", "signature", "erased_content")
        
    Returns:
        Canonical label for problem statement (e.g., "added_content", "erased_content")
        Falls back to "suspicious_region" if label not recognized
    """
    canonical = LABEL_CANONICAL_MAP.get(internal_label, "suspicious_region")
    return canonical


def standardize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize a single detection dict to canonical format.
    
    Preserves all fields but standardizes the 'label' field.
    
    Args:
        detection: Detection dict with optional "label" field
        
    Returns:
        Detection dict with standardized "label"
    """
    detection_copy = dict(detection)
    
    if "label" in detection_copy:
        old_label = detection_copy["label"]
        detection_copy["label"] = standardize_label(old_label)
        
        # Preserve original label if different for debugging
        if old_label != detection_copy["label"]:
            detection_copy["_original_label"] = old_label
    
    return detection_copy


def standardize_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize a list of detections to canonical format.
    
    Args:
        detections: List of detection dicts
        
    Returns:
        List of detections with standardized labels
    """
    return [standardize_detection(det) for det in detections]


def standardize_document_label(document_label: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize document-level label to canonical format.
    
    Args:
        document_label: Document-level label dict with "label" and "score"
        
    Returns:
        Standardized document label dict
    """
    result = dict(document_label)
    
    if "label" in result:
        result["label"] = standardize_label(result["label"])
    
    return result


def standardize_pipeline_output(pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize full pipeline output to canonical format.
    
    Applies label standardization to:
    - document_label
    - final_detections
    - intermediate_detections (for debugging)
    
    Args:
        pipeline_result: Full pipeline output dict
        
    Returns:
        Pipeline output with standardized labels
    """
    result = dict(pipeline_result)
    
    # Standardize document-level label
    if "document_label" in result and result["document_label"] is not None:
        result["document_label"] = standardize_document_label(result["document_label"])
    
    # Standardize final detections
    if "final_detections" in result and result["final_detections"] is not None:
        result["final_detections"] = standardize_detections(result["final_detections"])
    
    # Standardize intermediate detections (for debugging)
    if "intermediate_detections" in result and result["intermediate_detections"] is not None:
        result["intermediate_detections"] = standardize_detections(result["intermediate_detections"])
    
    return result
