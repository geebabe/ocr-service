import json
import re
from typing import Any, List
from app.schemas.response import InvoiceExtraction
from app.core.logger import logger

def normalize_bbox(bbox_1000: List[int], img_w: int, img_h: int) -> List[int]:
    """Scale Qwen-VL's 1000x1000 normalized bbox to original image pixel coordinates."""
    try:
        x1, y1, x2, y2 = bbox_1000
        return [
            int(x1 * img_w / 1000),
            int(y1 * img_h / 1000),
            int(x2 * img_w / 1000),
            int(y2 * img_h / 1000),
        ]
    except Exception:
        return bbox_1000

def _normalize_bboxes(node: Any, img_w: int, img_h: int) -> Any:
    """
    Recursively walk the parsed JSON dict and normalize bounding_box lists
    from Qwen's 0-1000 coordinate scale to actual pixel coordinates.
    """
    if isinstance(node, dict):
        # Create a new dict to avoid modifying in-place while iterating
        new_node = {}
        for k, v in node.items():
            if k == "bounding_box" and isinstance(v, list) and len(v) == 4:
                # Only normalize if it looks like normalized coordinates (all <= 1000)
                if all(isinstance(c, (int, float)) and c <= 1000 for c in v):
                    new_node[k] = normalize_bbox(v, img_w, img_h)
                else:
                    new_node[k] = v
            else:
                new_node[k] = _normalize_bboxes(v, img_w, img_h)
        return new_node
    elif isinstance(node, list):
        return [_normalize_bboxes(item, img_w, img_h) for item in node]
    return node

def _wrap_raw_values(node: Any) -> Any:
    """
    Recursively wrap raw scalar values into the BBoxField format.
    """
    if isinstance(node, dict):
        # If the dict looks like a BBoxField (has 'value' key), leave it as is
        if "value" in node:
            return node
        
        return {k: _wrap_raw_values(v) for k, v in node.items()}
    elif isinstance(node, list):
        return [_wrap_raw_values(item) for item in node]
    elif node is None:
        return None
    else:
        # It's a raw scalar (str, int, float, bool)
        return {"value": node, "bounding_box": [0, 0, 0, 0]}


def parse_vllm_output(output_text: str, img_w: int, img_h: int) -> InvoiceExtraction:
    """
    Parses the JSON output from vLLM (guided decoding).
    Since guided_json is used, the output is always valid JSON matching InvoiceExtraction schema.
    Only bbox normalization from Qwen's 0-1000 coordinate space to pixel coords is applied.
    """
    try:
        # Use regex to robustly extract the JSON object, ignoring any markdown or trailing tokens
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            clean_text = output_text.strip()

        parsed_dict = json.loads(clean_text)

        # Wrap scalar values to match BBoxField schema
        wrapped_dict = _wrap_raw_values(parsed_dict)

        # Normalize any bounding boxes that are in Qwen's 0-1000 scale
        normalized_dict = _normalize_bboxes(wrapped_dict, img_w, img_h)

        return InvoiceExtraction(**normalized_dict)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model output: {str(e)}\nRaw output: {output_text}")
        raise
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}\nWrapped dict: {wrapped_dict if 'wrapped_dict' in locals() else 'N/A'}", exc_info=True)
        raise
