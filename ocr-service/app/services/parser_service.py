import json
import re
import ast
from typing import Dict, Any, List, Optional, Tuple
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

def _extract_bbox_from_string(text: str) -> Optional[Tuple[str, List[int]]]:
    """Extracts value and bbox from 'value <|box_start|>(x1,y1),(x2,y2)<|box_end|>'."""
    if not isinstance(text, str):
        return None
    pattern = r'(.*?)\s*<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    m = re.search(pattern, text)
    if m:
        value = m.group(1).strip()
        bbox = [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
        return value, bbox
    return None

def _process_node(node: Any, img_w: int, img_h: int) -> Any:
    """Recursively traverse the dict to normalize bounding boxes and extract tags."""
    if isinstance(node, dict):
        # Base case: if dict has 'value'
        if "value" in node:
            val = node["value"]
            # Check if value string has bbox tokens
            if isinstance(val, str):
                extracted = _extract_bbox_from_string(val)
                if extracted:
                    node["value"] = extracted[0]
                    # Only override bounding_box if it's missing or empty
                    if "bounding_box" not in node or not node["bounding_box"]:
                        node["bounding_box"] = extracted[1]
            
            # Now normalize the bounding_box if it exists
            bbox = node.get("bounding_box")
            if isinstance(bbox, list) and len(bbox) == 4:
                needs_normalize = all(isinstance(c, (int, float)) and c <= 1000 for c in bbox)
                if needs_normalize:
                    node["bounding_box"] = normalize_bbox(bbox, img_w, img_h)
            
            # Continue processing other keys just in case
            for k, v in list(node.items()):
                if k not in ("value", "bounding_box"):
                    node[k] = _process_node(v, img_w, img_h)
            return node
        
        # Check if any string values in dict contain tags, convert them to dicts
        for k, v in list(node.items()):
            if isinstance(v, str):
                extracted = _extract_bbox_from_string(v)
                if extracted:
                    bbox = extracted[1]
                    needs_normalize = all(isinstance(c, (int, float)) and c <= 1000 for c in bbox)
                    if needs_normalize:
                        bbox = normalize_bbox(bbox, img_w, img_h)
                    node[k] = {"value": extracted[0], "bounding_box": bbox}
                else:
                    node[k] = v
            else:
                node[k] = _process_node(v, img_w, img_h)
        return node
        
    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, str):
                extracted = _extract_bbox_from_string(item)
                if extracted:
                    bbox = extracted[1]
                    needs_normalize = all(isinstance(c, (int, float)) and c <= 1000 for c in bbox)
                    if needs_normalize:
                        bbox = normalize_bbox(bbox, img_w, img_h)
                    node[i] = {"value": extracted[0], "bounding_box": bbox}
            else:
                node[i] = _process_node(item, img_w, img_h)
        return node
        
    elif isinstance(node, str):
        extracted = _extract_bbox_from_string(node)
        if extracted:
            bbox = extracted[1]
            needs_normalize = all(isinstance(c, (int, float)) and c <= 1000 for c in bbox)
            if needs_normalize:
                bbox = normalize_bbox(bbox, img_w, img_h)
            return {"value": extracted[0], "bounding_box": bbox}
            
    return node

def normalize_invoice_data(extraction: InvoiceExtraction, img_w: int, img_h: int) -> InvoiceExtraction:
    """
    Normalizes bounding boxes in an existing InvoiceExtraction object.
    """
    # Convert to dict, process, and convert back
    data_dict = extraction.model_dump()
    normalized_dict = _process_node(data_dict, img_w, img_h)
    return InvoiceExtraction(**normalized_dict)

def parse_vllm_output(output_text: str, img_w: int, img_h: int) -> InvoiceExtraction:
    """
    Parses the JSON/Dict output from Qwen3-VL, extracting bounding boxes and values.
    Handles both proper JSON and Python-like dictionary strings (single quotes, None).
    """
    try:
        if isinstance(output_text, InvoiceExtraction):
            # Already parsed (e.g. by instructor), just normalize
            return normalize_invoice_data(output_text, img_w, img_h)
            
        logger.info(f"RAW VLM OUTPUT:\n{output_text}\n---")
        
        # 1. Clean the output: extract anything between the first { and the last }
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON/Dict block found in model output")
            return InvoiceExtraction()
            
        raw_str = json_match.group()
        
        # 2. Attempt to parse
        parsed_dict = None
        
        # Strategy A: Try proper JSON first
        try:
            # Fix invalid JSON if tags are outside quotes: "val" <|box...|> -> "val <|box...|>"
            fixed_json_str = re.sub(
                r'"([^"]*?)"\s*(<\|box_start\|>\(\d+,\d+\),\(\d+,\d+\)<\|box_end\|>)',
                r'"\1 \2"',
                raw_str
            )
            parsed_dict = json.loads(fixed_json_str)
        except json.JSONDecodeError:
            # Strategy B: If JSON fails, it might be a Python literal (single quotes, None, etc.)
            try:
                # ast.literal_eval is safer than eval()
                parsed_dict = ast.literal_eval(raw_str)
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed both JSON and AST parsing: {str(e)}")
                # Last ditch effort: replace common Pythonisms and try JSON again
                try:
                    cleaned_str = raw_str.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
                    parsed_dict = json.loads(cleaned_str)
                except Exception:
                    logger.error("Final attempt at JSON parsing failed.")
                    return InvoiceExtraction()
        
        if not parsed_dict or not isinstance(parsed_dict, dict):
            return InvoiceExtraction()

        # 3. Recursively normalize all bounding boxes and extract string tags
        normalized_dict = _process_node(parsed_dict, img_w, img_h)
        
        # 4. Instantiate Pydantic model
        return InvoiceExtraction(**normalized_dict)
        
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}", exc_info=True)
        return InvoiceExtraction()
