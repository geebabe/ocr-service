import re
from typing import List, Dict, Any, Tuple, Optional
from thefuzz import fuzz
import unicodedata
import unidecode
from app.services.paddle_service import OCRToken
from app.schemas.response import InvoiceExtraction

def normalize_text(text: str) -> str:
    """Normalize text for Exact Match."""
    if not isinstance(text, str):
        text = str(text)
    # NFC
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    return " ".join(text.split())

def strip_accent(text: str) -> str:
    """Remove Vietnamese accents."""
    return unidecode.unidecode(text)

def normalize_number(text: str) -> str:
    """Keep only digits for numbers like 1.500.000đ or 1,500,000"""
    return re.sub(r'\D', '', text)

def normalize_tax_code(text: str) -> str:
    """Keep only digits, ignoring letters like 'O' mistaken for '0'."""
    text = text.replace('o', '0').replace('O', '0')
    return re.sub(r'\D', '', text)

def merge_bboxes(tokens: List[OCRToken]) -> Optional[Tuple[int, int, int, int]]:
    if not tokens:
        return None
    x1 = min(t.bbox[0] for t in tokens)
    y1 = min(t.bbox[1] for t in tokens)
    x2 = max(t.bbox[2] for t in tokens)
    y2 = max(t.bbox[3] for t in tokens)
    return (x1, y1, x2, y2)

def is_in_y_band(bbox: Tuple[int, int, int, int], y_band: Tuple[int, int]) -> bool:
    """Check if a bbox vertically overlaps with the given y_band."""
    y_min, y_max = y_band
    # Overlap condition: not (bbox is completely above or completely below)
    return not (bbox[3] < y_min or bbox[1] > y_max)

def match_value(
    vlm_value: str, 
    ocr_tokens: List[OCRToken], 
    field_name: str = "",
    y_band: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
    """
    Tries to match a string value against OCR tokens using 3 tiers.
    Returns (bbox, confidence, match_type)
    """
    if not vlm_value or not isinstance(vlm_value, str) or vlm_value == "...":
        return None, 0.0, "no_match"
        
    # Filter tokens by y_band if provided
    if y_band is not None:
        filtered_tokens = [t for t in ocr_tokens if is_in_y_band(t.bbox, y_band)]
    else:
        filtered_tokens = ocr_tokens

    # Apply field-specific normalization
    is_numeric_field = field_name in ["subtotal", "tax", "total_amount", "quantity", "unit_price"]
    is_tax_code = field_name == "tax_code"
    
    if is_numeric_field:
        norm_vlm = normalize_number(vlm_value)
    elif is_tax_code:
        norm_vlm = normalize_tax_code(vlm_value)
    else:
        norm_vlm = normalize_text(vlm_value)
        
    if not norm_vlm:
        return None, 0.0, "no_match"

    # Tier 1: Exact Match
    for token in filtered_tokens:
        if is_numeric_field:
            norm_token = normalize_number(token.text)
        elif is_tax_code:
            norm_token = normalize_tax_code(token.text)
        else:
            norm_token = normalize_text(token.text)
            
        if norm_token == norm_vlm:
            return token.bbox, 1.0, "exact"
            
    # Tier 2: Fuzzy Match
    best_fuzzy_score = 0
    best_fuzzy_token = None
    
    for token in filtered_tokens:
        if is_numeric_field:
            norm_token = normalize_number(token.text)
        elif is_tax_code:
            norm_token = normalize_tax_code(token.text)
        else:
            norm_token = normalize_text(token.text)
            
        if not norm_token:
            continue
            
        score = fuzz.ratio(norm_vlm, norm_token) / 100.0
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_token = token
            
    if best_fuzzy_score >= 0.85:
        return best_fuzzy_token.bbox, best_fuzzy_score, "fuzzy"
        
    # Tier 2 Fallback: Strip Accent Fuzzy
    if not is_numeric_field and not is_tax_code:
        vlm_stripped = strip_accent(norm_vlm)
        best_stripped_score = 0
        best_stripped_token = None
        
        for token in filtered_tokens:
            norm_token = normalize_text(token.text)
            score = fuzz.ratio(vlm_stripped, strip_accent(norm_token)) / 100.0
            if score > best_stripped_score:
                best_stripped_score = score
                best_stripped_token = token
                
        if best_stripped_score >= 0.85:
            return best_stripped_token.bbox, best_stripped_score * 0.95, "fuzzy_no_accent"
        
    # Tier 3: Span Match
    best_span_score = 0
    best_span_tokens = []
    
    MAX_WINDOW = 5
    for i in range(len(filtered_tokens)):
        for w in range(1, MAX_WINDOW + 1):
            if i + w > len(filtered_tokens):
                break
                
            span_tokens = filtered_tokens[i:i+w]
            
            # Constraint: Must be on same line roughly
            ys = [t.bbox[1] for t in span_tokens]
            if max(ys) - min(ys) > 20: # heuristic 20px
                continue
                
            span_text = " ".join([t.text for t in span_tokens])
            
            if is_numeric_field:
                norm_span = normalize_number(span_text)
            elif is_tax_code:
                norm_span = normalize_tax_code(span_text)
            else:
                norm_span = normalize_text(span_text)
                
            if not norm_span:
                continue
                
            score = fuzz.ratio(norm_vlm, norm_span) / 100.0
            
            if score > best_span_score:
                best_span_score = score
                best_span_tokens = span_tokens
                
    if best_span_score >= 0.80:
        bbox = merge_bboxes(best_span_tokens)
        return bbox, best_span_score * 0.90, "span"
        
    return None, 0.0, "no_match"

def process_dict_recursively(node: Any, ocr_tokens: List[OCRToken], current_key: str = "") -> Any:
    if isinstance(node, dict):
        result = {}
        for k, v in node.items():
            if isinstance(v, str):
                if v == "..." or not v.strip():
                    if k == "items":
                        result[k] = []
                        continue
                    result[k] = {"value": v, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
                    continue
                bbox, conf, mtype = match_value(v, ocr_tokens, field_name=k)
                result[k] = {"value": v, "bounding_box": bbox, "confidence": conf, "match_type": mtype}
            elif isinstance(v, list):
                if k == "items":
                    # Special logic for items
                    parsed_items = []
                    for item in v:
                        if isinstance(item, dict):
                            item_result = {}
                            y_band = None
                            
                            # Match description first to anchor the row
                            desc_val = item.get("description")
                            if isinstance(desc_val, str) and desc_val and desc_val != "...":
                                bbox, conf, mtype = match_value(desc_val, ocr_tokens, field_name="description")
                                item_result["description"] = {"value": desc_val, "bounding_box": bbox, "confidence": conf, "match_type": mtype}
                                if bbox:
                                    y_band = (bbox[1] - 15, bbox[3] + 15) # +/- 15px heuristic
                            elif "description" in item:
                                item_result["description"] = {"value": desc_val, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
                                
                            # Match the rest of the fields using the y_band
                            for ik, iv in item.items():
                                if ik == "description":
                                    continue
                                if isinstance(iv, str):
                                    if iv == "..." or not iv.strip():
                                        item_result[ik] = {"value": iv, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
                                    else:
                                        ibbox, iconf, imtype = match_value(iv, ocr_tokens, field_name=ik, y_band=y_band)
                                        item_result[ik] = {"value": iv, "bounding_box": ibbox, "confidence": iconf, "match_type": imtype}
                                else:
                                    item_result[ik] = {"value": iv, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
                            parsed_items.append(item_result)
                        else:
                            parsed_items.append(process_dict_recursively(item, ocr_tokens, k))
                    result[k] = parsed_items
                else:
                    result[k] = [process_dict_recursively(item, ocr_tokens, k) for item in v]
            elif isinstance(v, dict):
                result[k] = process_dict_recursively(v, ocr_tokens, k)
            else:
                result[k] = {"value": v, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
        return result
    elif isinstance(node, list):
        return [process_dict_recursively(item, ocr_tokens, current_key) for item in node]
    return node

def merge_ocr_results(parsed_vllm_json: dict, ocr_tokens: List[OCRToken]) -> InvoiceExtraction:
    """
    Merges VLLM output with PaddleOCR tokens to add bounding boxes,
    then returns the validated InvoiceExtraction object.
    """
    processed = process_dict_recursively(parsed_vllm_json, ocr_tokens)
    return InvoiceExtraction(**processed)
