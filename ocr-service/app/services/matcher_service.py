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

def merge_bboxes(tokens: List[OCRToken]) -> Optional[Tuple[int, int, int, int]]:
    if not tokens:
        return None
    x1 = min(t.bbox[0] for t in tokens)
    y1 = min(t.bbox[1] for t in tokens)
    x2 = max(t.bbox[2] for t in tokens)
    y2 = max(t.bbox[3] for t in tokens)
    return (x1, y1, x2, y2)

def match_value(vlm_value: str, ocr_tokens: List[OCRToken]) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
    """
    Tries to match a string value against OCR tokens using 3 tiers.
    Returns (bbox, confidence, match_type)
    """
    if not vlm_value or not isinstance(vlm_value, str) or vlm_value == "...":
        return None, 0.0, "no_match"
        
    norm_vlm = normalize_text(vlm_value)
    if not norm_vlm:
        return None, 0.0, "no_match"

    # Tier 1: Exact Match
    for token in ocr_tokens:
        if normalize_text(token.text) == norm_vlm:
            return token.bbox, 1.0, "exact"
            
    # Tier 2: Fuzzy Match
    best_fuzzy_score = 0
    best_fuzzy_token = None
    
    for token in ocr_tokens:
        score = fuzz.ratio(norm_vlm, normalize_text(token.text)) / 100.0
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_token = token
            
    if best_fuzzy_score >= 0.85:
        return best_fuzzy_token.bbox, best_fuzzy_score, "fuzzy"
        
    # Tier 2 Fallback: Strip Accent Fuzzy
    vlm_stripped = strip_accent(norm_vlm)
    best_stripped_score = 0
    best_stripped_token = None
    
    for token in ocr_tokens:
        score = fuzz.ratio(vlm_stripped, strip_accent(normalize_text(token.text))) / 100.0
        if score > best_stripped_score:
            best_stripped_score = score
            best_stripped_token = token
            
    if best_stripped_score >= 0.85:
        return best_stripped_token.bbox, best_stripped_score * 0.95, "fuzzy_no_accent"
        
    # Tier 3: Span Match
    best_span_score = 0
    best_span_tokens = []
    
    MAX_WINDOW = 5
    for i in range(len(ocr_tokens)):
        for w in range(1, MAX_WINDOW + 1):
            if i + w > len(ocr_tokens):
                break
                
            span_tokens = ocr_tokens[i:i+w]
            
            # Constraint: Must be on same line roughly
            ys = [t.bbox[1] for t in span_tokens]
            if max(ys) - min(ys) > 20: # heuristic 20px
                continue
                
            span_text = " ".join([t.text for t in span_tokens])
            score = fuzz.ratio(norm_vlm, normalize_text(span_text)) / 100.0
            
            if score > best_span_score:
                best_span_score = score
                best_span_tokens = span_tokens
                
    if best_span_score >= 0.80:
        bbox = merge_bboxes(best_span_tokens)
        return bbox, best_span_score * 0.90, "span"
        
    return None, 0.0, "no_match"

def process_dict_recursively(node: Any, ocr_tokens: List[OCRToken]) -> Any:
    if isinstance(node, dict):
        result = {}
        for k, v in node.items():
            if isinstance(v, str):
                if v == "..." or not v.strip():
                    result[k] = {"value": v, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
                    continue
                bbox, conf, mtype = match_value(v, ocr_tokens)
                result[k] = {"value": v, "bounding_box": bbox, "confidence": conf, "match_type": mtype}
            elif isinstance(v, list):
                result[k] = process_dict_recursively(v, ocr_tokens)
            elif isinstance(v, dict):
                result[k] = process_dict_recursively(v, ocr_tokens)
            else:
                result[k] = {"value": v, "bounding_box": None, "confidence": 0.0, "match_type": "no_match"}
        return result
    elif isinstance(node, list):
        return [process_dict_recursively(item, ocr_tokens) for item in node]
    return node

def merge_ocr_results(parsed_vllm_json: dict, ocr_tokens: List[OCRToken]) -> InvoiceExtraction:
    """
    Merges VLLM output with PaddleOCR tokens to add bounding boxes,
    then returns the validated InvoiceExtraction object.
    """
    processed = process_dict_recursively(parsed_vllm_json, ocr_tokens)
    return InvoiceExtraction(**processed)
