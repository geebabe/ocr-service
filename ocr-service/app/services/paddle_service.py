import base64
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import List, Tuple
import unicodedata
from app.core.logger import logger

# Initialize globally
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="vi")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {e}")
    ocr = None

class OCRToken(BaseModel):
    text: str
    bbox: Tuple[int, int, int, int] # x1, y1, x2, y2
    confidence: float

def run_paddle_ocr(image_base64: str) -> List[OCRToken]:
    """
    Runs PaddleOCR on a base64 encoded image and returns a list of OCRToken.
    """
    if not ocr:
        logger.error("PaddleOCR not initialized.")
        return []
    
    try:
        # Decode base64 to numpy array
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # PaddleOCR uses BGR by default if using cv2, but numpy array from PIL is RGB
        # It handles RGB/BGR reasonably well, or we can convert to BGR:
        image_np = np.array(image)
        image_np = image_np[:, :, ::-1] # RGB to BGR
        
        # Run OCR
        # result format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('text', confidence)]
        results = ocr.ocr(image_np)
    except Exception as e:
        logger.error(f"Error during PaddleOCR inference: {e}", exc_info=True)
        return []
        
    tokens = []
    if not results or not results[0]:
        return tokens
        
    for res in results[0]:
        if not res:
            continue
        box = res[0]
        text_data = res[1]
        text = text_data[0]
        confidence = float(text_data[1])
        
        # Filter low confidence
        if confidence < 0.5:
            continue
            
        # Get bbox x1, y1, x2, y2
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        
        # Normalize text to NFC
        normalized_text = unicodedata.normalize("NFC", text)
        
        tokens.append(OCRToken(text=normalized_text, bbox=bbox, confidence=confidence))
        
    return tokens
