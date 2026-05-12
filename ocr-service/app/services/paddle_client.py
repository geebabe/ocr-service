import logging
import os
import base64
import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from app.core.config import settings
from app.core.logger import logger
from typing import List, Dict, Any

# Suppress PaddleOCR logs
os.environ["PADDLEOCR_LOG_LEVEL"] = "WARNING"
logging.getLogger("ppocr").setLevel(logging.WARNING)

# Initialize PaddleOCR engine
# Note: In a production environment, you might want to initialize this 
# inside the app startup event if you want to control when it loads.
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info(f"Initializing PaddleOCR engine on {settings.PADDLE_DEVICE}...")
        _ocr_engine = PaddleOCR(
            lang="vi",
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=settings.PADDLE_DEVICE,
            text_rec_score_thresh=0.5,
            show_log=False
        )
        logger.info("✅ OCR engine ready")
    return _ocr_engine

async def call_paddleocr(image_base64: str, img_w: int, img_h: int) -> str:
    """
    Runs PaddleOCR locally, normalizes the bounding boxes to [0, 1000],
    and formats the result as a list of tuples to inject into the LLM prompt.
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)

        # Get OCR engine
        ocr = get_ocr_engine()
        
        # Standard PaddleOCR.ocr() returns: [ [ [box], (text, score) ], ... ]
        # We'll adapt this to the structure the user wants
        results = ocr.predict(img_np)
        
        if not results or not results[0]:
            return "Không tìm thấy văn bản nào."

        # Adapt to user's desired "rows" structure
        rows = []
        for i, line in enumerate(results[0]):
            box = line[0] # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text, score = line[1]
            
            # Extract min/max and normalize to [0, 1000]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            
            try:
                xmin = int((min(xs) / img_w) * 1000)
                ymin = int((min(ys) / img_h) * 1000)
                xmax = int((max(xs) / img_w) * 1000)
                ymax = int((max(ys) / img_h) * 1000)
                
                # Ensure within bounds
                xmin, ymin = max(0, min(1000, xmin)), max(0, min(1000, ymin))
                xmax, ymax = max(0, min(1000, xmax)), max(0, min(1000, ymax))
            except ZeroDivisionError:
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
                
            rows.append({
                "#": i,
                "text": text,
                "score": round(float(score), 3),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

        # Format rows into a list of tuples for the prompt
        # Example: ('VĂN BẢN', [10, 20, 100, 50])
        prompt_rows = [
            str((r["text"], [r["xmin"], r["ymin"], r["xmax"], r["ymax"]]))
            for r in rows
        ]

        return "\n".join(prompt_rows)
            
    except Exception as e:
        logger.error(f"PaddleOCR SDK execution failed: {str(e)}")
        return "Lỗi khi chạy OCR sơ bộ, hãy tự phân tích từ ảnh."
