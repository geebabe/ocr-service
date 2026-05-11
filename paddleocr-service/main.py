import logging
import os
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2

# Reduce logging noise
os.environ["PADDLEOCR_LOG_LEVEL"] = "WARNING"
logging.getLogger("ppocr").setLevel(logging.WARNING)

from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR Service")

# Initialize OCR engine globally
ocr = PaddleOCR(
    lang="vi",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
    text_rec_score_thresh=0.5,
)

class OcrRequest(BaseModel):
    image_base64: str

class OcrResult(BaseModel):
    text: str
    score: float
    bbox: list[int]  # [xmin, ymin, xmax, ymax]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/extract", response_model=list[OcrResult])
def extract_text(req: OcrRequest):
    try:
        # Decode base64
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(image)
        # Convert RGB to BGR for OpenCV (PaddleOCR prefers cv2 format internally)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Predict
        results = ocr.predict(img_bgr)
        
        if not results or not results[0]:
            return []
            
        page = results[0]
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        boxes = page.get("rec_boxes", [])
        
        out = []
        for i, (t, s) in enumerate(zip(texts, scores)):
            if i < len(boxes):
                b = boxes[i].tolist()
                bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
            else:
                bbox = [0, 0, 0, 0]
            
            out.append(OcrResult(text=t, score=float(s), bbox=bbox))
            
        return out
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

