import httpx
from app.core.config import settings
from app.core.logger import logger
from typing import List, Dict, Any

async def call_paddleocr(image_base64: str, img_w: int, img_h: int) -> str:
    """
    Calls the PaddleOCR microservice, normalizes the bounding boxes to [0, 1000],
    and formats the result as a text string to inject into the LLM prompt.
    """
    payload = {"image_base64": image_base64}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(settings.PADDLEOCR_URL, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            # Format and normalize coordinates
            formatted_results = []
            for item in results:
                text = item.get("text", "")
                bbox = item.get("bbox", [0, 0, 0, 0])
                
                # Normalize bbox to [0, 1000] scale
                # format: [xmin, ymin, xmax, ymax]
                try:
                    norm_bbox = [
                        int((bbox[0] / img_w) * 1000),
                        int((bbox[1] / img_h) * 1000),
                        int((bbox[2] / img_w) * 1000),
                        int((bbox[3] / img_h) * 1000)
                    ]
                    # Ensure within bounds
                    norm_bbox = [max(0, min(1000, x)) for x in norm_bbox]
                except ZeroDivisionError:
                    norm_bbox = [0, 0, 0, 0]
                
                formatted_results.append(f'- Text: "{text}", BBox: {norm_bbox}')
            
            if not formatted_results:
                return "Không tìm thấy văn bản nào."
                
            return "\n".join(formatted_results)
            
    except Exception as e:
        logger.error(f"PaddleOCR API call failed: {str(e)}")
        # We don't fail the whole request if OCR fails, we just return empty context
        return "Lỗi khi chạy OCR sơ bộ, hãy tự phân tích từ ảnh."

