import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction
import json

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên nghiệp. Trích xuất thông tin từ hóa đơn.
Với mỗi trường, hãy dùng grounding để chỉ rõ vị trí text trên ảnh.

Trả về ĐÚNG CẤU TRÚC JSON sau (bbox là tọa độ [xmin, ymin, xmax, ymax], chuẩn hóa 0-1000):

{
  "invoice_number": {"value": "...", "bounding_box": [...]},
  "invoice_date":   {"value": "...", "bounding_box": [...]},
  "vendor": {
    "name": {"value": "...", "bounding_box": [...]},
    "tax_code": {"value": "...", "bounding_box": [...]},
  },
  "total_amount": {"value": "...", "bounding_box": [...]},
}

CHỈ trả về JSON thuần túy, không giải thích thêm."""

async def call_vllm_inference(image_base64: str) -> InvoiceExtraction:
    """
    Calls the vLLM server to extract minimal invoice data.
    Uses guided_json directly via httpx to avoid openai SDK dependency issues.
    """
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "Trích xuất thông tin hóa đơn này. Bắt buộc trả về đúng định dạng JSON đã cho."
                }
            ]
        }
    ]
    
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE,
        "guided_json": InvoiceExtraction.model_json_schema()
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                settings.VLLM_URL,
                json=payload
            )
            response.raise_for_status()
            
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]
        
        if not content:
            raise ValueError("VLM returned an empty response")
            
        # Robustly extract JSON block in case the model still wraps it in markdown
        import re
        json_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
        logger.debug(f"VLM Output Content (Cleaned): {content}")
        
        # Parse the JSON string back into the Pydantic model
        return InvoiceExtraction.model_validate_json(content)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from VLM API: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Structured VLM API call failed: {str(e)}")
        raise
