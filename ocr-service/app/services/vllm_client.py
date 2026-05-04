import httpx
from app.core.config import settings
from app.core.logger import logger

SYSTEM_PROMPT = """Bạn là hệ thống OCR. Trích xuất thông tin từ hóa đơn.
Với mỗi trường, hãy dùng grounding để chỉ rõ vị trí text trên ảnh.

Trả về JSON theo đúng format (bbox là tọa độ [xmin, ymin, xmax, ymax]):

{
  "invoice_number": {"value": "...", "bounding_box": [xmin, ymin, xmax, ymax]},
  "invoice_date":   {"value": "...", "bounding_box": [xmin, ymin, xmax, ymax]},
  "vendor": {
    "name": {"value": "...", "bounding_box": [...]},
    "address": {"value": "...", "bounding_box": [...]},
    "tax_code": {"value": "...", "bounding_box": [...]},
    "phone": {"value": "...", "bounding_box": [...]}
  },
  "customer": {
     "name": {"value": "...", "bounding_box": [...]},
     "address": {"value": "...", "bounding_box": [...]},
     "tax_code": {"value": "...", "bounding_box": [...]}
  },
  "items": [
     {
       "description": {"value": "...", "bounding_box": [...]},
       "quantity": {"value": "...", "bounding_box": [...]},
       "unit_price": {"value": "...", "bounding_box": [...]},
       "total_amount": {"value": "...", "bounding_box": [...]}
     }
  ],
  "subtotal": {"value": "...", "bounding_box": [...]},
  "tax": {"value": "...", "bounding_box": [...]},
  "total_amount": {"value": "...", "bounding_box": [...]},
  "currency": {"value": "...", "bounding_box": [...]},
  "payment_info": {
    "method": {"value": "...", "bounding_box": [...]},
    "bank_account": {"value": "...", "bounding_box": [...]},
    "bank_name": {"value": "...", "bounding_box": [...]}
  }
}

Chỉ trả về JSON thuần, không giải thích."""

async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the vLLM server via HTTP asynchronously.
    """
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": [
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
                        "text": "Trích xuất thông tin từ hóa đơn này."
                    }
                ]
            }
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(settings.VLLM_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
    except Exception as e:
        logger.error(f"vLLM API call failed: {str(e)}")
        raise
