import httpx
from app.core.config import settings
from app.core.logger import logger

SYSTEM_PROMPT = "Bạn là hệ thống OCR hóa đơn chuyên nghiệp. Hãy trích xuất dữ liệu và trả về JSON theo yêu cầu."

async def call_vlm_inference(image_base64: str, model_url: str, model_name: str) -> str:
    """
    Calls a VLM server via HTTP asynchronously.
    Supports any OpenAI-compatible chat completion endpoint.
    """
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model_name,
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
                        "text": """Trích xuất thông tin từ hóa đơn này và trả về JSON theo đúng định dạng sau:

{
  "invoice_number": "...",
  "invoice_date": "...",
  "vendor": {
    "name": "...",
    "address": "...",
    "tax_code": "...",
    "phone": "..."
  },
  "customer": {
     "name": "...",
     "address": "...",
     "tax_code": "..."
  },
  "items": [
     {
       "description": "...",
       "quantity": "...",
       "unit_price": "...",
       "total_amount": "..."
     }
  ],
  "subtotal": "...",
  "tax": "...",
  "total_amount": "...",
  "currency": "...",
  "payment_info": {
    "method": "...",
    "bank_account": "...",
    "bank_name": "..."
  }
}

Chỉ trả về JSON, không có văn bản giải thích."""
                    }
                ]
            }
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(model_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
    except Exception as e:
        logger.error(f"VLM API call failed for {model_name} at {model_url}: {str(e)}")
        raise
