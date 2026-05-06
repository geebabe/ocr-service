import httpx
from app.core.config import settings
from app.core.logger import logger

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích hóa đơn. Hãy trích xuất thông tin từ ảnh và trả về JSON thuần túy.
Không cần tọa độ bounding box, chỉ cần giá trị văn bản chính xác nhất.
Trả về JSON theo đúng định dạng sau:
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
                        "text": "Trích xuất thông tin từ hóa đơn này. Bắt buộc trả về JSON theo đúng format đã yêu cầu. Không thêm bất kỳ giải thích nào."
                    }
                ]
            }
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE
    }
    
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(settings.VLLM_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
    except Exception as e:
        logger.error(f"vLLM API call failed: {str(e)}")
        raise
