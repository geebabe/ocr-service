import httpx
import instructor
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

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

# Initialize instructor client
# We use the base URL (without /v1/chat/completions) for instructor
base_url = settings.VLLM_URL.split("/chat/completions")[0]
instructor_client = instructor.from_httpx(
    httpx.AsyncClient(timeout=120.0),
    base_url=base_url,
)

async def call_vllm_inference(image_base64: str) -> InvoiceExtraction:
    """
    Calls the VLM server and returns a structured InvoiceExtraction object using instructor.
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
                    "text": "Trích xuất thông tin từ hóa đơn này. Bắt buộc trả về JSON theo đúng format đã yêu cầu. Không thêm bất kỳ giải thích nào."
                }
            ]
        }
    ]
    
    try:
        # Use instructor to get structured output
        # Mode.MD_JSON is often more robust for VLMs that like to wrap JSON in markdown blocks
        result = await instructor_client.chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=messages,
            response_model=InvoiceExtraction,
            max_tokens=settings.VLLM_MAX_TOKENS,
            temperature=settings.VLLM_TEMPERATURE,
            # We use MD_JSON mode as Qwen often outputs ```json ... ```
            mode=instructor.Mode.MD_JSON 
        )
        return result
            
    except Exception as e:
        logger.error(f"Structured VLM API call failed: {str(e)}")
        raise
