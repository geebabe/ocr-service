import json
import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """Bạn là chuyên gia OCR sử dụng mô hình Qwen-VL.
Nhiệm vụ: Trích xuất thông tin từ hóa đơn và cung cấp tọa độ chính xác cho mỗi trường.

QUY TẮC TRẢ VỀ:
1. Trả về JSON THUẦN theo schema.
2. Với mỗi trường thông tin, BẮT BUỘC phải có:
   - "value": Giá trị văn bản trích xuất được.
   - "bounding_box": Mảng 4 số nguyên [xmin, ymin, xmax, ymax] được CHUẨN HÓA (normalized) về thang đo 0-1000.
3. Sử dụng khả năng NATIVE GROUNDING của mô hình để xác định tọa độ. Tọa độ (0,0) là góc trên bên trái, (1000,1000) là góc dưới bên phải.

JSON Schema:
{schema_str}"""


async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the vLLM server via HTTP asynchronously with guided JSON structured output.
    Uses InvoiceExtraction schema to enforce valid JSON structure from the model.
    """
    headers = {"Content-Type": "application/json"}

    # Generate JSON schema from Pydantic model for guided decoding
    invoice_schema = InvoiceExtraction.model_json_schema()
    schema_str = json.dumps(invoice_schema, indent=2)
    system_prompt = SYSTEM_PROMPT.format(schema_str=schema_str)

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Hãy trích xuất tất cả các trường thông tin từ hóa đơn này. Với mỗi trường, hãy tìm (grounding) tọa độ chính xác của nó trên ảnh và trả về theo định dạng JSON [xmin, ymin, xmax, ymax] normalized 0-1000.",
                    },
                ],
            },
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE,
        # Guided decoding: forces vLLM to produce JSON matching InvoiceExtraction schema
        "guided_json": invoice_schema,
        "guided_decoding_backend": "outlines",
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

