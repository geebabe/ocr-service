import json
import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn.
Hãy phân tích ảnh hóa đơn và trả về thông tin theo đúng JSON schema được cung cấp.
Với mỗi trường, hãy cung cấp giá trị trích xuất và tọa độ bounding box [xmin, ymin, xmax, ymax] trên ảnh gốc.
Chỉ trả về JSON thuần, không thêm giải thích hay markdown.

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
                        "text": "Trích xuất toàn bộ thông tin từ hóa đơn này theo JSON schema đã định nghĩa.",
                    },
                ],
            },
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE,
        # Guided decoding: forces vLLM to produce JSON matching InvoiceExtraction schema
        "extra_body": {
            "guided_json": invoice_schema,
            "guided_decoding_backend": "outlines",
        },
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

