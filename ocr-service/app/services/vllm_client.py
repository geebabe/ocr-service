import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn.
Phân tích ảnh hóa đơn và trả về JSON theo đúng schema được cung cấp.
Nếu không tìm thấy giá trị của trường nào, để null.
KHÔNG liệt kê danh sách sản phẩm/mặt hàng."""

USER_MESSAGE = "Trích xuất thông tin từ hóa đơn này."


async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the vLLM server via HTTP asynchronously with guided JSON structured output.
    Uses InvoiceExtraction schema to enforce valid JSON from the model.
    """
    headers = {"Content-Type": "application/json"}

    # Generate JSON schema from Pydantic model for guided decoding
    invoice_schema = InvoiceExtraction.model_json_schema()

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
                        "text": USER_MESSAGE,
                    },
                ],
            },
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE,
        "guided_json": invoice_schema,
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
