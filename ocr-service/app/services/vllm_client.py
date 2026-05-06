import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn.
CHỈ trích xuất các trường sau:
- invoice_number (Số hóa đơn)
- invoice_date (Ngày hóa đơn)
- vendor_name (Tên người bán)
- vendor_tax_code (MST người bán)
- customer_name (Tên người mua)
- customer_tax_code (MST người mua)
- subtotal (Cộng tiền hàng)
- tax (Tiền thuế)
- total_amount (Tổng thanh toán)

TUYỆT ĐỐI KHÔNG trả về danh sách 'items' hay 'products'.
Trả về JSON đúng schema, không kèm giải thích."""

USER_MESSAGE = "Trích xuất thông tin từ hóa đơn này sang JSON."


async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the model server via HTTP asynchronously with guided JSON structured output.
    Uses InvoiceExtraction schema to constrain generation via outlines.
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
        "max_tokens": 1000,
        "temperature": 0,
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
