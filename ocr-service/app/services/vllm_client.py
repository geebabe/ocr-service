import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn Việt Nam.
Đọc ảnh hóa đơn và trả về JSON với ĐÚNG 7 trường bên dưới.

Mô tả từng trường:
1. "invoice_number": Số hóa đơn hoặc ký hiệu hóa đơn. Thường nằm ở đầu hóa đơn, có nhãn "Số:", "No:", "Số HĐ:", "Ký hiệu:". Ví dụ: "0012345", "AA/20E-0001234", "H:00064286".
2. "invoice_date": Ngày xuất hóa đơn. Thường nằm gần số hóa đơn, có nhãn "Ngày", "Date". Giữ nguyên định dạng gốc. Ví dụ: "14/08/2020", "2020-08-14".
3. "vendor_name": Tên đơn vị bán hàng / cửa hàng / công ty xuất hóa đơn. Thường nằm ở phần đầu hoặc header của hóa đơn. Ví dụ: "VinCommerce", "Công ty TNHH ABC", "Siêu thị Big C".
4. "vendor_tax_code": Mã số thuế (MST) của người bán. Thường có nhãn "MST:", "Mã số thuế:", "Tax code:". Ví dụ: "0123456789", "0100123456-001".
5. "subtotal": Cộng tiền hàng CHƯA bao gồm thuế. Thường có nhãn "Cộng tiền hàng", "Subtotal", "Thành tiền". Giữ nguyên con số gốc. Ví dụ: "1.500.000", "26500".
6. "tax": Số tiền thuế GTGT (VAT). Thường có nhãn "Thuế GTGT", "Tiền thuế", "VAT". Ví dụ: "150.000", "0".
7. "total_amount": Tổng cộng tiền thanh toán (đã gồm thuế). Thường có nhãn "Tổng thanh toán", "Tổng cộng", "Total", "Tổng tiền". Ví dụ: "1.650.000", "26500".

QUY TẮC:
- Nếu trường nào KHÔNG tìm thấy trên hóa đơn, trả về null cho trường đó.
- KHÔNG tự bịa giá trị. Chỉ trích xuất những gì thực sự có trên ảnh.
- KHÔNG trả về danh sách sản phẩm, mặt hàng, items, hay products.
- Chỉ trả về JSON thuần, KHÔNG kèm markdown, giải thích, hay bất kỳ text nào khác.
"""

USER_MESSAGE = "Hãy đọc ảnh hóa đơn này và trích xuất thông tin sang JSON theo đúng 7 trường đã mô tả."


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
