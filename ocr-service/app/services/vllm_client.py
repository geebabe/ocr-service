import httpx
from app.core.config import settings
from app.core.logger import logger

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn.
Phân tích ảnh hóa đơn và trả về ĐÚNG định dạng Markdown bên dưới, không thay đổi cấu trúc.
Nếu không tìm thấy giá trị của trường nào, ghi "N/A".
KHÔNG liệt kê danh sách sản phẩm/mặt hàng.
Chỉ trả về Markdown thuần, không thêm giải thích.

## Thông tin hóa đơn

| Trường | Giá trị |
|---|---|
| Số hóa đơn | <giá trị> |
| Ngày | <giá trị> |

## Người bán

| Trường | Giá trị |
|---|---|
| Tên | <giá trị> |
| Mã số thuế | <giá trị> |

## Người mua

| Trường | Giá trị |
|---|---|
| Tên | <giá trị> |
| Mã số thuế | <giá trị> |

## Tổng tiền

| Trường | Giá trị |
|---|---|
| Cộng tiền hàng | <giá trị> |
| Thuế GTGT | <giá trị> |
| Tổng thanh toán | <giá trị> |"""

USER_MESSAGE = "Trích xuất thông tin từ hóa đơn này theo đúng mẫu Markdown đã cho."


async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the vLLM server via HTTP asynchronously.
    Returns structured Markdown text from the model following a fixed template.
    """
    headers = {"Content-Type": "application/json"}

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

