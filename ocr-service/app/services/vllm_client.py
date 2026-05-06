import httpx
from app.core.config import settings
from app.core.logger import logger

SYSTEM_PROMPT = """Bạn là hệ thống OCR chuyên trích xuất thông tin từ hóa đơn.
Hãy phân tích ảnh hóa đơn và trả về thông tin dưới dạng Markdown.
Sử dụng các tiêu đề, bảng hoặc danh sách để trình bày thông tin một cách rõ ràng (ví dụ: thông tin người bán, người mua, danh sách mặt hàng, tổng tiền).
Chỉ trả về Markdown thuần, không thêm giải thích ngoài lề."""


async def call_vllm_inference(image_base64: str) -> str:
    """
    Calls the vLLM server via HTTP asynchronously.
    Returns raw Markdown text from the model.
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
                        "text": "Trích xuất toàn bộ thông tin từ hóa đơn này và trả về định dạng Markdown đẹp mắt.",
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

