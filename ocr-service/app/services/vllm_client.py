import json
import httpx
from app.core.config import settings
from app.core.logger import logger
from app.schemas.response import InvoiceExtraction

SYSTEM_PROMPT = """You are an advanced OCR and document understanding system. Your goal is to extract structured information from the provided document image with high precision.

### KEY INSTRUCTIONS:
1. **NATIVE GROUNDING**: For every field, you must provide the extracted text (value) and its precise coordinates (bounding_box).
2. **COORDINATE SYSTEM**: All bounding boxes MUST be normalized to a scale of [0, 1000]. The format is [xmin, ymin, xmax, ymax].
3. **OUTPUT FORMAT**: Return strictly valid JSON matching the provided schema. No markdown, no conversational fillers, and no explanations.

### CONTEXTUAL HINTS:
- **Language**: The document is primarily in Vietnamese. Pay close attention to diacritics and specialized terms.
- **Preliminary OCR**: Below is a draft OCR extraction (already normalized to [0, 1000]) to help you identify characters and locations. Use these as hints, but rely on your visual perception if the image contradicts these hints:
{ocr_context}

### SCHEMA DEFINITION:
{schema_str}"""


async def call_vllm_inference(image_base64: str, ocr_context: str = "") -> str:
    """
    Calls the vLLM server via HTTP asynchronously with guided JSON structured output.
    Uses InvoiceExtraction schema to enforce valid JSON structure from the model.
    """
    headers = {"Content-Type": "application/json"}

    # Generate JSON schema from Pydantic model for guided decoding
    invoice_schema = InvoiceExtraction.model_json_schema()
    schema_str = json.dumps(invoice_schema, indent=2)
    system_prompt = SYSTEM_PROMPT.format(schema_str=schema_str, ocr_context=ocr_context)

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
                        "text": "Analyze the attached document and perform structured extraction according to the schema. Utilize the preliminary OCR hints provided in the system context to refine your grounding and text recognition.",
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

