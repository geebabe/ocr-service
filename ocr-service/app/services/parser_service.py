import json
from app.schemas.response import InvoiceExtraction
from app.core.logger import logger


def parse_vllm_output(output_text: str) -> str:
    """
    Parses structured JSON from vLLM into an InvoiceExtraction model,
    then renders it as a consistent Markdown string.
    Falls back to raw text if parsing fails.
    """
    clean_text = output_text.strip()

    # Strip markdown code fences if the model wraps output in them
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3].strip()

    try:
        parsed_dict = json.loads(clean_text)
        invoice = InvoiceExtraction(**parsed_dict)
        logger.info("Successfully parsed structured JSON → Markdown")
        return invoice.to_markdown()
    except Exception as e:
        logger.error(f"Failed to parse structured output: {e}")
        logger.error(f"Raw output: {clean_text[:500]}")
        # Fallback: return raw text as-is
        return clean_text
