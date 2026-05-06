from app.core.logger import logger


def parse_vllm_output(output_text: str) -> str:
    """
    Cleans up raw Markdown output from vLLM.
    Strips surrounding code fences if present, returns clean Markdown text.
    """
    clean_text = output_text.strip()

    # Strip markdown code fences if the model wraps its output in them
    if clean_text.startswith("```markdown"):
        clean_text = clean_text[len("```markdown"):].strip()
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3].strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3].strip()

    logger.info(f"Parsed Markdown output ({len(clean_text)} chars)")
    return clean_text
