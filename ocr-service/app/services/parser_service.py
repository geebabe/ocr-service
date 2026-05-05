import json
import re
from typing import Any
from app.core.logger import logger

def normalize_keys_recursively(data: Any) -> Any:
    """
    Recursively normalizes dictionary keys to snake_case.
    e.g. 'InvoiceNumber' -> 'invoice_number', 'Tax Code' -> 'tax_code'
    """
    if isinstance(data, dict):
        normalized_dict = {}
        for k, v in data.items():
            # Convert CamelCase or Space Case to snake_case
            k_space = k.replace(" ", "_")
            # Insert underscore before capital letters (if not first char)
            k_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', k_space)
            # Lowercase and clean up multiple underscores
            normalized_key = re.sub(r'_+', '_', k_snake.lower()).strip('_')
            normalized_dict[normalized_key] = normalize_keys_recursively(v)
        return normalized_dict
    elif isinstance(data, list):
        return [normalize_keys_recursively(item) for item in data]
    return data

def parse_vllm_output(output_text: str) -> dict:
    """
    Parses the JSON output from VLM.
    Handles markdown code blocks and common model chatter.
    """
    if not output_text:
        logger.warning("VLM returned empty output")
        return {}

    try:
        # 0. Log raw output for debugging
        logger.info(f"Raw VLM output: {output_text}")
        
        parsed_dict = None
        
        # 1. Try to find JSON inside markdown code blocks first
        json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL | re.IGNORECASE)
        if json_block_match:
            try:
                parsed_dict = json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Found JSON block but failed to decode it, falling back to regex search")

        # 2. Fallback: extract anything between the first { and the last }
        if parsed_dict is None:
            json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in model output")
                return {}
                
            raw_json_str = json_match.group()
            parsed_dict = json.loads(raw_json_str)
        
        # 4. Normalize keys to match schema (lowercase, snake_case recursively)
        return normalize_keys_recursively(parsed_dict)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model output: {str(e)}\nRaw output snippet: {output_text[:200]}...")
        return {}
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}", exc_info=True)
        return {}
