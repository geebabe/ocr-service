import json
import re
from app.core.logger import logger

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
        
        # 1. Try to find JSON inside markdown code blocks first
        json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL | re.IGNORECASE)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Found JSON block but failed to decode it, falling back to regex search")

        # 2. Fallback: extract anything between the first { and the last }
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in model output")
            return {}
            
        raw_json_str = json_match.group()
        
        # 3. Clean up common issues like single quotes or trailing commas
        # (Be careful not to break valid JSON)
        # raw_json_str = raw_json_str.replace("'", '"') # Risky if values contain single quotes
        
        parsed_dict = json.loads(raw_json_str)
        
        # 4. Normalize keys to match schema (lowercase)
        # Some models might capitalize keys like "InvoiceNumber" instead of "invoice_number"
        if isinstance(parsed_dict, dict):
            normalized_dict = {}
            for k, v in parsed_dict.items():
                # Simple normalization: lowercase
                normalized_key = k.lower().replace(" ", "_")
                normalized_dict[normalized_key] = v
            return normalized_dict
            
        return parsed_dict
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model output: {str(e)}\nRaw output snippet: {output_text[:200]}...")
        return {}
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}", exc_info=True)
        return {}
