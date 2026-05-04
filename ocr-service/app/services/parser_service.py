import json
import re
from app.core.logger import logger

def parse_vllm_output(output_text: str) -> dict:
    """
    Parses the JSON output from Qwen3-VL.
    Returns a raw dictionary representing the extracted fields.
    """
    try:
        # 1. Clean the output: extract JSON string
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in model output")
            return {}
            
        raw_json_str = json_match.group()
        
        parsed_dict = json.loads(raw_json_str)
        return parsed_dict
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model output: {str(e)}\nRaw output: {output_text}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}", exc_info=True)
        return {}
