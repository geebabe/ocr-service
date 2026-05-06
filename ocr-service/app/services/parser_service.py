import json
import re
from app.schemas.response import InvoiceExtraction
from app.core.logger import logger

def parse_vllm_output(output_text: str, img_w: int, img_h: int) -> InvoiceExtraction:
    """
    Parses the JSON output from the semantic VLM.
    Extracts the JSON block and maps it to the InvoiceExtraction model.
    """
    try:
        logger.debug(f"RAW VLM OUTPUT:\n{output_text}\n---")
        
        # Extract JSON string block
        json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in model output")
            return InvoiceExtraction()
            
        raw_json_str = json_match.group()
        
        # Parse JSON
        parsed_dict = json.loads(raw_json_str)
        
        # Instantiate Pydantic model directly since the schema matches
        return InvoiceExtraction(**parsed_dict)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model output: {str(e)}\nRaw output: {output_text}")
        return InvoiceExtraction()
    except Exception as e:
        logger.error(f"Error parsing model output: {str(e)}", exc_info=True)
        return InvoiceExtraction()
