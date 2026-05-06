import json
import re
from typing import Any, Dict, Optional
from app.schemas.response import InvoiceExtraction
from app.core.logger import logger

# Maps common model output field names → our schema field names
FIELD_ALIASES: Dict[str, str] = {
    # invoice_number
    "invoice_number": "invoice_number",
    "invoice_no": "invoice_number",
    "invoice_id": "invoice_number",
    "so_hoa_don": "invoice_number",
    "số hóa đơn": "invoice_number",
    # invoice_date
    "invoice_date": "invoice_date",
    "date": "invoice_date",
    "ngay": "invoice_date",
    "ngày": "invoice_date",
    # vendor_name
    "vendor_name": "vendor_name",
    "merchant_name": "vendor_name",
    "seller_name": "vendor_name",
    "store_name": "vendor_name",
    "company_name": "vendor_name",
    "ten_nguoi_ban": "vendor_name",
    # vendor_tax_code
    "vendor_tax_code": "vendor_tax_code",
    "merchant_tax_code": "vendor_tax_code",
    "seller_tax_code": "vendor_tax_code",
    "mst_nguoi_ban": "vendor_tax_code",
    "tax_id": "vendor_tax_code",
    # customer_name
    "customer_name": "customer_name",
    "buyer_name": "customer_name",
    "ten_nguoi_mua": "customer_name",
    # customer_tax_code
    "customer_tax_code": "customer_tax_code",
    "buyer_tax_code": "customer_tax_code",
    "mst_nguoi_mua": "customer_tax_code",
    # subtotal
    "subtotal": "subtotal",
    "sub_total": "subtotal",
    "amount_before_tax": "subtotal",
    "cong_tien_hang": "subtotal",
    # tax
    "tax": "tax",
    "tax_amount": "tax",
    "vat": "tax",
    "vat_amount": "tax",
    "thue": "tax",
    "thue_gtgt": "tax",
    # total_amount
    "total_amount": "total_amount",
    "total": "total_amount",
    "grand_total": "total_amount",
    "tong_tien": "total_amount",
    "tong_thanh_toan": "total_amount",
}

# Fields to skip entirely
SKIP_FIELDS = {
    "items", "products", "line_items", "details",
    "merchant_address", "merchant_phone", "merchant_website",
    "invoice_time", "invoice_status", "tax_rate", "tax_included",
}


def _map_fields(raw_dict: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map arbitrary model output field names to our schema field names."""
    mapped: Dict[str, Optional[str]] = {}

    for key, value in raw_dict.items():
        lower_key = key.lower().strip()

        if lower_key in SKIP_FIELDS:
            continue

        target = FIELD_ALIASES.get(lower_key)
        if target and target not in mapped:
            # Convert numeric values to string
            if value is not None:
                mapped[target] = str(value)
            else:
                mapped[target] = None

    return mapped


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from text, handling code fences and truncation."""
    clean = text.strip()

    # Strip markdown code fences
    if "```json" in clean:
        clean = clean.split("```json")[1]
        if "```" in clean:
            clean = clean.split("```")[0]
        clean = clean.strip()
    elif clean.startswith("```"):
        clean = clean[3:].strip()
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    # Try parsing as-is
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Handle truncated JSON: try to close it
    # Find the last complete key-value pair and close the object
    try:
        # Remove any trailing incomplete string/value
        # Find last complete line ending with comma or value
        truncated = re.sub(r',\s*"[^"]*$', '', clean)  # remove incomplete key
        truncated = re.sub(r',\s*$', '', truncated)     # remove trailing comma
        # Close any unclosed brackets
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        truncated += ']' * max(0, open_brackets)
        truncated += '}' * max(0, open_braces)
        return json.loads(truncated)
    except json.JSONDecodeError:
        pass

    return None


def parse_vllm_output(output_text: str) -> str:
    """
    Parses JSON from vLLM, maps fields to InvoiceExtraction schema,
    then renders as consistent Markdown.
    Falls back to raw text if all parsing fails.
    """
    logger.info(f"Raw model output ({len(output_text)} chars): {output_text[:200]}...")

    parsed_dict = _extract_json(output_text)

    if parsed_dict is None:
        logger.error("Could not extract valid JSON from model output")
        logger.error(f"Full output: {output_text[:500]}")
        return output_text

    # Map whatever fields the model returned to our schema
    mapped = _map_fields(parsed_dict)
    logger.info(f"Mapped fields: {mapped}")

    invoice = InvoiceExtraction(**mapped)
    return invoice.to_markdown()
