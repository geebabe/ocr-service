from typing import Any
import json

def _wrap_raw_values(node: Any) -> Any:
    """
    Recursively wrap raw scalar values into the BBoxField format.
    """
    if isinstance(node, dict):
        # If the dict looks like a BBoxField (has 'value' key), leave it as is
        if "value" in node:
            return node
        
        return {k: _wrap_raw_values(v) for k, v in node.items()}
    elif isinstance(node, list):
        return [_wrap_raw_values(item) for item in node]
    elif node is None:
        return None
    else:
        # It's a raw scalar (str, int, float, bool)
        return {"value": node, "bounding_box": None}

raw = {
  "invoice_number": "00181535",
  "vendor": {
    "name": "VinCommerce",
    "address": "Số 112 Thanh Nien, P. Cám Thành, TP. Cà Mau, Quang Ninh"
  }
}

wrapped = _wrap_raw_values(raw)
print(json.dumps(wrapped, indent=2))
