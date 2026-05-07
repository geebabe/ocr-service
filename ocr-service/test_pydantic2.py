import json
from pydantic import BaseModel, Field
from typing import Optional, Union, Tuple

class BBoxField(BaseModel):
    value: Union[str, float, int, None] = Field(description="The extracted text or numeric value")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(default=None)

class VendorInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None)
    address: Optional[BBoxField] = Field(default=None)
    tax_code: Optional[BBoxField] = Field(default=None)
    phone: Optional[BBoxField] = Field(default=None)

class InvoiceExtraction(BaseModel):
    invoice_number: Optional[BBoxField] = Field(default=None)
    invoice_date: Optional[BBoxField] = Field(default=None)
    vendor: Optional[VendorInfo] = Field(default=None)
    subtotal: Optional[BBoxField] = Field(default=None)
    tax: Optional[BBoxField] = Field(default=None)
    total_amount: Optional[BBoxField] = Field(default=None)
    currency: Optional[BBoxField] = Field(default=None)

def _wrap_raw_values(node):
    if isinstance(node, dict):
        if "value" in node:
            return node
        return {k: _wrap_raw_values(v) for k, v in node.items()}
    elif isinstance(node, list):
        return [_wrap_raw_values(item) for item in node]
    elif node is None:
        return None
    else:
        return {"value": node, "bounding_box": None}

raw_json = """{
  "invoice_number": "00181535",
  "invoice_date": "12/08/2020",
  "vendor": {
    "name": "VinCommerce",
    "address": "Số 112 Thanh Nien, P. Cám Thành, TP. Cà Mau, Quang Ninh",
    "tax_code": "024.71066866-39671",
    "phone": "09015345"
  },
  "subtotal": 154.100,
  "tax": 0,
  "total_amount": 154.100,
  "currency": "VND"
}"""

parsed_dict = json.loads(raw_json)
wrapped_dict = _wrap_raw_values(parsed_dict)
try:
    obj = InvoiceExtraction(**wrapped_dict)
    print("SUCCESS")
    print(obj.model_dump_json(indent=2))
except Exception as e:
    print("ERROR:")
    print(e)
