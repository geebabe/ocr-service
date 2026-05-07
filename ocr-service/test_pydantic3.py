import json
import sys
sys.path.insert(0, '/Users/phoenix/Projects/ocr-invoices/ocr-service')
from app.schemas.response import InvoiceExtraction
from app.services.parser_service import _wrap_raw_values

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
  "currency": "VND",
  "invoice_type": "HÓA ĐƠN BÁN HÀNG"
}"""

parsed_dict = json.loads(raw_json)
wrapped_dict = _wrap_raw_values(parsed_dict)
print("Wrapped dict:", wrapped_dict)

try:
    obj = InvoiceExtraction(**wrapped_dict)
    print("SUCCESS")
    print(obj.model_dump_json(indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
