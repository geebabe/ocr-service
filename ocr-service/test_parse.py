import sys
sys.path.insert(0, '/Users/phoenix/Projects/ocr-invoices/ocr-service')
# Bypass config error by removing it from app/__init__.py or doing this:
import builtins
class MockConfig:
    pass
sys.modules['app.core.config'] = MockConfig()
sys.modules['app.core.config.settings'] = MockConfig()

from app.services.parser_service import parse_vllm_output

raw_text = """```json
{
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
}
```"""

res = parse_vllm_output(raw_text, 1000, 1000)
print(res.model_dump_json(indent=2))
