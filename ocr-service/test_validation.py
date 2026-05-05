from app.schemas.response import InvoiceExtraction

data = {
    "invoice_date": "18/08/2020",
    "vendor": {
        "name": "MINIMART ANAN",
        "address": "Chợ Sủi Phú Thị Gia Lâm"
    },
    "items": [
        {
            "description": "Nước tẩy toilet Gift 700ml NT",
            "quantity": "1",
            "unit_price": "23,000",
            "total_amount": "23,000"
        }
    ],
    "subtotal": "176,000",
    "tax": "0",
    "total_amount": "176,000",
    "currency": "VND"
}

try:
    obj = InvoiceExtraction(**data)
    print("Success!")
    print(obj.model_dump())
except Exception as e:
    print("Error:", e)
