import json
from app.schemas.response import InvoiceExtraction

schema = InvoiceExtraction.model_json_schema()
print(json.dumps(schema, indent=2))
