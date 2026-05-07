from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Tuple, Union, Any

class BBoxField(BaseModel):
    value: Union[str, float, int, None] = Field(description="The extracted text or numeric value")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        default=None, 
        description="Bounding box coordinates [xmin, ymin, xmax, ymax] in original image pixels"
    )

    @model_validator(mode='before')
    @classmethod
    def handle_primitives(cls, data: Any) -> Any:
        if isinstance(data, (str, int, float)) or data is None:
            return {"value": data}
        return data

class VendorInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None, description="Name of the vendor")
    tax_code: Optional[BBoxField] = Field(default=None, description="Tax identification number")

class InvoiceExtraction(BaseModel):
    invoice_number: Optional[BBoxField] = Field(default=None)
    invoice_date: Optional[BBoxField] = Field(default=None)
    vendor: Optional[VendorInfo] = Field(default=None)
    total_amount: Optional[BBoxField] = Field(default=None, description="Final total amount including tax")

class APIResponse(BaseModel):
    success: bool
    data: Optional[InvoiceExtraction] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
