from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Union

class BBoxField(BaseModel):
    value: Union[str, float, int, None] = Field(description="The extracted text or numeric value")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        default=None, 
        description="Bounding box coordinates [xmin, ymin, xmax, ymax] in original image pixels"
    )

class VendorInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None, description="Name of the vendor")
    tax_code: Optional[BBoxField] = Field(default=None, description="Tax identification number")

class CustomerInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None, description="Name of the customer")
    tax_code: Optional[BBoxField] = Field(default=None, description="Tax identification number")

class InvoiceExtraction(BaseModel):
    invoice_number: Optional[BBoxField] = Field(default=None)
    invoice_date: Optional[BBoxField] = Field(default=None)
    
    vendor: Optional[VendorInfo] = Field(default=None)
    customer: Optional[CustomerInfo] = Field(default=None)
    
    subtotal: Optional[BBoxField] = Field(default=None, description="Total amount before tax")
    tax: Optional[BBoxField] = Field(default=None, description="Tax amount")
    total_amount: Optional[BBoxField] = Field(default=None, description="Final total amount including tax")

class APIResponse(BaseModel):
    success: bool
    data: Optional[InvoiceExtraction] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
