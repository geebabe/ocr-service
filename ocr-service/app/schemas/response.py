from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Union

class BBoxField(BaseModel):
    value: Union[str, float, int, None] = Field(description="The extracted text or numeric value")
    bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        default=None, 
        description="Bounding box coordinates [xmin, ymin, xmax, ymax] in original image pixels"
    )

class InvoiceItem(BaseModel):
    description: BBoxField = Field(description="Item name or description")
    quantity: Optional[BBoxField] = Field(default=None, description="Quantity of the item")
    unit_price: Optional[BBoxField] = Field(default=None, description="Price per unit")
    total_amount: Optional[BBoxField] = Field(default=None, description="Total amount for this item")

class VendorInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None, description="Name of the vendor")
    address: Optional[BBoxField] = Field(default=None, description="Address of the vendor")
    tax_code: Optional[BBoxField] = Field(default=None, description="Tax identification number")
    phone: Optional[BBoxField] = Field(default=None, description="Contact phone number")

class CustomerInfo(BaseModel):
    name: Optional[BBoxField] = Field(default=None, description="Name of the customer")
    address: Optional[BBoxField] = Field(default=None, description="Address of the customer")
    tax_code: Optional[BBoxField] = Field(default=None, description="Tax identification number")

class PaymentInfo(BaseModel):
    method: Optional[BBoxField] = Field(default=None, description="Payment method (e.g., Cash, Credit Card, Bank Transfer)")
    bank_account: Optional[BBoxField] = Field(default=None, description="Bank account number if applicable")
    bank_name: Optional[BBoxField] = Field(default=None, description="Bank name if applicable")

class InvoiceExtraction(BaseModel):
    invoice_number: Optional[BBoxField] = Field(default=None)
    invoice_date: Optional[BBoxField] = Field(default=None)
    
    vendor: Optional[VendorInfo] = Field(default=None)
    # customer: Optional[CustomerInfo] = Field(default=None)

    # items: List[InvoiceItem] = Field(default_factory=list, description="List of items in the invoice")
    
    subtotal: Optional[BBoxField] = Field(default=None, description="Total amount before tax")
    tax: Optional[BBoxField] = Field(default=None, description="Tax amount")
    total_amount: Optional[BBoxField] = Field(default=None, description="Final total amount including tax")
    currency: Optional[BBoxField] = Field(default=None, description="Currency code (e.g., VND, USD)")
    
    # payment_info: Optional[PaymentInfo] = Field(default=None)

class APIResponse(BaseModel):
    success: bool
    data: Optional[InvoiceExtraction] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
