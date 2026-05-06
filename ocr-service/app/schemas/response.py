from pydantic import BaseModel, Field
from typing import Optional, List, Union

class InvoiceItem(BaseModel):
    description: Optional[str] = Field(default=None, description="Item name or description")
    quantity: Optional[Union[str, float]] = Field(default=None, description="Quantity of the item")
    unit_price: Optional[Union[str, float]] = Field(default=None, description="Price per unit")
    total_amount: Optional[Union[str, float]] = Field(default=None, description="Total amount for this item")

class VendorInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Name of the vendor")
    address: Optional[str] = Field(default=None, description="Address of the vendor")
    tax_code: Optional[str] = Field(default=None, description="Tax identification number")
    phone: Optional[str] = Field(default=None, description="Contact phone number")

class CustomerInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Name of the customer")
    address: Optional[str] = Field(default=None, description="Address of the customer")
    tax_code: Optional[str] = Field(default=None, description="Tax identification number")

class PaymentInfo(BaseModel):
    method: Optional[str] = Field(default=None, description="Payment method (e.g., Cash, Credit Card, Bank Transfer)")
    bank_account: Optional[str] = Field(default=None, description="Bank account number if applicable")
    bank_name: Optional[str] = Field(default=None, description="Bank name if applicable")

class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = Field(default=None)
    invoice_date: Optional[str] = Field(default=None)
    
    vendor: Optional[VendorInfo] = Field(default=None)
    customer: Optional[CustomerInfo] = Field(default=None)
    
    items: List[InvoiceItem] = Field(default_factory=list, description="List of items in the invoice")
    
    subtotal: Optional[Union[str, float]] = Field(default=None, description="Total amount before tax")
    tax: Optional[Union[str, float]] = Field(default=None, description="Tax amount")
    total_amount: Optional[Union[str, float]] = Field(default=None, description="Final total amount including tax")
    currency: Optional[str] = Field(default=None, description="Currency code (e.g., VND, USD)")
    
    payment_info: Optional[PaymentInfo] = Field(default=None)

class APIResponse(BaseModel):
    success: bool
    data: Optional[InvoiceExtraction] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
