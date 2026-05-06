from pydantic import BaseModel, Field
from typing import Optional


class VendorInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Tên người bán / công ty")
    tax_code: Optional[str] = Field(default=None, description="Mã số thuế người bán")


class CustomerInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Tên người mua / công ty")
    tax_code: Optional[str] = Field(default=None, description="Mã số thuế người mua")


class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = Field(default=None, description="Số hóa đơn")
    invoice_date: Optional[str] = Field(default=None, description="Ngày hóa đơn")
    vendor: Optional[VendorInfo] = Field(default=None, description="Thông tin người bán")
    customer: Optional[CustomerInfo] = Field(default=None, description="Thông tin người mua")
    subtotal: Optional[str] = Field(default=None, description="Cộng tiền hàng trước thuế")
    tax: Optional[str] = Field(default=None, description="Tiền thuế GTGT")
    total_amount: Optional[str] = Field(default=None, description="Tổng thanh toán")

    def to_markdown(self) -> str:
        """Render the extracted invoice data as a fixed Markdown template."""
        def val(v):
            return v if v else "N/A"

        vendor_name = val(self.vendor.name if self.vendor else None)
        vendor_tax = val(self.vendor.tax_code if self.vendor else None)
        customer_name = val(self.customer.name if self.customer else None)
        customer_tax = val(self.customer.tax_code if self.customer else None)

        return (
            "## Thông tin hóa đơn\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Số hóa đơn | {val(self.invoice_number)} |\n"
            f"| Ngày | {val(self.invoice_date)} |\n\n"
            "## Người bán\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Tên | {vendor_name} |\n"
            f"| Mã số thuế | {vendor_tax} |\n\n"
            "## Người mua\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Tên | {customer_name} |\n"
            f"| Mã số thuế | {customer_tax} |\n\n"
            "## Tổng tiền\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Cộng tiền hàng | {val(self.subtotal)} |\n"
            f"| Thuế GTGT | {val(self.tax)} |\n"
            f"| Tổng thanh toán | {val(self.total_amount)} |"
        )


class APIResponse(BaseModel):
    success: bool
    data: Optional[str] = Field(default=None, description="Extracted invoice content in Markdown format")
    error: Optional[str] = None
    metadata: Optional[dict] = None
