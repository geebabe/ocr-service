from pydantic import BaseModel, Field
from typing import Optional


class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = Field(default=None, description="Số hóa đơn")
    invoice_date: Optional[str] = Field(default=None, description="Ngày hóa đơn")
    vendor_name: Optional[str] = Field(default=None, description="Tên công ty người bán")
    vendor_tax_code: Optional[str] = Field(default=None, description="Mã số thuế người bán")
    customer_name: Optional[str] = Field(default=None, description="Tên người mua / đơn vị mua hàng")
    customer_tax_code: Optional[str] = Field(default=None, description="Mã số thuế người mua")
    subtotal: Optional[str] = Field(default=None, description="Cộng tiền hàng (chưa thuế)")
    tax: Optional[str] = Field(default=None, description="Tiền thuế GTGT")
    total_amount: Optional[str] = Field(default=None, description="Tổng cộng tiền thanh toán")

    def to_markdown(self) -> str:
        """Render the extracted invoice data as a fixed Markdown template."""
        def val(v):
            return v if v else "N/A"

        return (
            "## Thông tin hóa đơn\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Số hóa đơn | {val(self.invoice_number)} |\n"
            f"| Ngày | {val(self.invoice_date)} |\n\n"
            "## Người bán\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Tên | {val(self.vendor_name)} |\n"
            f"| Mã số thuế | {val(self.vendor_tax_code)} |\n\n"
            "## Người mua\n\n"
            "| Trường | Giá trị |\n"
            "|---|---|\n"
            f"| Tên | {val(self.customer_name)} |\n"
            f"| Mã số thuế | {val(self.customer_tax_code)} |\n\n"
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
