from pydantic import BaseModel, Field
from typing import Optional


class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = Field(
        default=None,
        description="Số hóa đơn / ký hiệu hóa đơn (VD: 0012345, AA/20E-0001234)"
    )
    invoice_date: Optional[str] = Field(
        default=None,
        description="Ngày xuất hóa đơn, giữ nguyên định dạng gốc (VD: 18/08/2020)"
    )
    vendor_name: Optional[str] = Field(
        default=None,
        description="Tên đơn vị / công ty / cửa hàng bán hàng (người bán)"
    )
    vendor_tax_code: Optional[str] = Field(
        default=None,
        description="Mã số thuế của người bán"
    )
    subtotal: Optional[str] = Field(
        default=None,
        description="Cộng tiền hàng chưa bao gồm thuế, giữ nguyên định dạng số gốc"
    )
    tax: Optional[str] = Field(
        default=None,
        description="Số tiền thuế GTGT (VAT), giữ nguyên định dạng số gốc"
    )
    total_amount: Optional[str] = Field(
        default=None,
        description="Tổng cộng tiền thanh toán (đã bao gồm thuế), giữ nguyên định dạng số gốc"
    )

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
