from pydantic import BaseModel, Field
from typing import Optional


class APIResponse(BaseModel):
    success: bool
    data: Optional[str] = Field(default=None, description="Extracted invoice content in Markdown format")
    error: Optional[str] = None
    metadata: Optional[dict] = None
