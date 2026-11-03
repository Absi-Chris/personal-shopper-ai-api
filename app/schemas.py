from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    session_id: Optional[str] = None

class ProductCreate(BaseModel):
    sku: str
    name: str
    brand: str = ""
    category: str = ""
    description: str = ""
    price: float = Field(ge=0)
    stock: int = Field(ge=0, default=0)
    rating: float = Field(ge=0, le=5, default=0)
    attributes: dict = {}

class CartAdd(BaseModel):
    product_id: int
    quantity: int = Field(gt=0, le=100)

class CartRemove(BaseModel):
    product_id: int
