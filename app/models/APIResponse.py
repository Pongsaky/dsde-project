from pydantic import BaseModel
from typing import Optional
from .GraphData import GraphData

class APIResponse(BaseModel):
    message: str
    newGraph: Optional[GraphData]