from pydantic import BaseModel
from typing import Optional
from .GraphData import GraphData

class APIResponse(BaseModel):
    chat_id : str
    message: str
    newGraph: Optional[GraphData]