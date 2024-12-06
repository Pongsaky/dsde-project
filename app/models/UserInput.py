from pydantic import BaseModel
from .GraphData import GraphData
from typing import Optional

class UserInput(BaseModel):
    chat_id: str
    message: str
    currentGraph: Optional[GraphData]