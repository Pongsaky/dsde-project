from pydantic import BaseModel
from enum import Enum
from .GraphData import GraphData  # Updated import

class MessageType(str, Enum):
    InitialChat = "InitialChat"
    ContinueChat = "ContinueChat"

class UserInput(BaseModel):
    message: str
    type: MessageType
    currentGraph: GraphData