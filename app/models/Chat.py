from pydantic import BaseModel

class ChatModel(BaseModel):
    role: str
    content: str
