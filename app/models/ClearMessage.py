from pydantic import BaseModel

class ClearMessage(BaseModel):
    chat_id : str