from pydantic import BaseModel

class DetectAdditionalData(BaseModel):
    isNeed: bool
    reason: str