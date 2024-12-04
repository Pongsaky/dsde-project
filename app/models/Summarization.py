from typing import Optional
from pydantic import BaseModel

class SummarizationQueryModel(BaseModel):
    text: str
    top_k: int = 5

class SummarizationCourseIdModel(BaseModel):
    course_id: str
    top_k: int = 5
    is_upsert: Optional[bool] = True

class RegenerateModel(BaseModel):
    course_id: str
    part: str
    top_k: int = 5