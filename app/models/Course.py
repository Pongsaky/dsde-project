from typing import List
from pydantic import BaseModel

class SummaryModel(BaseModel):
    attentionGrabber: str
    summaryPart: str
    instructor: str
    targetAudience: List[str]
    courseBenefit: List[str]
    callToAction: str
    hashtag: str

class CourseModel(BaseModel):
    course_id: str
    course_name: str
    description: str
    category: str
    instructor: str
    summary: SummaryModel
    cover_image: str
    status: str
    duration: str
    teachable_link: str
    transcript_path: str