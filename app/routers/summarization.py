from fastapi import APIRouter
from app.controllers.summarization import chat_response, summarize_by_query, summarize_by_filter_course_id, regenerate_summarization

router = APIRouter(
    prefix="/api/summarization",
    tags=["summarization"],
)

router.add_api_route(path="/chat_response", endpoint=chat_response, methods=["POST"])
router.add_api_route(path="/summarize_by_query", endpoint=summarize_by_query, methods=["POST"])
router.add_api_route(path="/summarize_by_filter_course_id", endpoint=summarize_by_filter_course_id, methods=["POST"])
router.add_api_route(path="/regenerate_summarization", endpoint=regenerate_summarization, methods=["POST"])