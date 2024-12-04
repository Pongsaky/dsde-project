from fastapi import APIRouter
from app.controllers.course_controller import (
    get_all_courses,
    get_course,
    get_draft_courses,
    get_published_courses,
    get_scheduled_courses,
    migration_course,
    upsert_course,
    delete_course,
)

router = APIRouter(
    prefix="/api/course",
    tags=["course"],
)

router.add_api_route(
    path="/get_course", endpoint=get_all_courses, methods=["GET"]
)

router.add_api_route(
    path="/get_course/{courseID}", endpoint=get_course, methods=["GET"]
)

router.add_api_route(
    path="/get_draft_course", endpoint=get_draft_courses, methods=["GET"]
)

router.add_api_route(
    path="/get_published_course", endpoint=get_published_courses, methods=["GET"]
)

router.add_api_route(
    path="/get_scheduled_course", endpoint=get_scheduled_courses, methods=["GET"]
)

router.add_api_route(
    path="/upsert_course", endpoint=upsert_course, methods=["POST"]
)

router.add_api_route(
    path="/migration_course", endpoint=migration_course, methods=["POST"]
)

router.add_api_route(
    path="/delete_course/{courseID}", endpoint=delete_course, methods=["DELETE"]
)