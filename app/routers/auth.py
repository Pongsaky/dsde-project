from fastapi import APIRouter
from app.controllers.auth import (
    login,
    get_me,
)

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
)

router.add_api_route(
    path="/login", endpoint=login, methods=["POST"]
)

router.add_api_route(
    path="/get_me", endpoint=get_me, methods=["GET"]
)