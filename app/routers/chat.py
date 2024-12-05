from fastapi import APIRouter
from app.controllers.chat_controller import start_chat, continue_chat

router = APIRouter(prefix="/api", tags=["api"])

# Initial chat: Use for starting a new chat
router.add_api_route(path="/initial_chat", endpoint=start_chat, methods=["POST"])

# Continue chat: Use for continuing an existing chat
router.add_api_route(path="/chat", endpoint=continue_chat, methods=["POST"])
