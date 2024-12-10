from fastapi import APIRouter
from app.controllers.chat_controller import initial_chat, continue_chat, clear_all_message, get_history_chat

router = APIRouter(prefix="/api", tags=["api"])

# Initial chat: Use for starting a new chat
router.add_api_route(path="/initial_chat", endpoint=initial_chat, methods=["POST"])

# Continue chat: Use for continuing an existing chat
router.add_api_route(path="/continue_chat", endpoint=continue_chat, methods=["POST"])

# Clear all messages: Use for clearing all messages in a chat
router.add_api_route(path="/clear_all_message", endpoint=clear_all_message, methods=["POST"])

# Get history chat: Use for getting history chat
router.add_api_route(path="/get_history_chat", endpoint=get_history_chat, methods=["POST"])
