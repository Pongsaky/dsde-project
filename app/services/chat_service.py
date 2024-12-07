from app.logger import logger
from app.models.UserInput import UserInput
from app.models.APIResponse import APIResponse
from app.models.ClearMessage import ClearMessage
from src.chat import Chat

chat = Chat()

async def initial_chat_(user_input: UserInput) -> APIResponse:
    try:
        chat_response = chat.initial_chat(user_input)
        logger.info("Starting chat")
        return chat_response
    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        raise e

async def continue_chat_(user_input: UserInput) -> APIResponse:
    try:
        chat_response = chat.continue_chat(user_input)
        logger.info("Continuing chat")
        return chat_response
    except Exception as e:
        logger.error(f"Error continuing chat: {e}")
        raise e
    
async def clear_all_message_(clear_message : ClearMessage) -> APIResponse:
    try:
        chat.clear_all_message(clear_message.chat_id)
        logger.info("Clearing chat")
        return APIResponse(chat_id=clear_message.chat_id, message="Chat cleared", newGraph=None)
    except Exception as e:
        logger.error(f"Error clearing chat: {e}")
        raise e
    
async def get_history_chat_(clear_message: ClearMessage) -> APIResponse:
    try:
        chat_response = chat.get_chat_history(clear_message.chat_id)
        print(chat_response)
        logger.info("Getting history chat")
        # return chat_response
    except Exception as e:
        logger.error(f"Error getting history chat: {e}")
        raise e