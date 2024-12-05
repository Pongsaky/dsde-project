from fastapi import status
from fastapi.responses import JSONResponse
from app.models.Chat import ChatModel
from typing import List
from app.services.chat_service import start_chat_service, continue_chat_service

async def start_chat(chat: ChatModel):
    try:
        chat_response = await start_chat_service(chat)
        return JSONResponse(content=chat_response, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e

async def continue_chat(chats: List[ChatModel]):
    try:
        chat_response = await continue_chat_service(chats)
        return JSONResponse(content=chat_response, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e
