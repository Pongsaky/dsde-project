from app.models.Chat import ChatModel
from src.database.qdrant import QdrantVectorDB
from app.logger import logger
from typing import List

url = ""
api_key = ""

chat_client = QdrantVectorDB(url, api_key)

async def start_chat_service(chat: ChatModel) -> List[ChatModel]:
    try:
        # Get the chat from the model
        chat_response = None
        # chat_response = await ...
        logger.info("Starting chat")
        return chat_response
    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        raise e

async def continue_chat_service(chats: List[ChatModel]) -> List[ChatModel]:
    try:
        # Get the chat from the model use chat list
        chat_response = None
        # chat_response = awiat ...
        logger.info("Continuing chat")
        return chat_response
    except Exception as e:
        logger.error(f"Error continuing chat: {e}")
        raise e

# No need for other methods
# There is chat room collecting in backend. Because we only using list of chat(previous chat) from frontend
