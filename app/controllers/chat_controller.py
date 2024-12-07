from fastapi import status
from fastapi.responses import JSONResponse
from app.services.chat_service import initial_chat_, continue_chat_, clear_all_message_, get_history_chat_
from app.models.UserInput import UserInput
from app.models.ClearMessage import ClearMessage

async def initial_chat(user_input: UserInput):
    try:
        chat_response = await initial_chat_(user_input=user_input)
        return JSONResponse(content=chat_response.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e

async def continue_chat(user_input: UserInput):
    try:
        chat_response = await continue_chat_(user_input=user_input)
        return JSONResponse(content=chat_response.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e
    
async def clear_all_message(clear_message: ClearMessage):
    try:
        chat_response = await clear_all_message_(clear_message=clear_message)
        return JSONResponse(content=chat_response.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e
    
async def get_history_chat(clear_message: ClearMessage):
    try:
        await get_history_chat_(clear_message=clear_message)
        return JSONResponse(content={"message" : "OK"}, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise e