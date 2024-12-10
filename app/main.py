import os
from fastapi import FastAPI
from dotenv import load_dotenv
from app.routers import (
    chat_router
)
from fastapi.middleware.cors import CORSMiddleware
from .logger import logger

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello": "World"}

app.include_router(router=chat_router.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="127.0.0.1", reload=True, port=8000, timeout_keep_alive=600
    )
