import os
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.routers import (
    # file_handling,
    audio_processing,
    summarization,
    transcription,
    social_media,
    course,
    auth,
    schedule_router,
    schedule_post_router
)
from fastapi.middleware.cors import CORSMiddleware

from src.database.azureCosmosCourse import AzureCosmosNoSQLCourse
from .scheduler import scheduler_service
from .logger import logger

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the scheduler when the app starts
    scheduler_service.start()

    # Yield to allow the app to run
    yield

    # Stop the scheduler when the app is shutting down
    scheduler_service.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database_name = "AFAST_LLMS_DB"
container_name = "AFAST_Content"

courseClient = AzureCosmosNoSQLCourse(
    host=os.getenv("COSMOS_NOSQL_DATABASE_ENDPOINT"),
    key=os.getenv("COSMOS_NOSQL_DATABASE_KEY"),
    database_name=database_name,
    container_name=container_name,
)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello": "World"}

app.include_router(router=auth.router)
app.include_router(router=course.router)
app.include_router(router=summarization.router)
app.include_router(router=schedule_router.router)
app.include_router(router=schedule_post_router.router)
app.include_router(router=social_media.router)

app.include_router(router=audio_processing.router)
app.include_router(router=transcription.router)
# app.include_router(router=file_handling.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="127.0.0.1", reload=True, port=8000, timeout_keep_alive=600
    )
