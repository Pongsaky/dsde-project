from fastapi.responses import JSONResponse
from app.models.Summarization import (
    SummarizationQueryModel,
    RegenerateModel,
    SummarizationCourseIdModel,
)
from app.services.summarization_service import (
    process_summarization,
    process_regeneration,
    process_query_summarization
)

async def chat_response(query: SummarizationQueryModel):
    try:
        result = await process_summarization(query, is_frontend=True)
        return JSONResponse(
            status_code=200,
            content={"content": result},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"},
        )

async def summarize_by_query(query: SummarizationQueryModel):
    try:
        result = await process_query_summarization(query)
        return JSONResponse(
            status_code=200,
            content={"content": result},
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"},
        )

async def summarize_by_filter_course_id(query: SummarizationCourseIdModel):
    try:
        result = await process_summarization(query, isUpsert=query.is_upsert)
        return JSONResponse(
            status_code=200,
            content={"content": result}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"},
        )

async def regenerate_summarization(query: RegenerateModel):
    try:
        result = await process_regeneration(query)
        
        return JSONResponse(
            status_code=200,
            content={"content": result},
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"},
        )
