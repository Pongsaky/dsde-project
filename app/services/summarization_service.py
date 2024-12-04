from app.models.Summarization import SummarizationQueryModel, RegenerateModel, SummarizationCourseIdModel

from src.database.course import Course
from src.summarization.summarization import LLMSummarization
from src.database.azureCosmosVector import AzureCosmosNoSQLVectorDataBase
from src.database.azureCosmosCourse import AzureCosmosNoSQLCourse
from src.summarization.openaiEmbedding import OpenAIEmbedding

from src.utils import normalize_text_regenerate, normalize_json_string, list_string_to_list
import json
import os

model_name = "gpt-4o-mini-2"
HOST = os.getenv("COSMOS_NOSQL_DATABASE_ENDPOINT")
KEY = os.getenv('COSMOS_NOSQL_DATABASE_KEY')
database_name = "AFAST_LLMS_DB"
collection_name = "AFAST_Vector_Container"

llmSummarization = LLMSummarization(
        model=model_name,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        isAzure=True,
    )

openai_embedding = OpenAIEmbedding(
    host=os.getenv("AZURE_OPENAI_ENDPOINT"), key=os.getenv("AZURE_OPENAI_API_KEY")
)

azureVectorClient = AzureCosmosNoSQLVectorDataBase(
    host=HOST,
    key=KEY,
    database_name=database_name,
)

azureCourseClient = AzureCosmosNoSQLCourse(
    host=HOST,
    key=KEY,
    database_name=database_name,
    container_name="AFAST_Content"
)

async def process_query_summarization(query: SummarizationQueryModel):
    try:
        embedding = openai_embedding.get_embedding(query.text)
        search_results = azureVectorClient.get_search_results(
            collection_name=collection_name, embedding=embedding, top_k=query.top_k
        )

        text, instructor, link2course = llmSummarization.concatenate_results(search_results)
        
        res = llmSummarization.summarize_root_transciption(
            text=text, instructor=instructor, link2course=link2course
        )
        
        normalized_res = normalize_json_string(res)
        normalized_res_json = json.loads(normalized_res)

        return normalized_res_json
    except Exception as e:
        raise RuntimeError(f"An error occurred during summarization: {str(e)}")

async def process_summarization(query: SummarizationCourseIdModel, isUpsert: bool = True):
    try:
        metadata_filter = {
            "course_id": query.course_id,
        }

        search_results = azureVectorClient.get_filter_by_metadata(
            collection_name=collection_name, filter=metadata_filter, top_k=query.top_k
        )

        text, instructor, link2course = llmSummarization.concatenate_results(search_results)
        
        res = llmSummarization.summarize_root_transciption(
            text=text, instructor=instructor, link2course=link2course
        )
        
        normalized_res = normalize_json_string(res)
        normalized_res_json = json.loads(normalized_res)

        if isUpsert:
            await process_migration_summarization(query, normalized_res_json)
        
        return normalized_res_json
    except Exception as e:
        raise RuntimeError(f"An error occurred during summarization: {str(e)}")
    
async def process_migration_summarization(query: SummarizationCourseIdModel, normalized_res_json: dict):
    try:
        course = azureCourseClient.getCourse(query.course_id)
        if course == {}:
            raise RuntimeError(f"Course with id {query.course_id} not found")
        if course["summary"]["attentionGrabber"] != "":
            return
        
        course.update({"summary": normalized_res_json})
        course = Course.from_dict(course)
        
        azureCourseClient.upsertCourse(course=course)

    except Exception as e:
        raise RuntimeError(f"An error occurred during migration: {str(e)}")

async def process_regeneration(query: RegenerateModel):
    try:
        metadata_filter = {
            "course_id": query.course_id,
        }

        search_results = azureVectorClient.get_filter_by_metadata(
            collection_name=collection_name, filter=metadata_filter, top_k=query.top_k
        )

        text, instructor, link2course = llmSummarization.concatenate_results(search_results)

        res = llmSummarization.regenerate_part(
            part_name=query.part, text=text, instructor=instructor, link2course=link2course
        )

        normalized_res = normalize_text_regenerate(res)
        normalized_res = normalized_res.replace("\n", "").replace("{", "").replace("}", "").replace("\"", "")
        if "[" in normalized_res:
            normalized_res = list_string_to_list(normalized_res)
        return {query.part: normalized_res}
    except Exception as e:
        raise RuntimeError(f"An error occurred during regeneration: {str(e)}")