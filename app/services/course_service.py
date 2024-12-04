import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.models.Course import CourseModel
from app.models.Summarization import SummarizationCourseIdModel
from src.database.azureCosmosCourse import AzureCosmosNoSQLCourse
from app.services.summarization_service import process_summarization
from app.logger import logger
from src.database.course import Course
import os

executor = ThreadPoolExecutor()

database_name = "AFAST_LLMS_DB"
container_name = "AFAST_Content"

courseClient = AzureCosmosNoSQLCourse(
    host=os.getenv("COSMOS_NOSQL_DATABASE_ENDPOINT"),
    key=os.getenv("COSMOS_NOSQL_DATABASE_KEY"),
    database_name=database_name,
    container_name=container_name,
)

async def get_course_service(courseID: str):
    try:
        course = courseClient.getCourse(courseID)
        logger.info(f"Getting course with ID: {courseID}")
        return course
    except Exception as e:
        logger.error(f"Error getting course: {e}")
        raise e

async def get_all_courses_service():
    try:
        courses = courseClient.getAllCourses()
        logger.info("Getting all courses")
        return courses
    except Exception as e:
        logger.error(f"Error getting all courses: {e}")
        raise e

async def upsert_course_service(course: CourseModel):
    try:
        course_dict = course.model_dump()
        course_obj = Course.from_dict(course_dict)
        return courseClient.upsertCourse(course_obj)
    except Exception as e:
        logger.error(f"Error upserting course: {e}")
        raise e

async def delete_course_service(courseID: str):
    try:
        isDeleted = courseClient.deleteCourse(courseID)
        if not isDeleted:
            raise Exception("Course not found")
        logger.info(f"Deleting course with ID: {courseID}")
        return isDeleted
    except Exception as e:
        logger.error(f"Error deleting course: {e}")
        raise e

async def get_draft_courses_service():
    return courseClient.get_course_by_status("draft")

async def get_scheduled_courses_service():
    return courseClient.get_course_by_status("scheduled")

async def get_published_courses_service():
    return courseClient.get_course_by_status("published")

async def migration_course_service():
    try:
        res = courseClient.migration_course()
        if res["status"] == "success":
            loop = asyncio.get_event_loop()
            loop.run_in_executor(executor, migration_summarization_process)
        logger.info("Migration course completed")
        return res
    except Exception as e:
        logger.error(f"Error in migration course: {e}")
        raise e

def migration_summarization_process():
    try:
        courses = courseClient.getAllCourses()
        for course in courses:
            if course["summary"]["attentionGrabber"] != "":
                continue
            summarizationCourse = SummarizationCourseIdModel(
                course_id=course["course_id"],
                top_k=10,
                is_upsert=True
            )
            asyncio.run(process_summarization(summarizationCourse, isUpsert=True))
            logger.info(f"Migration summarization for course {course['id']} completed")
    except Exception as e:
        logger.error(f"Error in migration summarization process: {e}")