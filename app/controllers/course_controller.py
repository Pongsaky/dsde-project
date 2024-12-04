from fastapi import Depends, status
from fastapi.responses import JSONResponse
from app.auth.auth import TokenData
from app.auth.dependencies import get_current_user
from app.services.course_service import (
    get_course_service, 
    get_all_courses_service, 
    upsert_course_service, 
    delete_course_service, 
    get_draft_courses_service, 
    get_scheduled_courses_service, 
    get_published_courses_service, 
    migration_course_service
)
from app.models.Course import CourseModel

async def get_course(courseID: str, _: TokenData = Depends(get_current_user)):
    try:
        course = await get_course_service(courseID)
        return JSONResponse(content=course, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "Course not found"}, status_code=status.HTTP_404_NOT_FOUND)

async def get_all_courses(_: TokenData = Depends(get_current_user)):
    try:
        courses = await get_all_courses_service()
        return JSONResponse(content=courses, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "No courses found"}, status_code=status.HTTP_404_NOT_FOUND)

async def upsert_course(course: CourseModel, _: TokenData = Depends(get_current_user)):
    try:
        result = await upsert_course_service(course)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "Course not found"}, status_code=status.HTTP_404_NOT_FOUND)

async def delete_course(courseID: str, _: TokenData = Depends(get_current_user)):
    try:
        isDeleted = await delete_course_service(courseID)
        if isDeleted:
            return JSONResponse(content={"message": "Course deleted"}, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "Course not found"}, status_code=status.HTTP_404_NOT_FOUND)

async def get_draft_courses(_: TokenData = Depends(get_current_user)):
    try:
        drafts = await get_draft_courses_service()
        return JSONResponse(content=drafts, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "No drafts found"}, status_code=status.HTTP_404_NOT_FOUND)

async def get_scheduled_courses(_: TokenData = Depends(get_current_user)):
    try:
        scheduled = await get_scheduled_courses_service()
        return JSONResponse(content=scheduled, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "No scheduled courses found"}, status_code=status.HTTP_404_NOT_FOUND)

async def get_published_courses(_: TokenData = Depends(get_current_user)):
    try:
        published = await get_published_courses_service()
        return JSONResponse(content=published, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"message": "No published courses found"}, status_code=status.HTTP_404_NOT_FOUND)

async def migration_course(_: TokenData = Depends(get_current_user)):
    try:
        result = await migration_course_service()
        if result["status"] == "success":
            return JSONResponse(content={"message": "Migration successful", "courseCount": result["courseCount"]}, status_code=status.HTTP_200_OK)
        return JSONResponse(content={"message": "Migration failed"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return JSONResponse(content={"message": "Migration failed"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)