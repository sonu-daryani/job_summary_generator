"""
FastAPI routes for the Job Summary Generator.
Implements RESTful API endpoints with proper error handling.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import List
import logging

from models import BulletPointRequest, SummaryResponse, ErrorResponse
from ai_service import get_ai_service, AIService

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["job-summary"])

@router.post(
    "/generate-summary",
    response_model=SummaryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def generate_job_summary(request: BulletPointRequest):
    """
    Generate a professional job summary from bullet points.
    
    Args:
        request: BulletPointRequest containing list of bullet points
        
    Returns:
        SummaryResponse with generated summary and original bullet points
        
    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        logger.info(f"Received request with {len(request.bullet_points)} bullet points")
        
        # Validate input
        if not request.bullet_points:
            raise HTTPException(
                status_code=400,
                detail="At least one bullet point is required"
            )
        
        # Get AI service and generate summary
        ai_service = await get_ai_service()
        summary = await ai_service.generate_summary(request.bullet_points)
        
        logger.info("Summary generated successfully")
        
        return SummaryResponse(
            summary=summary,
            original_bullet_points=request.bullet_points
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_job_summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while generating summary"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "job-summary-generator"}

@router.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        logger.error("Frontend template not found")
        raise HTTPException(status_code=404, detail="Frontend not available")
