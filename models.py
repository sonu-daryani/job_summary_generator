"""
Pydantic models for request and response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class BulletPointRequest(BaseModel):
    """Request model for bullet points input."""
    bullet_points: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=20,
        description="List of bullet points describing work experience or skills"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "bullet_points": [
                    "5 years of Python development experience",
                    "Led a team of 3 developers on a major project",
                    "Implemented microservices architecture",
                    "Experience with FastAPI and Django frameworks"
                ]
            }
        }

class SummaryResponse(BaseModel):
    """Response model for generated job summary."""
    summary: str = Field(..., description="Generated professional job summary")
    original_bullet_points: List[str] = Field(..., description="Original input bullet points")
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": "Experienced Python developer with 5 years of expertise in leading development teams and implementing scalable microservices architecture. Proven track record with FastAPI and Django frameworks, successfully managing projects and mentoring junior developers.",
                "original_bullet_points": [
                    "5 years of Python development experience",
                    "Led a team of 3 developers on a major project",
                    "Implemented microservices architecture",
                    "Experience with FastAPI and Django frameworks"
                ]
            }
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
