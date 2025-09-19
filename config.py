"""
Configuration module for the Job Summary Generator application.
Handles environment variables and application settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        self.app_name: str = os.getenv("APP_NAME", "Job Summary Generator")
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # Hugging Face API settings
        self.huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
        self.huggingface_api_url: str = os.getenv(
            "HUGGINGFACE_API_URL", 
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        )
        self.model_index: int = int(os.getenv("MODEL_INDEX", "0"))
        
        # Validate required settings
        if not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required. Please set it in your .env file.")
    
    def get_huggingface_headers(self) -> dict:
        """Get headers for Hugging Face API requests."""
        return {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json"
        }

# Global settings instance
settings = Settings()
