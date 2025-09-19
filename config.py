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
        
        # AI Model settings - now supports Hugging Face API, local Llama 2, and Llama 4 Scout
        self.use_local_model: bool = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"
        self.use_llama4: bool = os.getenv("USE_LLAMA4", "true").lower() == "true"
        
        # Hugging Face API settings (fallback)
        self.huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
        self.huggingface_api_url: str = os.getenv(
            "HUGGINGFACE_API_URL", 
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        )
        
        # Llama 4 Scout settings
        self.llama4_model_id: str = os.getenv("LLAMA4_MODEL_ID", "llama-4-scout")
        self.llama4_custom_url: str = os.getenv(
            "LLAMA4_CUSTOM_URL",
            "https://llama4.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMDYzejlsa2hkZmJoOHJ6cmU5M2xmY212IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWE0LmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg0NDkxMzN9fX1dfQ__&Signature=dy02PS3FpS3OFM2y7mbdyHp5eHxvLDj5hBVMyLthoJUPqB2otTe5Pi5PduKNfXEFmecNYFwhFodvokciqye8T2TRuwhyJ5q0RVI4lcRgmUaeUkKvlfNRWrbDyHeTyFTW-COEEogR1F7VfSuQ0JDoeyBCybZ6KTzeTTjLpkmp61Tsr%7EYGfQUHcsQngGjoE9Lvp%7EYC4opCkLhx%7E3I4qwn8G3Ynbj1vF8b%7E1W%7EKFQvfYqiyKYopAR4hmgvH3VmVGkO9Vmhqg%7EnYhN1uNQwXX4dSuOxvKCpOUChM5Qxu3Rxmby9XpAY%7E7FUlkefSKUINKeJXy3GJH4-IKI1DC9IJSZn6Yg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2033659194105612"
        )
        
        # Llama 2 model settings (fallback)
        self.llama_model_name: str = os.getenv(
            "LLAMA_MODEL_NAME", 
            "meta-llama/Llama-2-7b-chat-hf"
        )
        self.device: str = os.getenv("DEVICE", "auto")  # auto, cpu, cuda, mps
        self.max_memory: Optional[str] = os.getenv("MAX_MEMORY")  # e.g., "8GB" for GPU memory limit
        self.model_index: int = int(os.getenv("MODEL_INDEX", "0"))
        
        # Validate settings based on model choice
        if not self.use_local_model and not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required when USE_LOCAL_MODEL=false. Please set it in your .env file.")
    
    def get_huggingface_headers(self) -> dict:
        """Get headers for Hugging Face API requests."""
        return {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json"
        }
    
    def get_llama4_config(self) -> dict:
        """Get configuration for Llama 4 Scout model."""
        return {
            "model_id": self.llama4_model_id,
            "custom_url": self.llama4_custom_url,
        }
    
    def get_llama_config(self) -> dict:
        """Get configuration for Llama 2 model."""
        config = {
            "model_name": self.llama_model_name,
            "device": self.device,
        }
        
        if self.max_memory:
            config["max_memory"] = self.max_memory
            
        return config

# Global settings instance
settings = Settings()
