"""
AI service module for integrating with Hugging Face API.
Implements optimized text generation for job summaries.
"""
import asyncio
import aiohttp
import json
from typing import List, Optional
from config import settings
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    """Service class for AI-powered content generation."""
    
    def __init__(self):
        # Use a model that works well with free tier
        self.api_url = os.getenv("HUGGINGFACE_API_URL")
        self.headers = settings.get_huggingface_headers()
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Headers: {self.headers}")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _create_prompt(self, bullet_points: List[str]) -> str:
        """Create an optimized prompt for job summary generation."""
        bullet_text = " ".join(bullet_points)
        
        # DialoGPT works better with conversational prompts
        prompt = f"User: Create a professional job summary for someone with these skills: {bullet_text}\nBot:"
        
        return prompt
    
    async def generate_summary(self, bullet_points: List[str]) -> str:
        """
        Generate a professional job summary from bullet points.
        Uses Hugging Face API with optimized parameters.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            prompt = self._create_prompt(bullet_points)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,  # Optimized for concise summaries
                    "temperature": 0.8,     # Balanced creativity and consistency
                    "top_p": 0.9,          # Focused generation
                    "do_sample": True,
                    "return_full_text": False,
                    "repetition_penalty": 1.1
                }
            }
            
            logger.info(f"Generating summary for {len(bullet_points)} bullet points")
            
            async with self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        
                        # Clean up the generated text
                        summary = self._clean_generated_text(generated_text)
                        logger.info("Summary generated successfully")
                        return summary
                    else:
                        logger.error(f"Unexpected API response format: {result}")
                        return self._fallback_summary(bullet_points)
                
                elif response.status == 503:
                    # Model is loading, wait and retry
                    logger.warning("Model is loading, waiting 10 seconds...")
                    await asyncio.sleep(10)
                    return await self.generate_summary(bullet_points)
                
                elif response.status == 403:
                    # Permission denied - use fallback
                    logger.warning("API permission denied, using fallback summary")
                    return self._fallback_summary(bullet_points)
                
                elif response.status == 404:
                    # Model not found - try alternative model
                    logger.warning("Model not found, trying alternative approach")
                    return self._fallback_summary(bullet_points)
                
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    return self._fallback_summary(bullet_points)
        
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            return self._fallback_summary(bullet_points)
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self._fallback_summary(bullet_points)
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean and format the generated text."""
        # Remove any remaining prompt text
        if "This person is a" in text:
            text = text.split("This person is a")[-1]
        
        # Clean up whitespace and formatting
        text = text.strip()
        
        # Remove any incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure it ends with a period
        if text and not text.endswith(('.', '!', '?')):
            text += "."
        
        return text
    
    def _fallback_summary(self, bullet_points: List[str]) -> str:
        """Generate a fallback summary when AI service fails."""
        logger.info("Using fallback summary generation")
        
        # Enhanced template-based fallback
        experience_points = []
        skill_points = []
        leadership_points = []
        
        for point in bullet_points:
            point_lower = point.lower()
            if any(word in point_lower for word in ['led', 'managed', 'team', 'supervised', 'directed']):
                leadership_points.append(point)
            elif any(word in point_lower for word in ['years', 'experience', 'developed', 'built', 'created']):
                experience_points.append(point)
            else:
                skill_points.append(point)
        
        # Build professional summary
        summary_parts = []
        
        if experience_points:
            years_match = None
            for point in experience_points:
                import re
                years = re.search(r'(\d+)\s*years?', point.lower())
                if years:
                    years_match = years.group(1)
                    break
            
            if years_match:
                summary_parts.append(f"Experienced professional with {years_match} years of expertise")
            else:
                summary_parts.append("Experienced professional with proven track record")
        
        if leadership_points:
            summary_parts.append(f"Demonstrated leadership in {leadership_points[0].lower()}")
        
        if skill_points:
            skills_text = ', '.join(skill_points[:2])
            summary_parts.append(f"Strong technical skills in {skills_text}")
        
        if experience_points and not leadership_points:
            summary_parts.append(f"Successfully delivered {experience_points[0].lower()}")
        
        # Join parts and ensure proper ending
        summary = ". ".join(summary_parts) + "."
        
        # Clean up any double periods
        summary = summary.replace("..", ".")
        
        return summary

# Global AI service instance (singleton pattern for optimization)
_ai_service_instance = None

async def get_ai_service() -> AIService:
    """Get or create AI service instance (singleton pattern)."""
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIService()
    return _ai_service_instance
