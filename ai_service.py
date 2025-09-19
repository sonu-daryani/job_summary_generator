"""
AI service module for integrating with Llama 4 Scout, Llama 2, and Hugging Face API.
Implements optimized text generation for job summaries using local Llama models.
"""
import asyncio
import aiohttp
import json
import torch
from typing import List, Optional
from config import settings
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    """Service class for AI-powered content generation using Llama 4 Scout, Llama 2, or Hugging Face API."""
    
    def __init__(self):
        self.use_local_model = settings.use_local_model
        self.use_llama4 = settings.use_llama4
        self.llama4_client = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Hugging Face API fallback settings
        self.api_url = os.getenv("HUGGINGFACE_API_URL")
        self.headers = settings.get_huggingface_headers()
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Using {'Llama 4 Scout' if self.use_llama4 else 'Llama 2' if self.use_local_model else 'Hugging Face API'}")
        
        if self.use_local_model:
            if self.use_llama4:
                self._initialize_llama4()
            else:
                self._initialize_llama2()
    
    def _initialize_llama4(self):
        """Initialize the Llama 4 Scout model using llama-stack."""
        try:
            from llama import Llama
            
            logger.info("Initializing Llama 4 Scout...")
            
            # Get Llama 4 configuration
            llama4_config = settings.get_llama4_config()
            
            # Initialize Llama 4 client
            self.llama4_client = Llama(
                model_id=llama4_config["model_id"],
                custom_url=llama4_config["custom_url"]
            )
            
            logger.info("Llama 4 Scout initialized successfully")
            
        except ImportError as e:
            logger.error(f"llama-stack not installed: {e}")
            logger.info("Falling back to Llama 2")
            self.use_llama4 = False
            self._initialize_llama2()
        except Exception as e:
            logger.error(f"Failed to initialize Llama 4 Scout: {e}")
            logger.info("Falling back to Llama 2")
            self.use_llama4 = False
            self._initialize_llama2()
    
    def _initialize_llama2(self):
        """Initialize the Llama 2 model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            logger.info("Loading Llama 2 model...")
            
            # Get model configuration
            llama_config = settings.get_llama_config()
            model_name = llama_config["model_name"]
            device = llama_config["device"]
            
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("Using CUDA for Llama 2")
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("Using MPS for Llama 2")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for Llama 2")
            else:
                self.device = device
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.device == "cuda" and "max_memory" in llama_config:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Llama 2 model loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.info("Falling back to Hugging Face API")
            self.use_local_model = False
        except Exception as e:
            logger.error(f"Failed to load Llama 2 model: {e}")
            logger.info("Falling back to Hugging Face API")
            self.use_local_model = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.use_local_model:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _create_llama4_prompt(self, bullet_points: List[str]) -> str:
        """Create an optimized prompt for Llama 4 Scout."""
        bullet_text = " ".join(bullet_points)
        
        prompt = f"""Create a professional job summary for someone with these skills and experiences: {bullet_text}

The summary should be 2-3 sentences and highlight the person's key skills and experience in a professional manner."""
        
        return prompt
    
    def _create_llama2_prompt(self, bullet_points: List[str]) -> str:
        """Create an optimized prompt for Llama 2."""
        bullet_text = " ".join(bullet_points)
        
        # Llama 2 chat format
        prompt = f"""<s>[INST] <<SYS>>
You are a professional resume writer. Create a concise, professional job summary based on the provided bullet points. The summary should be 2-3 sentences and highlight the person's key skills and experience.
<</SYS>>

Create a professional job summary for someone with these skills and experiences: {bullet_text} [/INST]"""
        
        return prompt
    
    def _create_hf_prompt(self, bullet_points: List[str]) -> str:
        """Create a prompt for Hugging Face API fallback."""
        bullet_text = " ".join(bullet_points)
        prompt = f"User: Create a professional job summary for someone with these skills: {bullet_text}\nBot:"
        return prompt
    
    async def generate_summary(self, bullet_points: List[str]) -> str:
        """
        Generate a professional job summary from bullet points.
        Uses Llama 4 Scout, Llama 2, or Hugging Face API in order of preference.
        """
        if self.use_local_model:
            if self.use_llama4 and self.llama4_client:
                return await self._generate_with_llama4(bullet_points)
            elif self.model is not None:
                return await self._generate_with_llama2(bullet_points)
        
        return await self._generate_with_hf_api(bullet_points)
    
    async def _generate_with_llama4(self, bullet_points: List[str]) -> str:
        """Generate summary using Llama 4 Scout."""
        try:
            prompt = self._create_llama4_prompt(bullet_points)
            
            logger.info(f"Generating summary with Llama 4 Scout for {len(bullet_points)} bullet points")
            
            # Use Llama 4 Scout to generate response
            response = self.llama4_client.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract generated text
            generated_text = response.get("text", "")
            
            # Clean up the generated text
            summary = self._clean_generated_text(generated_text)
            logger.info("Summary generated successfully with Llama 4 Scout")
            return summary
            
        except Exception as e:
            logger.error(f"Llama 4 Scout generation failed: {e}")
            logger.info("Falling back to Llama 2")
            return await self._generate_with_llama2(bullet_points)
    
    async def _generate_with_llama2(self, bullet_points: List[str]) -> str:
        """Generate summary using local Llama 2 model."""
        try:
            prompt = self._create_llama2_prompt(bullet_points)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            logger.info(f"Generating summary with Llama 2 for {len(bullet_points)} bullet points")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean up the generated text
            summary = self._clean_generated_text(generated_text)
            logger.info("Summary generated successfully with Llama 2")
            return summary
            
        except Exception as e:
            logger.error(f"Llama 2 generation failed: {e}")
            logger.info("Falling back to Hugging Face API")
            return await self._generate_with_hf_api(bullet_points)
    
    async def _generate_with_hf_api(self, bullet_points: List[str]) -> str:
        """Generate summary using Hugging Face API (fallback)."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            prompt = self._create_hf_prompt(bullet_points)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False,
                    "repetition_penalty": 1.1
                }
            }
            
            logger.info(f"Generating summary with Hugging Face API for {len(bullet_points)} bullet points")
            
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
                        summary = self._clean_generated_text(generated_text)
                        logger.info("Summary generated successfully with Hugging Face API")
                        return summary
                    else:
                        logger.error(f"Unexpected API response format: {result}")
                        return self._fallback_summary(bullet_points)
                
                elif response.status == 503:
                    logger.warning("Model is loading, waiting 10 seconds...")
                    await asyncio.sleep(10)
                    return await self._generate_with_hf_api(bullet_points)
                
                elif response.status in [403, 404]:
                    logger.warning("API access denied or model not found, using fallback summary")
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
        # Remove any remaining prompt text or special tokens
        text = text.strip()
        
        # Remove common prefixes that might appear
        prefixes_to_remove = [
            "This person is a",
            "The candidate is",
            "This individual is",
            "This professional is",
            "This person has",
            "The person has",
            "This individual has",
            "This professional has"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Clean up whitespace and formatting
        text = text.strip()
        
        # Remove any incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure it ends with a period
        if text and not text.endswith(('.', '!', '?')):
            text += "."
        
        # Remove any remaining special tokens or artifacts
        text = text.replace('<s>', '').replace('</s>', '').replace('[INST]', '').replace('[/INST]', '')
        text = text.replace('<<SYS>>', '').replace('<</SYS>>', '')
        
        return text.strip()
    
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