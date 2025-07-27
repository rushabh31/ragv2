"""
Groq Vision AI Model with Universal Authentication Support.

This module provides a Groq-based vision model that integrates with the universal
authentication system and supports vision processing capabilities.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from groq import Groq

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GroqVisionAI:
    """Groq Vision AI model with universal authentication."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 max_pages: Optional[int] = None,
                 max_concurrent_pages: Optional[int] = None,
                 **kwargs):
        """Initialize Groq Vision AI model.
        
        Args:
            model_name: Name of the Groq vision model to use (overrides config)
            temperature: Temperature for generation (overrides config)
            max_tokens: Max tokens for generation (overrides config)
            top_p: Top-p for generation (overrides config)
            max_pages: Maximum pages to process (overrides config)
            max_concurrent_pages: Max concurrent pages (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        vision_config = config_manager.get_section("vision", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or vision_config.get("model", "llama-3.2-11b-vision-preview")
        self.temperature = temperature or vision_config.get("temperature", 0.2)
        self.max_tokens = max_tokens or vision_config.get("max_tokens", 1024)
        self.top_p = top_p or vision_config.get("top_p", 0.95)
        self.max_pages = max_pages or vision_config.get("max_pages", 100)
        self.max_concurrent_pages = max_concurrent_pages or vision_config.get("max_concurrent_pages", 5)
        
        # Store additional config parameters
        self.config = {**vision_config, **kwargs}
        
        # Initialize universal auth manager
        self.auth_manager = UniversalAuthManager()
        
        # Groq API configuration
        self.api_base = "https://api.groq.com/openai/v1"
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        logger.info(f"Initialized GroqVisionAI with model: {self.model_name} (from config)")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager.
        
        Returns:
            Authentication token or None if unavailable
        """
        try:
            return self.auth_manager.get_token("groq")
        except Exception as e:
            logger.error(f"Failed to get Groq token: {str(e)}")
            return None
    
    async def validate_authentication(self) -> bool:
        """Validate Groq authentication.
        
        Returns:
            True if authentication is valid, False otherwise
        """
        try:
            if not self.api_key:
                logger.error("Groq API key not available")
                return False
            
            # Test API call to validate authentication
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.get(
                    f"{self.api_base}/models",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info("Groq vision authentication validated successfully")
                    return True
                else:
                    logger.error(f"Groq vision authentication failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Groq vision authentication validation error: {str(e)}")
            return False
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get authentication health status.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "service": "groq_vision",
            "api_key_available": bool(self.api_key),
            "model_name": self.model_name,
            "status": "healthy" if self.api_key else "unhealthy"
        }
    
    async def parse_text_from_image(self, base64_image: str, prompt: str = None, **kwargs) -> str:
        """Parse text from image using Groq Vision API.
        
        Args:
            base64_image: Base64 encoded image data
            prompt: Optional prompt for text extraction
            **kwargs: Additional parameters
            
        Returns:
            Extracted text from the image
        """
        try:
            if not self.api_key:
                raise ValueError("Groq API key not available")
            
            # Default prompt for text extraction
            if not prompt:
                prompt = "Extract all text from this image. Return only the text content without any additional formatting or commentary."
            
            # Prepare image content
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ]
            
            # Merge generation parameters
            gen_params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p)
            }
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                **gen_params
            }
            
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0  # Longer timeout for vision processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Extracted text length: {len(content)}")
                    return content
                else:
                    error_msg = f"Groq Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Groq vision text extraction failed: {str(e)}")
            raise
    
    async def analyze_image(self, base64_image: str, prompt: str, **kwargs) -> str:
        """Analyze image using Groq Vision API.
        
        Args:
            base64_image: Base64 encoded image data
            prompt: Analysis prompt
            **kwargs: Additional parameters
            
        Returns:
            Analysis result as string
        """
        try:
            if not self.api_key:
                raise ValueError("Groq API key not available")
            
            # Prepare image content
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ]
            
            # Merge generation parameters
            gen_params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p)
            }
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                **gen_params
            }
            
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Image analysis length: {len(content)}")
                    return content
                else:
                    error_msg = f"Groq Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Groq vision image analysis failed: {str(e)}")
            raise
    
    async def extract_structured_data(self, base64_image: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Extract structured data from image using Groq Vision API.
        
        Args:
            base64_image: Base64 encoded image data
            schema: Schema definition for structured extraction
            **kwargs: Additional parameters
            
        Returns:
            Extracted structured data as dictionary
        """
        try:
            # Create prompt for structured extraction
            prompt = f"""
            Extract structured data from this image according to the following schema:
            {schema}
            
            Return the data as a JSON object that matches the schema exactly.
            """
            
            # Use analyze_image for structured extraction
            response = await self.analyze_image(base64_image, prompt, **kwargs)
            
            # Try to parse as JSON
            import json
            try:
                structured_data = json.loads(response)
                return structured_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse structured data as JSON, returning raw response")
                return {"raw_response": response}
                
        except Exception as e:
            logger.error(f"Groq vision structured data extraction failed: {str(e)}")
            raise
