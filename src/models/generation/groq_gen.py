"""
Groq Generation AI Model with Universal Authentication Support.

This module provides a Groq-based generation model that integrates with the universal
authentication system and supports the standard generation model interface.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from groq import Groq

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GroqGenAI:
    """Groq Generation AI model with universal authentication."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs):
        """Initialize Groq Generation AI model.
        
        Args:
            model_name: Name of the Groq model to use (overrides config)
            temperature: Temperature for generation (overrides config)
            max_tokens: Max tokens for generation (overrides config)
            top_p: Top-p for generation (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        generation_config = config_manager.get_section("generation", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or generation_config.get("model", "llama-3.1-70b-versatile")
        self.temperature = temperature or generation_config.get("temperature", 0.2)
        self.max_tokens = max_tokens or generation_config.get("max_tokens", 1024)
        self.top_p = top_p or generation_config.get("top_p", 0.95)
        
        # Store additional config parameters
        self.config = {**generation_config, **kwargs}
        
        # Initialize universal auth manager
        self.auth_manager = UniversalAuthManager()
        
        # Groq API configuration
        self.api_base = "https://api.groq.com/openai/v1"
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        logger.info(f"Initialized GroqGenAI with model: {self.model_name} (from config)")
    
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
                    logger.info("Groq authentication validated successfully")
                    return True
                else:
                    logger.error(f"Groq authentication failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Groq authentication validation error: {str(e)}")
            return False
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get authentication health status.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "service": "groq",
            "api_key_available": bool(self.api_key),
            "model_name": self.model_name,
            "status": "healthy" if self.api_key else "unhealthy"
        }
    
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using Groq API.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content as string
        """
        try:
            if not self.api_key:
                raise ValueError("Groq API key not available")
            
            # Merge generation parameters
            gen_params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p)
            }
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
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
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Generated content length: {len(content)}")
                    return content
                else:
                    error_msg = f"Groq API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Groq content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate chat completion using Groq API.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response as string
        """
        try:
            if not self.api_key:
                raise ValueError("Groq API key not available")
            
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                if "parts" in msg:
                    # Handle Vertex AI format
                    content = " ".join([part.get("text", "") for part in msg["parts"]])
                    groq_messages.append({
                        "role": msg["role"],
                        "content": content
                    })
                else:
                    # Handle standard format
                    groq_messages.append({
                        "role": msg["role"],
                        "content": msg.get("content", "")
                    })
            
            # Merge generation parameters
            gen_params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p)
            }
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": groq_messages,
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
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Chat completion length: {len(content)}")
                    return content
                else:
                    error_msg = f"Groq API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Groq chat completion failed: {str(e)}")
            raise
