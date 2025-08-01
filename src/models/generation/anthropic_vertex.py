"""
Anthropic Vertex AI Generation Model.

This module provides a standardized interface for Anthropic models running on Vertex AI
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from anthropic import AnthropicVertex
from google.oauth2.credentials import Credentials
import httpx

from src.utils import UniversalAuthManager

logger = logging.getLogger(__name__)


def log_response(response):
    """Log response with r2d2-request-id header."""
    print('r2d2-request-id: ', response.headers.get('x-r2d2-request-id', ''))


class AnthropicVertexGenAI:
    """
    Anthropic Vertex AI generation model with universal authentication.
    
    This class provides a standardized interface for Anthropic models running on
    Vertex AI platform using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: str = "claude-3-5-sonnet@20240229",
                 region: str = "us-east5",
                 project_id: Optional[str] = None):
        """
        Initialize Anthropic Vertex AI client.
        
        Args:
            model_name: Name of the Anthropic model to use
            region: GCP region for Vertex AI
            project_id: GCP project ID (will use env var if not provided)
        """
        self.model_name = model_name
        self.region = region
        # Use environment manager for configuration
        from src.utils.env_manager import env
        self.project_id = project_id or env.get_string("PROJECT_ID")
        self._client = None
        self._auth_manager = UniversalAuthManager(f"anthropic_vertex_{model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized AnthropicVertexGenAI with model: {model_name}")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("anthropic_vertex")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_client(self):
        """Initialize Anthropic Vertex client lazily."""
        if self._client is None:
            try:
                # Get credentials using universal auth manager
                credentials = await self._auth_manager.get_credentials()
                
                # Create HTTP client with response logging
                http_client = httpx.Client(event_hooks={'response': [log_response]})
                
                # Initialize Anthropic Vertex client
                self._client = AnthropicVertex(
                    region=self.region,
                    project_id=self.project_id,
                    credentials=credentials,
                    http_client=http_client,
                    base_url="https://url"  # Will be overridden by region/project
                )
                
                logger.info(f"Initialized Anthropic Vertex client for {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic Vertex client: {str(e)}")
                raise
    
    async def generate_content(self,
                             messages: List[Dict[str, str]],
                             max_tokens: int = 4000,
                             temperature: float = 0.7,
                             **kwargs) -> Dict[str, Any]:
        """
        Generate content using Anthropic Vertex AI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        await self._init_client()
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make the API call
            response = await asyncio.to_thread(
                self._client.messages.create,
                extra_headers={"x-r2d2-user": os.getenv("USERNAME", "")},
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=anthropic_messages,
                **kwargs
            )
            
            logger.info(f"Generated content using {self.model_name}")
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self,
                            prompt: str,
                            max_tokens: int = 4000,
                            temperature: float = 0.7,
                            **kwargs) -> str:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.generate_content(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.content[0].text if response.content else ""
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()


# Example usage
async def example_usage():
    """Example of how to use AnthropicVertexGenAI."""
    
    # Initialize client
    client = AnthropicVertexGenAI()
    
    try:
        # Simple chat completion
        response = await client.chat_completion("Send me a recipe for banana bread.")
        print(f"Response: {response}")
        
        # Advanced message format
        messages = [
            {"role": "user", "content": "Send me a recipe for banana bread."}
        ]
        
        full_response = await client.generate_content(messages)
        print(f"Full Response: {full_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["SSL_CERT_FILE"] = "<PATH TO PROD ca.pem>"
    
    # Run example
    asyncio.run(example_usage())
