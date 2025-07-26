"""
Vertex AI Generation Model.

This module provides a standardized interface for Google Vertex AI generative models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from vertexai.generative_models import GenerativeModel
import vertexai

from src.utils import UniversalAuthManager

logger = logging.getLogger(__name__)


class VertexGenAI:
    """
    Vertex AI generation model with universal authentication.
    
    This class provides a standardized interface for Google Vertex AI generative models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: str = "gemini-1.5-pro-002",
                 project_id: Optional[str] = None,
                 location: str = "us-central1"):
        """
        Initialize Vertex AI client.
        
        Args:
            model_name: Name of the Vertex AI model to use
            project_id: GCP project ID (will use env var if not provided)
            location: GCP location for Vertex AI
        """
        self.model_name = model_name
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        self.location = location
        self._model = None
        self._auth_manager = UniversalAuthManager(f"vertex_gen_{model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized VertexGenAI with model: {model_name}")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("vertex_ai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_model(self):
        """Initialize Vertex AI model lazily."""
        if self._model is None:
            try:
                # Get credentials using universal auth manager
                credentials = await self._auth_manager.get_credentials()
                
                # Initialize Vertex AI
                vertexai.init(
                    project=self.project_id,
                    api_transport="rest",  # uses UAT PROJECT
                    api_endpoint=os.environ.get("VERTEXAI_API_ENDPOINT"),  # uses R2D2 UAT
                    credentials=credentials,
                )
                
                # Set metadata for user tracking
                self.metadata = [("x-r2d2-user", os.getenv("USERNAME", ""))]
                
                # Initialize the generative model
                self._model = GenerativeModel(self.model_name)
                
                logger.info(f"Initialized Vertex AI model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI model: {str(e)}")
                raise
    
    async def generate_content(self,
                             prompt: str,
                             temperature: float = 0.7,
                             max_output_tokens: int = 2048,
                             top_p: float = 1.0,
                             top_k: int = 40,
                             **kwargs) -> Any:
        """
        Generate content using Vertex AI.
        
        Args:
            prompt: Text prompt for generation
            temperature: Temperature for generation
            max_output_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        await self._init_model()
        
        try:
            # Set generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k,
                **kwargs
            }
            
            # Generate content
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config=generation_config,
                metadata=self.metadata,
            )
            
            logger.info(f"Generated content using {self.model_name}")
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self,
                            prompt: str,
                            temperature: float = 0.7,
                            max_output_tokens: int = 2048,
                            **kwargs) -> str:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            temperature: Temperature for generation
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        response = await self.generate_content(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs
        )
        
        return response.text if hasattr(response, 'text') else str(response)
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()


# Example usage
async def example_usage():
    """Example of how to use VertexGenAI."""
    
    # Set environment variables for testing
    os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem files>"
    
    # Initialize client
    vertex_gen_ai = VertexGenAI()
    
    try:
        # Simple chat completion
        response = await vertex_gen_ai.chat_completion("Provide interesting trivia.")
        print(f"Response: {response}")
        
        # Advanced generation
        full_response = await vertex_gen_ai.generate_content(
            prompt="Provide interesting trivia.",
            temperature=0.8,
            max_output_tokens=1024
        )
        print(f"Full Response: {full_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
