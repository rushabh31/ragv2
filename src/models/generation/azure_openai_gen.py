"""
Azure OpenAI Generation Model.

This module provides a standardized interface for Azure OpenAI models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from openai import AzureOpenAI

from src.utils import UniversalAuthManager

logger = logging.getLogger(__name__)


class AzureOpenAIGenAI:
    """
    Azure OpenAI generation model with universal authentication.
    
    This class provides a standardized interface for Azure OpenAI models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: str = "GPT4-o",
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2023-05-15"):
        """
        Initialize Azure OpenAI client.
        
        Args:
            model_name: Name of the Azure OpenAI model to use
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version to use
        """
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self._client = None
        self._auth_manager = UniversalAuthManager(f"azure_openai_{model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized AzureOpenAIGenAI with model: {model_name}")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("azure_openai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_client(self):
        """Initialize Azure OpenAI client lazily."""
        if self._client is None:
            try:
                # Set SSL certificate path
                os.environ["SSL_CERT_FILE"] = "Path\\to\\ca-bundle.cer"
                # Ensure you have the correct path to your SSL certificate bundle
                
                # Get authentication token
                token = self.get_coin_token()
                if not token:
                    raise ValueError("Failed to get authentication token")
                
                # Initialize Azure OpenAI client
                self._client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token_provider=lambda: token,
                    api_version=self.api_version,
                )
                
                logger.info(f"Initialized Azure OpenAI client for {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
                raise
    
    async def generate_content(self,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.2,
                             stream: bool = False,
                             max_tokens: Optional[int] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Generate content using Azure OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        await self._init_client()
        
        try:
            # Make the API call
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                user=os.getenv("USERNAME", ""),
                temperature=temperature,
                stream=stream,
                max_tokens=max_tokens,
                **kwargs
            )
            
            logger.info(f"Generated content using {self.model_name}")
            return completion
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self,
                            prompt: str,
                            system_message: Optional[str] = None,
                            temperature: float = 0.2,
                            stream: bool = False,
                            max_tokens: Optional[int] = None,
                            **kwargs) -> str:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Temperature for generation
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.generate_content(
            messages=messages,
            temperature=temperature,
            stream=stream,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content if response.choices else ""
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()


# Example usage
async def example_usage():
    """Example of how to use AzureOpenAIGenAI."""
    
    # Initialize client
    client = AzureOpenAIGenAI()
    
    try:
        # Simple chat completion
        response = await client.chat_completion(
            prompt="Explain the difference between a bank and a credit union.",
            system_message="You are an expert on financial institutions.",
            temperature=0.2,
            stream=False
        )
        print(f"Response: {response}")
        
        # Advanced message format
        messages = [
            {"role": "system", "content": "You are an expert on financial institutions."},
            {"role": "user", "content": "Explain the difference between a bank and a credit union."}
        ]
        
        full_response = await client.generate_content(
            messages=messages,
            temperature=0.2,
            stream=False
        )
        print(f"Full Response: {full_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
