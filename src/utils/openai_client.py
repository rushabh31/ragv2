"""
OpenAI Client with Universal Authentication.

This module demonstrates how to use the universal authentication system
for OpenAI API calls, providing a consistent authentication approach
across all services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import httpx

from .auth_manager import UniversalAuthManager

logger = logging.getLogger(__name__)


class UniversalOpenAIClient:
    """
    OpenAI client using universal authentication system.
    
    This class demonstrates how to integrate the universal auth manager
    with OpenAI API calls, providing consistent token management across
    all services in the system.
    """
    
    def __init__(self, 
                 base_url: str = "https://api.openai.com/v1",
                 service_name: str = "openai"):
        """
        Initialize OpenAI client with universal authentication.
        
        Args:
            base_url: OpenAI API base URL
            service_name: Service name for auth manager
        """
        self.base_url = base_url
        self.service_name = service_name
        self._auth_manager = UniversalAuthManager(service_name)
        self._auth_manager.configure()
        self._client = None
        
        logger.info(f"Initialized UniversalOpenAIClient for {service_name}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with authentication headers."""
        if self._client is None:
            headers = await self._auth_manager.get_openai_headers()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0
            )
        return self._client
    
    async def chat_completion(self,
                             messages: List[Dict[str, str]],
                             model: str = "gpt-3.5-turbo",
                             **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using OpenAI API with universal authentication.
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            **kwargs: Additional parameters for the API call
            
        Returns:
            API response dictionary
        """
        try:
            client = await self._get_client()
            
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    async def embeddings(self,
                        input_text: Union[str, List[str]],
                        model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """
        Create embeddings using OpenAI API with universal authentication.
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model name
            
        Returns:
            API response dictionary
        """
        try:
            client = await self._get_client()
            
            payload = {
                "model": model,
                "input": input_text
            }
            
            response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Embeddings generation failed: {str(e)}")
            raise
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()
    
    async def close(self):
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        self._auth_manager.close()
        logger.info(f"Closed UniversalOpenAIClient for {self.service_name}")


# Example usage function
async def example_usage():
    """Example of how to use the UniversalOpenAIClient."""
    
    # Initialize client
    client = UniversalOpenAIClient(service_name="openai_example")
    
    try:
        # Check authentication health
        health_status = client.get_auth_health_status()
        print(f"Auth Health Status: {health_status}")
        
        # Validate authentication
        is_valid = await client.validate_authentication()
        print(f"Authentication Valid: {is_valid}")
        
        if is_valid:
            # Example chat completion
            messages = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            
            response = await client.chat_completion(messages)
            print(f"Chat Response: {response}")
            
            # Example embeddings
            embedding_response = await client.embeddings("Hello world")
            print(f"Embedding Response: {embedding_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        await client.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
