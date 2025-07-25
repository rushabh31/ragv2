"""
Azure OpenAI Embedding Model.

This module provides a standardized interface for Azure OpenAI embedding models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from openai import AzureOpenAI

from src.utils import UniversalAuthManager

logger = logging.getLogger(__name__)


class AzureOpenAIEmbeddingAI:
    """
    Azure OpenAI embedding model with universal authentication.
    
    This class provides a standardized interface for Azure OpenAI embedding models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: str = "modelname",
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2023-05-15"):
        """
        Initialize Azure OpenAI embedding client.
        
        Args:
            model_name: Name of the Azure OpenAI embedding model to use
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version to use
        """
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self._client = None
        self._auth_manager = UniversalAuthManager(f"azure_openai_embedding_{model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized AzureOpenAIEmbeddingAI with model: {model_name}")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            print("Requesting API token")  # Debug step 1
            token = self._auth_manager.get_token_sync()
            print("Token received successfully")  # Debug step 2
            return token
        except Exception as e:
            print(f"Failed to get token! Error: {str(e)}")
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
                
                # Create a Client first providing the endpoint, api key and version
                self._client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token_provider=lambda: token,
                    api_version=self.api_version,
                )
                
                logger.info(f"Initialized Azure OpenAI embedding client for {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI embedding client: {str(e)}")
                raise
    
    async def get_embeddings(self,
                           texts: Union[str, List[str]],
                           batch_size: int = 100,
                           **kwargs) -> List[List[float]]:
        """
        Get embeddings for one or more text strings.
        
        Args:
            texts: A single text string or a list of text strings
            batch_size: Batch size for embedding requests
            **kwargs: Additional parameters
            
        Returns:
            A list of embedding vectors
        """
        await self._init_client()
        
        try:
            # Handle single string case
            if isinstance(texts, str):
                texts = [texts]
            
            all_embeddings = []
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Prepare input - handle both single strings and lists
                input_data = []
                for text in batch:
                    if isinstance(text, str):
                        input_data.append(text)
                    else:
                        # Handle more complex input formats if needed
                        input_data.append(str(text))
                
                # Example input format based on the image
                if len(input_data) == 1:
                    # Single input case
                    input_text = input_data[0]
                else:
                    # Multiple inputs - you might want to handle this differently
                    # based on your specific use case
                    input_text = input_data
                
                # Create embedding request
                embedding_response = await asyncio.to_thread(
                    self._client.embeddings.create,
                    model=self.model_name,
                    user=os.getenv("USERNAME", ""),
                    input=input_text,
                    **kwargs
                )
                
                # Extract embedding vectors
                batch_embeddings = [data.embedding for data in embedding_response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings using {self.model_name}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def embed_single(self, text: str, **kwargs) -> List[float]:
        """
        Get embedding for a single text string.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector
        """
        embeddings = await self.get_embeddings([text], **kwargs)
        return embeddings[0] if embeddings else []
    
    async def embed_lorem_ipsum(self) -> List[float]:
        """
        Example method to embed lorem ipsum text as shown in the image.
        
        Returns:
            Embedding vector for lorem ipsum text
        """
        lorem_text = [
            "lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
        ]
        
        embeddings = await self.get_embeddings(lorem_text)
        return embeddings
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()


# Example usage
async def example_usage():
    """Example of how to use AzureOpenAIEmbeddingAI."""
    
    # Initialize client
    client = AzureOpenAIEmbeddingAI(
        model_name="modelname",
        azure_endpoint="endpoint_url"
    )
    
    try:
        # Test lorem ipsum embedding as shown in image
        embeddings = await client.embed_lorem_ipsum()
        print(f"Generated {len(embeddings)} lorem ipsum embeddings")
        
        # Test single embedding
        embedding = await client.embed_single("test input")
        print(f"Single embedding length: {len(embedding)}")
        
        # Test authentication health
        health = client.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
