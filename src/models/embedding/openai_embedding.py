"""
OpenAI Embedding Model.

This module provides a standardized interface for OpenAI embedding models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class OpenAIEmbeddingAI:
    """
    OpenAI embedding model with universal authentication.
    
    This class provides a standardized interface for OpenAI embedding models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 base_url: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 **kwargs):
        """
        Initialize OpenAI embedding client.
        
        Args:
            model_name: Name of the OpenAI embedding model to use (overrides config)
            base_url: Custom base URL for OpenAI API (overrides config)
            batch_size: Batch size for embedding requests (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        embedding_config = config_manager.get_section("embedding", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or embedding_config.get("model", "all-mpnet-base-v2")
        self.base_url = base_url or embedding_config.get("base_url")
        self.batch_size = batch_size or embedding_config.get("batch_size", 100)
        
        # Store additional config parameters
        self.config = {**embedding_config, **kwargs}
        
        self._client = None
        self._auth_manager = UniversalAuthManager(f"openai_embedding_{self.model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized OpenAIEmbeddingAI with model: {self.model_name} (from config)")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("openai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_client(self):
        """Initialize OpenAI client lazily."""
        if self._client is None:
            try:
                # Set SSL certificate path
                os.environ["SSL_CERT_FILE"] = "<PATH TO PROD ca.pem>"
                
                # Get authentication headers
                headers = await self._auth_manager.get_openai_headers()
                
                # Initialize OpenAI client
                self._client = OpenAI(
                    base_url=self.base_url,
                    default_headers=headers
                )
                
                logger.info(f"Initialized OpenAI embedding client for {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding client: {str(e)}")
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
                
                # Create embedding request
                embedding_response = await asyncio.to_thread(
                    self._client.embeddings.create,
                    model=self.model_name,
                    input=batch,
                    user=os.getenv("USERNAME", ""),
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
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()


# Example usage
async def example_usage():
    """Example of how to use OpenAIEmbeddingAI."""
    
    # Initialize client
    client = OpenAIEmbeddingAI()
    
    try:
        # Test single embedding
        embedding = await client.embed_single("input text")
        print(f"Single embedding length: {len(embedding)}")
        
        # Test batch embeddings
        texts = ["input text", "another text", "more text"]
        embeddings = await client.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Test authentication health
        health = client.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
