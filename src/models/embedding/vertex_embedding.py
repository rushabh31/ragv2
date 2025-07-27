"""
Vertex AI Embedding Model.

This module provides a standardized interface for Google Vertex AI text embedding models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from vertexai.language_models import TextEmbeddingModel
import vertexai

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class VertexEmbeddingAI:
    """
    Vertex AI embedding model with universal authentication.
    
    This class provides a standardized interface for Google Vertex AI text embedding models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 **kwargs):
        """
        Initialize Vertex AI embedding client.
        
        Args:
            model_name: Name of the Vertex AI embedding model to use (overrides config)
            project_id: GCP project ID (overrides env var)
            location: GCP location for Vertex AI (overrides config)
            batch_size: Batch size for embedding requests (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        embedding_config = config_manager.get_section("embedding", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or embedding_config.get("model", "text-embedding-004")
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        self.location = location or embedding_config.get("location", "us-central1")
        self.batch_size = batch_size or embedding_config.get("batch_size", 100)
        
        # Store additional config parameters
        self.config = {**embedding_config, **kwargs}
        
        self._model = None
        self._auth_manager = UniversalAuthManager(f"vertex_embedding_{self.model_name}")
        self._auth_manager.configure()
        
        # Set metadata for user tracking
        self.metadata = [("x-r2d2-user", os.getenv("USERNAME", ""))]
        
        logger.info(f"Initialized VertexEmbeddingAI with model: {self.model_name} (from config)")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("vertex_ai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_model(self):
        """Initialize Vertex AI embedding model lazily."""
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
                
                # Initialize the text embedding model
                self._model = TextEmbeddingModel.from_pretrained(
                    self.model_name, 
                    metadata=self.metadata
                )
                
                logger.info(f"Initialized Vertex AI embedding model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI embedding model: {str(e)}")
                raise
    
    async def get_embeddings(self,
                           texts: Union[str, List[str]],
                           batch_size: Optional[int] = None,
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
        await self._init_model()
        
        try:
            # Handle single string case
            if isinstance(texts, str):
                texts = [texts]
            
            # Use provided batch_size or fall back to instance default
            effective_batch_size = batch_size or self.batch_size
            
            all_embeddings = []
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i:i + effective_batch_size]
                
                # Get embeddings for the batch
                embeddings = await asyncio.to_thread(
                    self._model.get_embeddings,
                    batch,
                    metadata=self.metadata
                )
                
                # Extract the embedding vectors
                batch_embeddings = [embedding.values for embedding in embeddings]
                all_embeddings.extend(batch_embeddings)
                
                # Print debug info for each batch
                for j, embedding in enumerate(batch_embeddings):
                    vector = embedding
                    print(f"Length of Embedding Vector: {len(vector)}")
            
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
    """Example of how to use VertexEmbeddingAI."""
    
    # Set environment variables for testing
    os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem files>"
    
    # Initialize client
    vertex_embedding_ai = VertexEmbeddingAI()
    
    try:
        # Test single embedding
        embedding = await vertex_embedding_ai.embed_single("What is life?")
        print(f"Single embedding length: {len(embedding)}")
        
        # Test batch embeddings
        texts = ["What is life?", "How does AI work?", "Explain machine learning"]
        embeddings = await vertex_embedding_ai.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Test authentication health
        health = vertex_embedding_ai.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
