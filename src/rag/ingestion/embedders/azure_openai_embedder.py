"""
Azure OpenAI Embedder for RAG System.

This module provides a RAG-compatible embedder wrapper for Azure OpenAI embedding models
using the universal authentication system.
"""

import logging
from typing import Dict, Any, List

from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.models.embedding import AzureOpenAIEmbeddingAI

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedder(BaseEmbedder):
    """
    Azure OpenAI embedder for RAG system.
    
    This class wraps the AzureOpenAIEmbeddingAI model to provide RAG-compatible
    embedding capabilities with the universal authentication system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure OpenAI embedder.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "modelname")
        self.azure_endpoint = self.config.get("azure_endpoint")
        self.api_version = self.config.get("api_version", "2023-05-15")
        self.batch_size = self.config.get("batch_size", 100)
        
        self._azure_model = None
        
        logger.info(f"Initialized AzureOpenAIEmbedder with model: {self.model_name}")
    
    async def _init_azure_model(self):
        """Initialize Azure OpenAI embedding model lazily."""
        if self._azure_model is None:
            self._azure_model = AzureOpenAIEmbeddingAI(
                model_name=self.model_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            logger.info(f"Initialized Azure OpenAI embedding model: {self.model_name}")
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """
        Generate embeddings using Azure OpenAI.
        
        Args:
            texts: List of texts to embed
            config: Generation configuration
            
        Returns:
            List of embedding vectors
        """
        await self._init_azure_model()
        
        try:
            # Get batch size from config or use default
            batch_size = config.get("batch_size", self.batch_size)
            
            # Generate embeddings
            embeddings = await self._azure_model.get_embeddings(
                texts=texts,
                batch_size=batch_size
            )
            
            logger.debug(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        if self._azure_model:
            return self._azure_model.get_auth_health_status()
        return {"status": "not_initialized"}
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        await self._init_azure_model()
        return await self._azure_model.validate_authentication()
