"""
Vertex AI Embedder for RAG System.

This module provides a RAG-compatible embedder wrapper for Vertex AI embedding models
using the universal authentication system.
"""

import logging
from typing import Dict, Any, List

from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.models.embedding import VertexEmbeddingAI

logger = logging.getLogger(__name__)


class VertexAIEmbedder(BaseEmbedder):
    """
    Vertex AI embedder for RAG system.
    
    This class wraps the VertexEmbeddingAI model to provide RAG-compatible
    embedding capabilities with the universal authentication system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Vertex AI embedder.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "text-embedding-004")
        self.project_id = self.config.get("project_id")
        self.location = self.config.get("location", "us-central1")
        self.batch_size = self.config.get("batch_size", 100)
        
        self._vertex_model = None
        
        logger.info(f"Initialized VertexAIEmbedder with model: {self.model_name}")
    
    async def _init_vertex_model(self):
        """Initialize Vertex AI embedding model lazily."""
        if self._vertex_model is None:
            self._vertex_model = VertexEmbeddingAI(
                model_name=self.model_name,
                project_id=self.project_id,
                location=self.location
            )
            logger.info(f"Initialized Vertex AI embedding model: {self.model_name}")
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """
        Generate embeddings using Vertex AI.
        
        Args:
            texts: List of texts to embed
            config: Generation configuration
            
        Returns:
            List of embedding vectors
        """
        await self._init_vertex_model()
        
        try:
            # Get batch size from config or use default
            batch_size = config.get("batch_size", self.batch_size)
            
            # Generate embeddings
            embeddings = await self._vertex_model.get_embeddings(
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
        if self._vertex_model:
            return self._vertex_model.get_auth_health_status()
        return {"status": "not_initialized"}
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        await self._init_vertex_model()
        return await self._vertex_model.validate_authentication()
