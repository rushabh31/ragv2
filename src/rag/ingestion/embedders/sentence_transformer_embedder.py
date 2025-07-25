"""Sentence Transformer Embedder for RAG system."""

import logging
from typing import Dict, Any, List

from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.models.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddingAI
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformer embedder using local models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Sentence Transformer embedder.
        
        Args:
            config: Configuration dictionary with keys:
                - model: Model name (default: all-mpnet-base-v2)
                - batch_size: Batch size for processing (default: 32)
                - device: Device to use (cpu/cuda) (default: cpu)
                - normalize_embeddings: Whether to normalize embeddings (default: True)
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "all-mpnet-base-v2")
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cpu")
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)
        self._embedding_model = None
        
        # Load configuration from system config (optional)
        config_manager = ConfigManager()
        system_config = config_manager.get_config("embedding")
        if system_config and system_config.get("provider") == "sentence_transformer":
            embedding_config = system_config.get("config", {})
            self.model_name = embedding_config.get("model", self.model_name)
            self.batch_size = embedding_config.get("batch_size", self.batch_size)
            self.device = embedding_config.get("device", self.device)
            self.normalize_embeddings = embedding_config.get("normalize_embeddings", self.normalize_embeddings)
    
    async def _init_embedding_model(self):
        """Initialize Sentence Transformer model lazily."""
        if self._embedding_model is None:
            try:
                # Create embedding model using the sentence transformer model
                self._embedding_model = SentenceTransformerEmbeddingAI(
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    device=self.device,
                    normalize_embeddings=self.normalize_embeddings
                )
                
                # Validate model availability
                is_valid = await self._embedding_model.validate_authentication()
                if is_valid:
                    logger.info(f"Initialized Sentence Transformer model: {self.model_name} on {self.device}")
                else:
                    logger.warning("Sentence Transformer model validation failed")
            except Exception as e:
                logger.error(f"Failed to initialize Sentence Transformer model: {str(e)}")
                raise
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformer model.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
        """
        await self._init_embedding_model()
        
        # Generate embeddings using local model
        embeddings = await self._embedding_model.get_embeddings(texts)
        
        logger.debug(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "sentence_transformer",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "local_model": True
        }
