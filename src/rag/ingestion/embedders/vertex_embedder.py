import logging
from typing import Dict, Any, List

from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.models.embedding.embedding_factory import EmbeddingModelFactory
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class VertexEmbedder(BaseEmbedder):
    """Embedder that uses Google Cloud VertexAI text embedding models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the embedder with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - model: Name of the embedding model to use
                - batch_size: Number of texts to embed in a single API call
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "text-embedding-004")
        self.batch_size = self.config.get("batch_size", 100)
        self._embedding_model = None
        
        # Load embedding configuration from system config
        config_manager = ConfigManager()
        system_config = config_manager.get_config("embedding")
        if system_config:
            self.embedding_provider = system_config.get("provider", "vertex_ai")
            self.embedding_config = system_config.get("config", {})
        else:
            # Default to vertex_ai if no embedding config found
            self.embedding_provider = "vertex_ai"
            self.embedding_config = {}
    
    async def _init_embedding_model(self):
        """Initialize embedding model lazily using factory."""
        if self._embedding_model is None:
            try:
                # Create embedding model using factory based on configuration
                self._embedding_model = EmbeddingModelFactory.create_model(
                    provider=self.embedding_provider,
                    model_name=self.model_name,
                    **self.embedding_config
                )
                
                # Validate authentication
                is_valid = await self._embedding_model.validate_authentication()
                if is_valid:
                    logger.info(f"Initialized {self.embedding_provider} embedding model: {self.model_name}")
                else:
                    logger.warning(f"{self.embedding_provider} embedding model authentication validation failed")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using configurable embedding model.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
        """
        await self._init_embedding_model()
        
        # Generate embeddings using configurable model
        embeddings = await self._embedding_model.get_embeddings(texts)
        
        logger.debug(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        return embeddings
