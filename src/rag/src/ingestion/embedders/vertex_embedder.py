import logging
from typing import Dict, Any, List

from src.rag.src.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.src.shared.utils.vertex_ai import VertexGenAI
from src.rag.src.shared.utils.config_manager import ConfigManager

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
        self._vertex_ai = None
        
        # Load vertex configuration for token-based authentication
        config_manager = ConfigManager()
        self.vertex_config = config_manager.get_config("vertex")
    
    async def _init_vertex_ai(self):
        """Initialize VertexAI client lazily."""
        if self._vertex_ai is None:
            # Initialize with embedding model
            self._vertex_ai = VertexGenAI(embedding_model=self.model_name)
            
            # Set the vertex configuration for token-based auth
            if self.vertex_config:
                self._vertex_ai.vertex_config = self.vertex_config
                logger.info(f"Initialized VertexAI embedder with model: {self.model_name} using token-based authentication")
            else:
                logger.warning("Vertex AI configuration not found. Token-based authentication won't be available for embeddings.")
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using VertexAI text embedding model.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
        """
        await self._init_vertex_ai()
        
        # Get configuration
        batch_size = config.get("batch_size", self.batch_size)
        
        # Generate embeddings
        embeddings = await self._vertex_ai.get_embeddings(texts, batch_size=batch_size)
        
        logger.debug(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        return embeddings
