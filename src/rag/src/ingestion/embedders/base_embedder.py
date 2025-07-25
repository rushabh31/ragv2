import logging
from abc import abstractmethod
from typing import Dict, Any, List

from src.rag.src.core.interfaces.base import Embedder
from src.rag.src.core.exceptions.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class BaseEmbedder(Embedder):
    """Base class for text embedders."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the embedder with configuration.
        
        Args:
            config: Configuration dictionary for the embedder
        """
        self.config = config or {}
    
    @abstractmethod
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Implement this method in subclasses for specific embedding strategies.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
        """
        pass
    
    async def embed(self, texts: List[str], config: Dict[str, Any] = None) -> List[List[float]]:
        """Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Skip empty list
            if not texts:
                logger.warning("Empty text list provided for embedding")
                return []
            
            # Merge provided config with default config
            merged_config = {**self.config, **(config or {})}
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = await self._generate_embeddings(texts, merged_config)
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingError(error_msg) from e
