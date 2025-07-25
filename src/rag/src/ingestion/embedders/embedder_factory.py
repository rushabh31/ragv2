"""Factory for creating embedder instances based on configuration."""

import logging
from typing import Dict, Any, Optional, Type

from src.rag.src.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.src.ingestion.embedders.vertex_embedder import VertexEmbedder
from src.rag.src.ingestion.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.rag.src.ingestion.embedders.openai_embedder import OpenAIEmbedder
from src.rag.src.core.exceptions.exceptions import EmbeddingError
from src.rag.src.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Registry of available embedders
EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "vertex": VertexEmbedder,
    "openai": OpenAIEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder
}

class EmbedderFactory:
    """Factory for creating embedder instances."""
    
    @staticmethod
    async def create_embedder(config: Dict[str, Any]) -> BaseEmbedder:
        """Create an embedder instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized embedder
            
        Raises:
            EmbeddingError: If provider is not supported
        """
        # Default to sentence_transformer as fallback
        provider = config.get("provider", "sentence_transformer").lower()
        
        # Check if we have token-based auth configuration
        config_manager = ConfigManager()
        vertex_config = config_manager.get_config("vertex")
        
        # No need to force sentence_transformer if we have token-based auth
        if vertex_config:
            logger.info("Token-based authentication is available for cloud embedders")
        else:
            # Only force to sentence_transformer if no token auth and no direct API config
            if provider in ["vertex", "openai"] and not config.get("api_key"):
                logger.warning(f"{provider.capitalize()} configuration not found. Defaulting to sentence_transformer provider.")
                provider = "sentence_transformer"
        
        if provider not in EMBEDDER_REGISTRY:
            logger.error(f"Unsupported embedder provider: {provider}")
            raise EmbeddingError(f"Unsupported embedder provider: {provider}")
        
        try:
            embedder_class = EMBEDDER_REGISTRY[provider]
            logger.info(f"Creating embedder with provider: {provider}")
            return embedder_class(config)
        except Exception as e:
            logger.error(f"Failed to create embedder with provider {provider}: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to create embedder with provider {provider}: {str(e)}")
    
    @staticmethod
    def register_embedder(name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """Register a new embedder type.
        
        Args:
            name: Name for the embedder type
            embedder_class: Embedder class to register
        """
        EMBEDDER_REGISTRY[name.lower()] = embedder_class
        logger.info(f"Registered embedder type: {name}")
    
    @staticmethod
    def get_available_embedders() -> Dict[str, Type[BaseEmbedder]]:
        """Get all registered embedder types.
        
        Returns:
            Dictionary of embedder types
        """
        return EMBEDDER_REGISTRY.copy()
