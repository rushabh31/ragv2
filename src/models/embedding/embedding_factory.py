"""
Embedding Model Factory.

This module provides a factory for creating embedding models across different providers
using the universal authentication system.
"""

import logging
from typing import Dict, Any, Optional, Type, Union, List
from enum import Enum

from .vertex_embedding import VertexEmbeddingAI
from .openai_embedding import OpenAIEmbeddingAI
from .azure_openai_embedding import AzureOpenAIEmbeddingAI
from .sentence_transformer_embedding import SentenceTransformerEmbeddingAI
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class EmbeddingModelFactory:
    """
    Factory for creating embedding models with universal authentication.
    
    This factory provides a unified interface for creating embedding models
    across different providers while using the universal authentication system.
    """
    
    _model_registry: Dict[EmbeddingProvider, Type] = {
        EmbeddingProvider.VERTEX_AI: VertexEmbeddingAI,
        EmbeddingProvider.OPENAI: OpenAIEmbeddingAI,
        EmbeddingProvider.AZURE_OPENAI: AzureOpenAIEmbeddingAI,
        EmbeddingProvider.SENTENCE_TRANSFORMER: SentenceTransformerEmbeddingAI,
    }
    
    _default_models: Dict[EmbeddingProvider, str] = {
        EmbeddingProvider.VERTEX_AI: "text-embedding-004",
        EmbeddingProvider.OPENAI: "all-mpnet-base-v2",
        EmbeddingProvider.AZURE_OPENAI: "modelname",
        EmbeddingProvider.SENTENCE_TRANSFORMER: "all-mpnet-base-v2",
    }
    
    @classmethod
    def create_model(cls,
                    provider: Optional[Union[str, EmbeddingProvider]] = None,
                    model_name: Optional[str] = None,
                    **kwargs) -> Union[VertexEmbeddingAI, OpenAIEmbeddingAI, AzureOpenAIEmbeddingAI, SentenceTransformerEmbeddingAI]:
        """
        Create an embedding model for the specified provider.
        
        Args:
            provider: Provider name or enum (optional, reads from config if not provided)
            model_name: Optional model name (reads from config if not provided)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized embedding model
            
        Raises:
            ValueError: If provider is not supported
        """
        # Get configuration from ConfigManager
        config_manager = ConfigManager()
        embedding_config = config_manager.get_section("embedding", {})
        
        # Use provider from config if not provided
        if provider is None:
            provider = embedding_config.get("provider", "vertex_ai")
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            try:
                provider = EmbeddingProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}. "
                               f"Supported providers: {[p.value for p in EmbeddingProvider]}")
        
        # Get model class
        model_class = cls._model_registry.get(provider)
        if not model_class:
            raise ValueError(f"No model class registered for provider: {provider}")
        
        # Use model name from config if not provided
        if model_name is None:
            model_config = embedding_config.get("config", {})
            model_name = model_config.get("model") or cls._default_models[provider]
        
        # Merge config parameters with kwargs
        model_config = embedding_config.get("config", {})
        final_kwargs = {**model_config, **kwargs}
        
        # Create and return model instance
        try:
            logger.info(f"Creating {provider.value} embedding model: {model_name} with config from YAML")
            return model_class(model_name=model_name, **final_kwargs)
        except Exception as e:
            logger.error(f"Failed to create {provider.value} embedding model: {str(e)}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported provider names."""
        return [provider.value for provider in EmbeddingProvider]
    
    @classmethod
    def get_default_model(cls, provider: Union[str, EmbeddingProvider]) -> str:
        """Get default model name for a provider."""
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider.lower())
        
        return cls._default_models.get(provider, "")
    
    @classmethod
    def register_model(cls,
                      provider: EmbeddingProvider,
                      model_class: Type,
                      default_model: str):
        """
        Register a new model class for a provider.
        
        Args:
            provider: Provider enum
            model_class: Model class to register
            default_model: Default model name for this provider
        """
        cls._model_registry[provider] = model_class
        cls._default_models[provider] = default_model
        logger.info(f"Registered {provider.value} with model class {model_class.__name__}")


# Convenience functions for creating specific models
def create_vertex_embedding_model(model_name: Optional[str] = None, **kwargs) -> VertexEmbeddingAI:
    """Create Vertex AI embedding model."""
    return EmbeddingModelFactory.create_model(
        EmbeddingProvider.VERTEX_AI, model_name, **kwargs
    )


def create_openai_embedding_model(model_name: Optional[str] = None, **kwargs) -> OpenAIEmbeddingAI:
    """Create OpenAI embedding model."""
    return EmbeddingModelFactory.create_model(
        EmbeddingProvider.OPENAI, model_name, **kwargs
    )


def create_azure_openai_embedding_model(model_name: Optional[str] = None, **kwargs) -> AzureOpenAIEmbeddingAI:
    """Create Azure OpenAI embedding model."""
    return EmbeddingModelFactory.create_model(
        EmbeddingProvider.AZURE_OPENAI, model_name, **kwargs
    )


# Example usage
async def example_usage():
    """Example of how to use the EmbeddingModelFactory."""
    
    try:
        # Create models using factory
        vertex_model = EmbeddingModelFactory.create_model("vertex_ai")
        openai_model = EmbeddingModelFactory.create_model("openai")
        azure_model = EmbeddingModelFactory.create_model("azure_openai")
        
        # Test each model
        test_text = "What is artificial intelligence?"
        
        print("Testing Vertex AI Embedding:")
        embedding = await vertex_model.embed_single(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        
        print("\\nTesting OpenAI Embedding:")
        embedding = await openai_model.embed_single(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        
        print("\\nTesting Azure OpenAI Embedding:")
        embedding = await azure_model.embed_single(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        
        # Show supported providers
        print(f"\\nSupported providers: {EmbeddingModelFactory.get_supported_providers()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
