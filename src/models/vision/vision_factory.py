"""
Vision Model Factory for Multi-Provider Vision Services.

This factory provides a unified interface for creating vision models across
different providers using the universal authentication system.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Union, List

from .vertex_vision import VertexVisionAI
from .groq_vision import GroqVisionAI
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class VisionProvider(Enum):
    """Enumeration of supported vision providers."""
    VERTEX_AI = "vertex_ai"
    GROQ = "groq"
    # Future providers can be added here
    # OPENAI = "openai"
    # AZURE_OPENAI = "azure_openai"


class VisionModelFactory:
    """Factory for creating vision models across different providers."""
    
    # Registry of available vision models
    _model_registry: Dict[VisionProvider, type] = {
        VisionProvider.VERTEX_AI: VertexVisionAI,
        VisionProvider.GROQ: GroqVisionAI,
    }
    
    # Default model names for each provider
    _default_models: Dict[VisionProvider, str] = {
        VisionProvider.VERTEX_AI: "gemini-1.5-pro-002",
        VisionProvider.GROQ: "llama-3.2-11b-vision-preview",
    }
    
    @classmethod
    def create_model(
        cls,
        provider: Optional[Union[str, VisionProvider]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Union[VertexVisionAI]:
        """Create a vision model instance.
        
        Args:
            provider: Vision provider name or enum (optional, reads from config if not provided)
            model_name: Optional model name (reads from config if not provided)
            **kwargs: Additional model configuration parameters
            
        Returns:
            Configured vision model instance
            
        Raises:
            ValueError: If provider is not supported
            Exception: If model creation fails
        """
        # Get configuration from ConfigManager
        config_manager = ConfigManager()
        vision_config = config_manager.get_section("vision", {})
        
        # Use provider from config if not provided
        if provider is None:
            provider = vision_config.get("provider", "vertex_ai")
        
        # Convert string provider to enum
        if isinstance(provider, str):
            try:
                provider_enum = VisionProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported vision provider: {provider}")
        else:
            provider_enum = provider
        
        # Check if provider is supported
        if provider_enum not in cls._model_registry:
            raise ValueError(f"Vision provider {provider_enum.value} is not registered")
        
        # Get model class
        model_class = cls._model_registry[provider_enum]
        
        # Use model name from config if not provided
        if model_name is None:
            model_config = vision_config.get("config", {})
            model_name = model_config.get("model") or cls._default_models[provider_enum]
        
        # Merge config parameters with kwargs
        model_config = vision_config.get("config", {})
        final_kwargs = {**model_config, **kwargs}
        
        try:
            # Create and return model instance
            logger.info(f"Creating {provider_enum.value} vision model: {model_name} with config from YAML")
            return model_class(model_name=model_name, **final_kwargs)
        except Exception as e:
            logger.error(f"Failed to create {provider_enum.value} vision model: {str(e)}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported vision providers.
        
        Returns:
            List of provider names
        """
        return [provider.value for provider in cls._model_registry.keys()]
    
    @classmethod
    def get_default_model(cls, provider: Union[str, VisionProvider]) -> str:
        """Get default model name for a provider.
        
        Args:
            provider: Vision provider name or enum
            
        Returns:
            Default model name
            
        Raises:
            ValueError: If provider is not supported
        """
        if isinstance(provider, str):
            try:
                provider_enum = VisionProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported vision provider: {provider}")
        else:
            provider_enum = provider
        
        if provider_enum not in cls._default_models:
            raise ValueError(f"No default model defined for provider: {provider_enum.value}")
        
        return cls._default_models[provider_enum]
    
    @classmethod
    def register_model(
        cls,
        provider: VisionProvider,
        model_class: type,
        default_model: str
    ) -> None:
        """Register a new vision model type.
        
        Args:
            provider: Vision provider enum
            model_class: Model class to register
            default_model: Default model name for this provider
        """
        cls._model_registry[provider] = model_class
        cls._default_models[provider] = default_model
        logger.info(f"Registered vision model: {provider.value}")


# Convenience functions for easy model creation
def create_vertex_vision_model(
    model_name: str = "gemini-1.5-pro-002",
    **kwargs
) -> VertexVisionAI:
    """Create a Vertex AI vision model with default settings.
    
    Args:
        model_name: Name of the vision model
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VertexVisionAI instance
    """
    return VisionModelFactory.create_model(
        provider=VisionProvider.VERTEX_AI,
        model_name=model_name,
        **kwargs
    )


async def example_usage():
    """Example of how to use VisionModelFactory."""
    # Create vision model using factory
    vision_model = VisionModelFactory.create_model("vertex_ai")
    
    # Check authentication
    token = vision_model.get_coin_token()
    print(f"Authentication token available: {'Yes' if token else 'No'}")
    
    # Get model info
    info = vision_model.get_model_info()
    print(f"Model info: {info}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
