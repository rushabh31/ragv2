"""
Generation Model Factory.

This module provides a factory for creating generation models across different providers
using the universal authentication system.
"""

import logging
from typing import Dict, Any, Optional, Type, Union, List
from enum import Enum

from .anthropic_vertex import AnthropicVertexGenAI
from .openai_gen import OpenAIGenAI
from .vertex_gen import VertexGenAI
from .azure_openai_gen import AzureOpenAIGenAI
from .groq_gen import GroqGenAI
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class GenerationProvider(Enum):
    """Supported generation providers."""
    ANTHROPIC_VERTEX = "anthropic_vertex"
    OPENAI = "openai"
    VERTEX_AI = "vertex_ai"
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"


class GenerationModelFactory:
    """
    Factory for creating generation models with universal authentication.
    
    This factory provides a unified interface for creating generation models
    across different providers while using the universal authentication system.
    """
    
    _model_registry: Dict[GenerationProvider, Type] = {
        GenerationProvider.ANTHROPIC_VERTEX: AnthropicVertexGenAI,
        GenerationProvider.OPENAI: OpenAIGenAI,
        GenerationProvider.VERTEX_AI: VertexGenAI,
        GenerationProvider.AZURE_OPENAI: AzureOpenAIGenAI,
        GenerationProvider.GROQ: GroqGenAI,
    }
    
    _default_models: Dict[GenerationProvider, str] = {
        GenerationProvider.ANTHROPIC_VERTEX: "claude-3-5-sonnet@20240229",
        GenerationProvider.OPENAI: "Meta-Llama-3-70B-Instruct",
        GenerationProvider.VERTEX_AI: "gemini-1.5-pro-002",
        GenerationProvider.AZURE_OPENAI: "GPT4-o",
        GenerationProvider.GROQ: "llama-3.1-70b-versatile",
    }
    
    @classmethod
    def create_model(cls,
                    provider: Optional[Union[str, GenerationProvider]] = None,
                    model_name: Optional[str] = None,
                    **kwargs) -> Union[AnthropicVertexGenAI, OpenAIGenAI, VertexGenAI, AzureOpenAIGenAI]:
        """
        Create a generation model for the specified provider.
        
        Args:
            provider: Provider name or enum (optional, reads from config if not provided)
            model_name: Optional model name (reads from config if not provided)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized generation model
            
        Raises:
            ValueError: If provider is not supported
        """
        # Get configuration from ConfigManager
        config_manager = ConfigManager()
        generation_config = config_manager.get_section("generation", {})
        
        # Use provider from config if not provided
        if provider is None:
            provider = generation_config.get("provider", "vertex")
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            try:
                provider = GenerationProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}. "
                               f"Supported providers: {[p.value for p in GenerationProvider]}")
        
        # Get model class
        model_class = cls._model_registry.get(provider)
        if not model_class:
            raise ValueError(f"No model class registered for provider: {provider}")
        
        # Use model name from config if not provided
        if model_name is None:
            model_config = generation_config.get("config", {})
            model_name = model_config.get("model_name") or cls._default_models[provider]
        
        # Merge config parameters with kwargs
        model_config = generation_config.get("config", {})
        final_kwargs = {**model_config, **kwargs}
        
        # Create and return model instance
        try:
            logger.info(f"Creating {provider.value} generation model: {model_name} with config from YAML")
            return model_class(model_name=model_name, **final_kwargs)
        except Exception as e:
            logger.error(f"Failed to create {provider.value} generation model: {str(e)}")
            raise
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported provider names."""
        return [provider.value for provider in GenerationProvider]
    
    @classmethod
    def get_default_model(cls, provider: Union[str, GenerationProvider]) -> str:
        """Get default model name for a provider."""
        if isinstance(provider, str):
            provider = GenerationProvider(provider.lower())
        
        return cls._default_models.get(provider, "")
    
    @classmethod
    def register_model(cls,
                      provider: GenerationProvider,
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
def create_anthropic_vertex_model(model_name: Optional[str] = None, **kwargs) -> AnthropicVertexGenAI:
    """Create Anthropic Vertex AI model."""
    return GenerationModelFactory.create_model(
        GenerationProvider.ANTHROPIC_VERTEX, model_name, **kwargs
    )


def create_openai_model(model_name: Optional[str] = None, **kwargs) -> OpenAIGenAI:
    """Create OpenAI model."""
    return GenerationModelFactory.create_model(
        GenerationProvider.OPENAI, model_name, **kwargs
    )


def create_vertex_ai_model(model_name: Optional[str] = None, **kwargs) -> VertexGenAI:
    """Create Vertex AI model."""
    return GenerationModelFactory.create_model(
        GenerationProvider.VERTEX_AI, model_name, **kwargs
    )


def create_azure_openai_model(model_name: Optional[str] = None, **kwargs) -> AzureOpenAIGenAI:
    """Create Azure OpenAI model."""
    return GenerationModelFactory.create_model(
        GenerationProvider.AZURE_OPENAI, model_name, **kwargs
    )


# Example usage
async def example_usage():
    """Example of how to use the GenerationModelFactory."""
    
    try:
        # Create models using factory
        anthropic_model = GenerationModelFactory.create_model("anthropic_vertex")
        openai_model = GenerationModelFactory.create_model("openai")
        vertex_model = GenerationModelFactory.create_model("vertex_ai")
        azure_model = GenerationModelFactory.create_model("azure_openai")
        
        # Test each model
        prompt = "What is artificial intelligence?"
        
        print("Testing Anthropic Vertex AI:")
        response = await anthropic_model.chat_completion(prompt)
        print(f"Response: {response[:100]}...")
        
        print("\\nTesting OpenAI:")
        response = await openai_model.chat_completion(prompt)
        print(f"Response: {response[:100]}...")
        
        print("\\nTesting Vertex AI:")
        response = await vertex_model.chat_completion(prompt)
        print(f"Response: {response[:100]}...")
        
        print("\\nTesting Azure OpenAI:")
        response = await azure_model.chat_completion(prompt)
        print(f"Response: {response[:100]}...")
        
        # Show supported providers
        print(f"\\nSupported providers: {GenerationModelFactory.get_supported_providers()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
