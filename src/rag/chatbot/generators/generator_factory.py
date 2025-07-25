"""Factory for creating generator instances based on configuration."""

import logging
import os
from typing import Dict, Any, Optional, Type

from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.chatbot.generators.vertex_generator import VertexGenerator
from src.rag.chatbot.generators.anthropic_vertex_generator import AnthropicVertexGenerator
from src.rag.chatbot.generators.openai_generator import OpenAIGenerator
from src.rag.chatbot.generators.azure_openai_generator import AzureOpenAIGenerator
from src.rag.chatbot.generators.groq_generator import GroqGenerator
from src.rag.core.exceptions.exceptions import GenerationError

logger = logging.getLogger(__name__)

# Registry of available generators
GENERATOR_REGISTRY: Dict[str, Type[BaseGenerator]] = {
    "vertex": VertexGenerator,
    "anthropic_vertex": AnthropicVertexGenerator,
    "openai": OpenAIGenerator,
    "azure_openai": AzureOpenAIGenerator,
    "groq": GroqGenerator
}

class GeneratorFactory:
    """Factory for creating generator instances."""
    
    @staticmethod
    async def create_generator(config: Dict[str, Any]) -> BaseGenerator:
        """Create a generator instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized generator
            
        Raises:
            GenerationError: If provider is not supported
        """
        # Get the provider from config with vertex as default
        provider = config.get("provider", "vertex").lower()
        logger.info(f"Generator provider from config: {provider}")
        
        # Allow Vertex to run with token-based auth instead of forcing Groq
        # Our updated implementation now supports token-based authentication for Vertex AI
        
        if provider not in GENERATOR_REGISTRY:
            logger.error(f"Unsupported generator provider: {provider}")
            raise GenerationError(f"Unsupported generator provider: {provider}")
        
        try:
            generator_class = GENERATOR_REGISTRY[provider]
            logger.info(f"Creating generator with provider: {provider}")
            return generator_class(config)
        except Exception as e:
            logger.error(f"Failed to create generator with provider {provider}: {str(e)}", exc_info=True)
            raise GenerationError(f"Failed to create generator with provider {provider}: {str(e)}")
    
    @staticmethod
    def register_generator(name: str, generator_class: Type[BaseGenerator]) -> None:
        """Register a new generator type.
        
        Args:
            name: Name for the generator type
            generator_class: Generator class to register
        """
        GENERATOR_REGISTRY[name.lower()] = generator_class
        logger.info(f"Registered generator type: {name}")
    
    @staticmethod
    def get_available_generators() -> Dict[str, Type[BaseGenerator]]:
        """Get all registered generator types.
        
        Returns:
            Dictionary of generator types
        """
        return GENERATOR_REGISTRY.copy()
