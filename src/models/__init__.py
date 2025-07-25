"""
Universal Models Package for Multi-Provider AI Services.

This package provides standardized interfaces for generation, embedding, and vision models
across different providers (Anthropic, OpenAI, Vertex AI, Azure OpenAI) using the
universal authentication system.
"""

from .generation import (
    AnthropicVertexGenAI,
    OpenAIGenAI,
    VertexGenAI,
    AzureOpenAIGenAI
)

from .embedding import (
    VertexEmbeddingAI,
    OpenAIEmbeddingAI,
    AzureOpenAIEmbeddingAI
)

from .vision import (
    VertexVisionAI
)

__all__ = [
    # Generation models
    "AnthropicVertexGenAI",
    "OpenAIGenAI", 
    "VertexGenAI",
    "AzureOpenAIGenAI",
    # Embedding models
    "VertexEmbeddingAI",
    "OpenAIEmbeddingAI",
    "AzureOpenAIEmbeddingAI",
    # Vision models
    "VertexVisionAI"
]
