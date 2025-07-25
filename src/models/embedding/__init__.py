"""
Embedding Models Package.

This package provides standardized embedding model interfaces across different providers
using the universal authentication system.
"""

from .vertex_embedding import VertexEmbeddingAI
from .openai_embedding import OpenAIEmbeddingAI
from .azure_openai_embedding import AzureOpenAIEmbeddingAI
from .sentence_transformer_embedding import SentenceTransformerEmbeddingAI
from .embedding_factory import EmbeddingModelFactory, EmbeddingProvider

__all__ = [
    "VertexEmbeddingAI",
    "OpenAIEmbeddingAI",
    "AzureOpenAIEmbeddingAI",
    "SentenceTransformerEmbeddingAI",
    "EmbeddingModelFactory",
    "EmbeddingProvider"
]
