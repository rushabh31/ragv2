class RAGException(Exception):
    """Base exception class for all RAG system exceptions."""
    pass


class DocumentProcessingError(RAGException):
    """Exception raised for errors during document processing."""
    pass


class ParsingError(RAGException):
    """Exception raised for errors during document parsing."""
    pass


class ChunkingError(RAGException):
    """Exception raised for errors during document chunking."""
    pass


class EmbeddingError(RAGException):
    """Exception raised for errors during text embedding."""
    pass


class IndexingError(RAGException):
    """Exception raised for errors during document indexing."""
    pass


class VectorStoreError(RAGException):
    """Exception raised for vector store related errors."""
    pass


class RetrievalError(RAGException):
    """Exception raised for errors during document retrieval."""
    pass


class RerankerError(RAGException):
    """Exception raised for errors during document reranking."""
    pass


class GenerationError(RAGException):
    """Exception raised for errors during response generation."""
    pass


class MemoryError(RAGException):
    """Exception raised for errors related to conversation memory."""
    pass


class ConfigError(RAGException):
    """Exception raised for errors related to system configuration."""
    pass


class AuthenticationError(RAGException):
    """Exception raised for authentication failures."""
    pass


class RateLimitError(RAGException):
    """Exception raised when rate limits are exceeded."""
    pass


class ResourceNotFoundError(RAGException):
    """Exception raised when a requested resource is not found."""
    pass
