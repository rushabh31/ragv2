import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.core.interfaces.base import VectorStore, Chunk, SearchResult
from src.rag.core.exceptions.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class BaseVectorStore(VectorStore):
    """Base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the vector store with configuration.
        
        Args:
            config: Configuration dictionary for the vector store
        """
        self.config = config or {}
    
    @abstractmethod
    async def _add_vectors(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Implement this method in subclasses to add vectors to the store.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings
        """
        pass
    
    @abstractmethod
    async def _search_vectors(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Implement this method in subclasses to search vectors in the store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store. Must be called before adding or searching."""
        pass
    
    async def add(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings. If not provided, embeddings 
                       from the chunks will be used if available.
        
        Raises:
            VectorStoreError: If adding vectors fails
        """
        try:
            # Skip empty list
            if not chunks:
                logger.warning("Empty chunk list provided for vector store")
                return
            
            # Validate embeddings if provided
            if embeddings is not None:
                if len(chunks) != len(embeddings):
                    raise ValueError(f"Number of chunks ({len(chunks)}) doesn't match number of embeddings ({len(embeddings)})")
            
            # Add vectors
            logger.debug(f"Adding {len(chunks)} vectors to store")
            await self._add_vectors(chunks, embeddings)
            
        except Exception as e:
            error_msg = f"Failed to add vectors to store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search for similar chunks in the vector store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
            
        Raises:
            VectorStoreError: If searching vectors fails
        """
        try:
            # Validate query embedding
            if not query_embedding:
                raise ValueError("Query embedding cannot be empty")
            
            # Search vectors
            logger.debug(f"Searching for top {top_k} similar vectors")
            results = await self._search_vectors(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
