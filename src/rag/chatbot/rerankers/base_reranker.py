import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.core.interfaces.base import Reranker, Document
from src.rag.core.exceptions.exceptions import RerankerError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BaseReranker(Reranker):
    """Base class for document rerankers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the reranker with configuration.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.top_k = self.config.get("top_k", 5)
    
    @abstractmethod
    async def _rerank_documents(self, query: str, documents: List[Document], config: Dict[str, Any]) -> List[Document]:
        """Implement this method in subclasses for specific reranking strategies.
        
        Args:
            query: User query string
            documents: List of documents to rerank
            config: Configuration parameters for reranking
            
        Returns:
            Reranked list of documents
        """
        pass
    
    async def rerank(self, query: str, documents: List[Document], config: Dict[str, Any] = None) -> List[Document]:
        """Rerank a list of documents based on relevance to the query.
        
        Args:
            query: User query string
            documents: List of documents to rerank
            config: Configuration parameters for reranking
            
        Returns:
            Reranked list of documents
            
        Raises:
            RerankerError: If reranking fails
        """
        try:
            # Skip reranking if disabled or no documents to rerank
            merged_config = {**self.config, **(config or {})}
            if not merged_config.get("enabled", self.enabled):
                logger.debug("Reranking is disabled, returning original documents")
                return documents
            
            if not documents:
                logger.warning("No documents to rerank")
                return []
            
            if not query.strip():
                logger.warning("Empty query provided for reranking")
                return documents
            
            # Rerank documents
            logger.debug(f"Reranking {len(documents)} documents for query: {query}")
            reranked_documents = await self._rerank_documents(query, documents, merged_config)
            
            # Limit to top_k if specified
            top_k = merged_config.get("top_k", self.top_k)
            if top_k > 0 and top_k < len(reranked_documents):
                reranked_documents = reranked_documents[:top_k]
            
            logger.info(f"Reranked to {len(reranked_documents)} documents")
            return reranked_documents
            
        except Exception as e:
            error_msg = f"Document reranking failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RerankerError(error_msg) from e
