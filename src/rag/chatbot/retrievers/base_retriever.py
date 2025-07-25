import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.core.interfaces.base import Retriever, Document
from src.rag.core.exceptions.exceptions import RetrievalError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BaseRetriever(Retriever):
    """Base class for document retrievers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the retriever with configuration.
        
        Args:
            config: Configuration dictionary for the retriever
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        self.top_k = self.config.get("top_k", 10)
    
    @abstractmethod
    async def _retrieve_documents(self, query: str, config: Dict[str, Any]) -> List[Document]:
        """Implement this method in subclasses for specific retrieval strategies.
        
        Args:
            query: User query string
            config: Configuration parameters for retrieval
            
        Returns:
            List of retrieved documents
        """
        pass
    
    async def retrieve(self, query: str, config: Dict[str, Any] = None) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            config: Configuration parameters for retrieval
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Skip empty query
            if not query.strip():
                logger.warning("Empty query provided for retrieval")
                return []
            
            # Merge provided config with default config
            merged_config = {**self.config, **(config or {})}
            
            # Retrieve documents
            logger.debug(f"Retrieving documents for query: {query}")
            documents = await self._retrieve_documents(query, merged_config)
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            error_msg = f"Document retrieval failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RetrievalError(error_msg) from e
