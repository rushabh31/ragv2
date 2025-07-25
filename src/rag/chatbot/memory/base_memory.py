import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.core.interfaces.base import Memory
from src.rag.core.exceptions.exceptions import MemoryError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BaseMemory(Memory):
    """Base class for conversation memory systems."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the memory system with configuration.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        self.max_history = self.config.get("max_history", 10)
    
    @abstractmethod
    async def _add_interaction(self, 
                             session_id: str,
                             query: str, 
                             response: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Implement this method in subclasses for specific memory storage strategies.
        
        Args:
            session_id: Unique session identifier
            query: User query string
            response: System response
            metadata: Additional metadata about the interaction
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _get_conversation_history(self, 
                                      session_id: str, 
                                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Implement this method in subclasses for specific history retrieval strategies.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation interactions
        """
        pass
    
    @abstractmethod
    async def _get_relevant_history(self, 
                                  session_id: str, 
                                  query: str, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Implement this method in subclasses for query-relevant history retrieval.
        
        Args:
            session_id: Unique session identifier
            query: Current query to find relevant history for
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of relevant conversation interactions
        """
        pass
    
    async def add(self, 
                session_id: str,
                query: str, 
                response: str, 
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new interaction to the conversation memory.
        
        Args:
            session_id: Unique session identifier
            query: User query string
            response: System response
            metadata: Additional metadata about the interaction
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemoryError: If adding the interaction fails
        """
        try:
            return await self._add_interaction(session_id, query, response, metadata)
        except Exception as e:
            error_msg = f"Failed to add interaction to memory: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryError(error_msg) from e
    
    async def get_history(self, 
                        session_id: str, 
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation interactions
            
        Raises:
            MemoryError: If retrieving history fails
        """
        try:
            max_history = limit or self.max_history
            return await self._get_conversation_history(session_id, max_history)
        except Exception as e:
            error_msg = f"Failed to retrieve conversation history: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryError(error_msg) from e
    
    async def get_relevant(self, 
                         session_id: str, 
                         query: str, 
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history.
        
        Args:
            session_id: Unique session identifier
            query: Current query to find relevant history for
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of relevant conversation interactions
            
        Raises:
            MemoryError: If retrieving relevant history fails
        """
        try:
            max_history = limit or self.max_history
            return await self._get_relevant_history(session_id, query, max_history)
        except Exception as e:
            error_msg = f"Failed to retrieve relevant conversation history: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryError(error_msg) from e
