"""
No Checkpoint Memory Implementation

This memory implementation provides a stateless conversation experience
where no conversation history is persisted. Each interaction is treated
as independent with no memory of previous exchanges.

Useful for:
- Stateless API endpoints
- Privacy-sensitive applications
- High-volume scenarios where persistence is not needed
- Testing and development environments
"""

import logging
from typing import Dict, Any, List, Optional

from src.rag.chatbot.memory.base_memory import BaseMemory

logger = logging.getLogger(__name__)


class NoCheckpointMemory(BaseMemory):
    """
    Memory implementation that provides no persistent storage.
    
    This implementation:
    - Does not store any conversation history
    - Returns empty history for all queries
    - Always reports successful operations without actual storage
    - Provides maximum privacy and minimal resource usage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the no-checkpoint memory system.
        
        Args:
            config: Configuration dictionary (mostly ignored for this implementation)
        """
        super().__init__(config)
        self._enabled = config.get("enabled", True) if config else True
        logger.info("Initialized NoCheckpointMemory - no conversation history will be stored")
    
    async def _add_interaction(self, 
                             session_id: str,
                             query: str, 
                             response: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Simulate adding an interaction without actually storing it.
        
        Args:
            session_id: Unique session identifier (ignored)
            query: User query string (ignored)
            response: System response (ignored)
            metadata: Additional metadata (ignored)
            
        Returns:
            Always returns True to indicate "successful" operation
        """
        if not self._enabled:
            return False
            
        logger.debug(f"NoCheckpointMemory: Simulated adding interaction for session {session_id}")
        return True
    
    async def _get_conversation_history(self, 
                                      session_id: str, 
                                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return empty conversation history.
        
        Args:
            session_id: Unique session identifier (ignored)
            limit: Maximum number of interactions to retrieve (ignored)
            
        Returns:
            Always returns an empty list
        """
        logger.debug(f"NoCheckpointMemory: Returning empty history for session {session_id}")
        return []
    
    async def _get_relevant_history(self, 
                                  session_id: str, 
                                  query: str, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return empty relevant history.
        
        Args:
            session_id: Unique session identifier (ignored)
            query: Current query to find relevant history for (ignored)
            limit: Maximum number of interactions to retrieve (ignored)
            
        Returns:
            Always returns an empty list
        """
        logger.debug(f"NoCheckpointMemory: Returning empty relevant history for session {session_id}")
        return []
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Simulate clearing a session (no-op since nothing is stored).
        
        Args:
            session_id: Session identifier to clear (ignored)
            
        Returns:
            Always returns True
        """
        logger.debug(f"NoCheckpointMemory: Simulated clearing session {session_id}")
        return True
    
    async def get_sessions_for_soeid(self, soeid: str) -> List[Dict[str, Any]]:
        """
        Return empty sessions list for SOEID.
        
        Args:
            soeid: User identifier (ignored)
            
        Returns:
            Always returns an empty list
        """
        logger.debug(f"NoCheckpointMemory: Returning empty sessions for SOEID {soeid}")
        return []
    
    async def get_chat_history_by_soeid_and_date(self, 
                                               soeid: str, 
                                               days: int = 7, 
                                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return empty chat history for SOEID and date range.
        
        Args:
            soeid: User identifier (ignored)
            days: Number of days to look back (ignored)
            limit: Maximum number of messages (ignored)
            
        Returns:
            Always returns an empty list
        """
        logger.debug(f"NoCheckpointMemory: Returning empty chat history for SOEID {soeid}")
        return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Return memory statistics (all zeros for no-checkpoint memory).
        
        Returns:
            Dictionary with zero statistics
        """
        return {
            "total_sessions": 0,
            "total_messages": 0,
            "unique_soeids": 0,
            "memory_type": "no_checkpoint",
            "storage_enabled": self._enabled,
            "oldest_session": None,
            "newest_session": None,
            "average_messages_per_session": 0
        }
    
    def is_enabled(self) -> bool:
        """
        Check if memory is enabled.
        
        Returns:
            Boolean indicating if memory operations are enabled
        """
        return self._enabled
    
    def disable(self) -> None:
        """Disable memory operations."""
        self._enabled = False
        logger.info("NoCheckpointMemory disabled")
    
    def enable(self) -> None:
        """Enable memory operations."""
        self._enabled = True
        logger.info("NoCheckpointMemory enabled")
