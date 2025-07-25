"""Simple in-memory conversation memory system."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from src.rag.src.chatbot.memory.base_memory import BaseMemory

logger = logging.getLogger(__name__)

class SimpleMemory(BaseMemory):
    """Simple in-memory conversation memory system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the simple memory system with configuration.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        super().__init__(config)
        # Dictionary to store session conversations: {session_id: [messages]}
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()  # For thread safety
        
    async def add(self, user_id: str = None, messages: List[Dict[str, str]] = None, session_id: str = None, query: str = None, response: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Add messages to the conversation memory.
        
        Args:
            user_id: User identifier (legacy parameter)
            messages: List of message dictionaries with 'role' and 'content' keys (legacy parameter)
            session_id: Session identifier (new parameter)
            query: User query (new parameter)
            response: System response (new parameter)
            metadata: Additional metadata (new parameter)
        """
        async with self._lock:
            # Handle both old and new parameter styles
            if session_id is not None:
                # New style parameters
                user_id = session_id
                
                # Create messages from query and response
                generated_messages = []
                if query:
                    generated_messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    })
                
                if response:
                    generated_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    })
                
                # Ensure session exists
                if user_id not in self._sessions:
                    self._sessions[user_id] = []
                
                # Add generated messages
                self._sessions[user_id].extend(generated_messages)
                logger.debug(f"Added {len(generated_messages)} messages to session {user_id}")
                return True
            
            elif messages:
                # Old style parameters
                if user_id not in self._sessions:
                    self._sessions[user_id] = []
                    
                # Add all messages to the session
                for message in messages:
                    self._sessions[user_id].append({
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    })
                logger.debug(f"Added {len(messages)} messages to session {user_id}")
                return True
            else:
                logger.warning("No valid parameters provided to SimpleMemory.add()")
                return False
    
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        history = await self._get_conversation_history(user_id, limit)
        # Convert to the simpler format expected by the interface
        return [{
            "role": msg["role"], 
            "content": msg["content"]
        } for msg in history]
    
    async def _add_interaction(self, 
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
        """
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            
            # Create interaction records for both query and response
            timestamp = datetime.now().isoformat()
            
            # User message
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            
            # System response
            system_message = {
                "role": "assistant",
                "content": response,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            
            # Add to session history
            self._sessions[session_id].append(user_message)
            self._sessions[session_id].append(system_message)
            
            return True
    
    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session. Implementation of BaseMemory.get_history.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation interactions including role, content, timestamp and metadata
        """
        logger.debug(f"Retrieving history for session {session_id} with limit {limit}")
        return await self._get_conversation_history(session_id, limit)
        
    async def _get_conversation_history(self, 
                                      session_id: str, 
                                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation interactions
        """
        async with self._lock:
            if session_id not in self._sessions:
                logger.debug(f"Session {session_id} not found in memory")
                return []
            
            history = self._sessions[session_id]
            logger.debug(f"Found {len(history)} messages in session {session_id}")
            
            # Apply limit if specified
            if limit is not None:
                history = history[-limit:]
            
            return history
    
    async def _get_relevant_history(self, 
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
        """
        # For simple implementation, just return the latest history
        # A more sophisticated implementation would use semantic similarity to find relevant messages
        return await self._get_conversation_history(session_id, limit)
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False if session not found
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False 