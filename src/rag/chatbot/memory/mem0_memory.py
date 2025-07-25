"""Mem0 memory implementation for the RAG system."""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from datetime import datetime
import asyncio
import uuid

# Import conditionally to avoid startup errors if mem0 is not installed
try:
    import mem0
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    
from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)

class Mem0Memory(BaseMemory):
    """Memory implementation using mem0 for local storage.
    
    This implementation uses mem0's local storage capabilities for chat history,
    without requiring the hosted API.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the mem0 memory system with configuration.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        super().__init__(config)
        
        if not MEM0_AVAILABLE:
            error_msg = "mem0 package is not installed. Please install with 'pip install mem0'"
            logger.error(error_msg)
            raise MemoryError(error_msg)
        
        # Configuration
        self._config = config or {}
        
        # Initialize mem0 client for local storage
        self._client = self._init_client()
        self._collection_name = self._config.get("collection_name", "chat_history")
        
        # Create the collection if it doesn't exist
        self._ensure_collection_exists()
        
        logger.info("Mem0 memory system initialized with local storage")
    
    def _init_client(self):
        """Initialize the mem0 client for local storage.
        
        Returns:
            mem0 client
        """
        try:
            # Use mem0 in local mode
            storage_path = self._config.get("storage_path")
            if not storage_path:
                # Default to a directory in the project
                base_dir = Path(__file__).resolve().parents[3]  # Go up to project root
                storage_path = os.path.join(base_dir, 'data', 'mem0')
                
            # Ensure the directory exists
            os.makedirs(storage_path, exist_ok=True)
            
            # Initialize mem0 client with local storage
            client = mem0.Client(local_storage_path=storage_path)
            logger.info(f"Initialized mem0 client with local storage at {storage_path}")
            return client
        except Exception as e:
            error_msg = f"Failed to initialize mem0 client: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e
    
    def _ensure_collection_exists(self):
        """Ensure the mem0 collection exists."""
        try:
            collections = self._client.list_collections()
            if self._collection_name not in collections:
                self._client.create_collection(self._collection_name)
                logger.info(f"Created mem0 collection: {self._collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring mem0 collection exists: {str(e)}")
    
    async def add(self, session_id: str = None, query: str = None, response: str = None, 
                 metadata: Dict[str, Any] = None, user_id: str = None, 
                 messages: List[Dict[str, str]] = None) -> None:
        """Add messages to the conversation memory.
        
        Args:
            session_id: Session identifier 
            query: User query
            response: System response
            metadata: Additional metadata
            user_id: Legacy user identifier parameter
            messages: Legacy messages parameter
        """
        try:
            # Handle different parameter styles
            if session_id is not None and (query is not None or response is not None):
                # New style parameters
                timestamp = datetime.now().isoformat()
                
                # Add user query if provided
                if query:
                    user_message = {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "role": "user",
                        "content": query,
                        "timestamp": timestamp,
                        "metadata": metadata or {}
                    }
                    self._client.add_memory(
                        collection=self._collection_name,
                        memory_id=user_message["id"],
                        memory=user_message
                    )
                    logger.debug(f"Added user message to mem0 for session {session_id}")
                
                # Add system response if provided
                if response:
                    system_message = {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp,
                        "metadata": metadata or {}
                    }
                    self._client.add_memory(
                        collection=self._collection_name,
                        memory_id=system_message["id"],
                        memory=system_message
                    )
                    logger.debug(f"Added system response to mem0 for session {session_id}")
                    
            # Legacy style parameters
            elif user_id is not None and messages:
                session_id = user_id  # Use user_id as session_id for legacy calls
                timestamp = datetime.now().isoformat()
                
                # Add each message to mem0
                for message in messages:
                    mem0_message = {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                        "timestamp": timestamp,
                        "metadata": metadata or {}
                    }
                    self._client.add_memory(
                        collection=self._collection_name,
                        memory_id=mem0_message["id"],
                        memory=mem0_message
                    )
                logger.debug(f"Added {len(messages)} legacy messages to mem0 for session {session_id}")
            else:
                logger.warning("Invalid parameters provided to Mem0Memory.add()")
                
        except Exception as e:
            error_msg = f"Failed to add messages to mem0: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e
    
    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
            
        Raises:
            MemoryError: If retrieving history fails
        """
        try:
            # Query mem0 for messages with this session_id
            query = {
                "session_id": session_id
            }
            
            # Get all messages for this session
            results = self._client.query_collection(
                collection=self._collection_name,
                query=query
            )
            
            # Sort messages by timestamp
            messages = sorted(
                results,
                key=lambda x: x.get("timestamp", "")
            )
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                messages = messages[-limit:]
                
            logger.debug(f"Retrieved {len(messages)} messages from mem0 for session {session_id}")
            return messages
            
        except Exception as e:
            error_msg = f"Failed to retrieve conversation history from mem0: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Query for all messages with this session_id
            query = {
                "session_id": session_id
            }
            
            # Get message IDs to delete
            results = self._client.query_collection(
                collection=self._collection_name,
                query=query
            )
            
            # Delete each message
            for message in results:
                self._client.delete_memory(
                    collection=self._collection_name,
                    memory_id=message["id"]
                )
            
            logger.info(f"Cleared {len(results)} messages for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session in mem0: {str(e)}")
            return False
    
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Legacy method to get recent conversation history for a user.
        
        Args:
            user_id: User identifier (used as session_id)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with role and content
        """
        # Map to the new method, treating user_id as session_id
        history = await self.get_history(user_id, limit)
        
        # Convert to the simpler format expected by older interfaces
        return [{
            "role": msg["role"], 
            "content": msg["content"]
        } for msg in history]
