"""LangGraph memory implementation for the RAG system."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import uuid
import json

# Import LangGraph store components with fallback
try:
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import BaseStore
    LANGGRAPH_STORE_AVAILABLE = True
except ImportError:
    LANGGRAPH_STORE_AVAILABLE = False
    InMemoryStore = None
    BaseStore = None

# Try to import PostgreSQL store, but handle gracefully if not available
try:
    from langgraph.store.postgres import PostgresStore
    POSTGRES_STORE_AVAILABLE = True
except ImportError:
    PostgresStore = None
    POSTGRES_STORE_AVAILABLE = False

from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)

class LangGraphMemory(BaseMemory):
    """LangGraph memory implementation using LangGraph stores.
    
    This implementation supports both in-memory and PostgreSQL storage
    for conversation history and long-term memory with SOEID support.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LangGraph memory system with configuration.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        super().__init__(config)
        
        # Check if LangGraph store is available
        if not LANGGRAPH_STORE_AVAILABLE:
            error_msg = "LangGraph store is not available. Please install langgraph with store support."
            logger.error(error_msg)
            raise MemoryError(error_msg)
        
        # Configuration
        self._config = config or {}
        self._store_type = self._config.get("store_type", "in_memory")
        self._embedding_function = self._config.get("embedding_function")
        self._embedding_dimensions = self._config.get("embedding_dimensions", 384)
        
        # Initialize the appropriate store
        self._store = self._init_store()
        
        # Session management
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"LangGraph memory system initialized with {self._store_type} store")
    
    def _init_store(self) -> BaseStore:
        """Initialize the appropriate LangGraph store.
        
        Returns:
            BaseStore: The initialized store
        """
        try:
            if self._store_type == "in_memory":
                return self._init_in_memory_store()
            elif self._store_type == "postgres":
                if not POSTGRES_STORE_AVAILABLE:
                    # Log warning and fallback to in-memory store instead of raising an error
                    logger.warning("PostgreSQL store is not available. Falling back to in-memory store.")
                    self._store_type = "in_memory"  # Update the store type to reflect reality
                    return self._init_in_memory_store()
                return self._init_postgres_store()
            else:
                raise MemoryError(f"Unsupported store type: {self._store_type}")
        except Exception as e:
            error_msg = f"Failed to initialize {self._store_type} store: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e
    
    def _init_in_memory_store(self) -> InMemoryStore:
        """Initialize in-memory store.
        
        Returns:
            InMemoryStore: The in-memory store
        """
        # Create a simple embedding function if none provided
        if not self._embedding_function:
            def simple_embed(texts: List[str]) -> List[List[float]]:
                # Simple hash-based embedding for demo purposes
                # In production, use a proper embedding model
                import hashlib
                embeddings = []
                for text in texts:
                    hash_obj = hashlib.md5(text.encode())
                    # Create a simple embedding vector
                    embedding = [float(hash_obj.digest()[i % 16]) / 255.0 for i in range(self._embedding_dimensions)]
                    embeddings.append(embedding)
                return embeddings
            
            self._embedding_function = simple_embed
        
        return InMemoryStore(
            index={
                "embed": self._embedding_function,
                "dims": self._embedding_dimensions
            }
        )
    
    def _init_postgres_store(self) -> PostgresStore:
        """Initialize PostgreSQL store.
        
        Returns:
            PostgresStore: The PostgreSQL store
        """
        # Get PostgreSQL configuration
        pg_config = self._config.get("postgres", {})
        connection_string = pg_config.get("connection_string")
        
        if not connection_string:
            raise MemoryError("PostgreSQL connection string is required for postgres store type")
        
        # Create embedding function if not provided
        if not self._embedding_function:
            def simple_embed(texts: List[str]) -> List[List[float]]:
                # Simple hash-based embedding for demo purposes
                # In production, use a proper embedding model
                import hashlib
                embeddings = []
                for text in texts:
                    hash_obj = hashlib.md5(text.encode())
                    # Create a simple embedding vector
                    embedding = [float(hash_obj.digest()[i % 16]) / 255.0 for i in range(self._embedding_dimensions)]
                    embeddings.append(embedding)
                return embeddings
            
            self._embedding_function = simple_embed
        
        return PostgresStore(
            connection_string=connection_string,
            index={
                "embed": self._embedding_function,
                "dims": self._embedding_dimensions
            }
        )
    
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
        try:
            async with self._lock:
                # Extract SOEID from metadata if available
                soeid = metadata.get("soeid") if metadata else None
                
                # Create namespace for this session
                namespace = (session_id, "conversation")
                
                # Create interaction record
                interaction_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()
                
                interaction_data = {
                    "session_id": session_id,
                    "soeid": soeid,
                    "query": query,
                    "response": response,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                    "interaction_id": interaction_id
                }
                
                # Store the interaction
                self._store.put(namespace, interaction_id, interaction_data)
                
                # Also store as separate messages for easier retrieval
                user_message_id = f"{interaction_id}_user"
                assistant_message_id = f"{interaction_id}_assistant"
                
                user_message = {
                    "session_id": session_id,
                    "soeid": soeid,
                    "role": "user",
                    "content": query,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                    "message_id": user_message_id
                }
                
                assistant_message = {
                    "session_id": session_id,
                    "soeid": soeid,
                    "role": "assistant", 
                    "content": response,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                    "message_id": assistant_message_id
                }
                
                # Store messages separately
                self._store.put(namespace, user_message_id, user_message)
                self._store.put(namespace, assistant_message_id, assistant_message)
                
                # If SOEID is provided, also store in user-specific namespace
                if soeid:
                    user_namespace = (soeid, "user_conversations")
                    self._store.put(user_namespace, interaction_id, interaction_data)
                    self._store.put(user_namespace, user_message_id, user_message)
                    self._store.put(user_namespace, assistant_message_id, assistant_message)
                    logger.info(f"Added interaction {interaction_id} to SOEID namespace {user_namespace}")
                else:
                    logger.warning(f"No SOEID provided for interaction {interaction_id}, not storing in SOEID namespace.")
                
                logger.debug(f"Added interaction {interaction_id} to session {session_id} with SOEID {soeid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add interaction to LangGraph memory: {str(e)}", exc_info=True)
            return False
    
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
        try:
            async with self._lock:
                namespace = (session_id, "conversation")
                
                # Get all messages for this session
                messages = self._store.search(namespace, query="", limit=limit or self.max_history)
                
                # Sort by timestamp
                sorted_messages = sorted(messages, key=lambda x: x.value.get("timestamp", ""))
                
                # Convert to standard format
                history = []
                for msg in sorted_messages:
                    history.append({
                        "role": msg.value.get("role", "user"),
                        "content": msg.value.get("content", ""),
                        "timestamp": msg.value.get("timestamp", ""),
                        "soeid": msg.value.get("soeid"),
                        "metadata": msg.value.get("metadata", {})
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}", exc_info=True)
            return []
    
    async def _get_relevant_history(self, 
                                  session_id: str, 
                                  query: str, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history using semantic search.
        
        Args:
            session_id: Unique session identifier
            query: Current query to find relevant history for
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of relevant conversation interactions
        """
        try:
            async with self._lock:
                namespace = (session_id, "conversation")
                
                # Use semantic search to find relevant messages
                relevant_messages = self._store.search(
                    namespace, 
                    query=query, 
                    limit=limit or self.max_history
                )
                
                # Convert to standard format
                history = []
                for msg in relevant_messages:
                    history.append({
                        "role": msg.value.get("role", "user"),
                        "content": msg.value.get("content", ""),
                        "timestamp": msg.value.get("timestamp", ""),
                        "soeid": msg.value.get("soeid"),
                        "metadata": msg.value.get("metadata", {})
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get relevant history: {str(e)}", exc_info=True)
            # Fall back to regular history
            return await self._get_conversation_history(session_id, limit)
    
    async def get_user_history_by_soeid(self, soeid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all sessions (with all messages) where any message in the session has the given SOEID."""
        try:
            async with self._lock:
                # Gather all session_ids from namespaces or known sessions
                session_ids = set()
                if hasattr(self._store, 'namespaces'):
                    for ns in self._store.namespaces():
                        if isinstance(ns, tuple) and len(ns) == 2 and ns[1] == "conversation":
                            session_ids.add(ns[0])
                else:
                    session_ids = set(getattr(self, '_sessions', {}).keys())
                
                sessions = []
                total_messages = 0
                for session_id in session_ids:
                    session_namespace = (session_id, "conversation")
                    try:
                        msgs = self._store.search(session_namespace, query="", limit=10000)
                        # Convert to dicts
                        msg_dicts = [
                            {
                                "role": msg.value.get("role", "user"),
                                "content": msg.value.get("content", ""),
                                "timestamp": msg.value.get("timestamp", ""),
                                "soeid": msg.value.get("soeid"),
                                "session_id": msg.value.get("session_id"),
                                "metadata": msg.value.get("metadata", {})
                            }
                            for msg in msgs
                        ]
                        # If any message in this session has the SOEID, include the whole session
                        if any(m.get("soeid") == soeid for m in msg_dicts):
                            sessions.append({
                                "session_id": session_id,
                                "messages": msg_dicts
                            })
                            total_messages += len(msg_dicts)
                    except Exception as e:
                        logger.warning(f"Error searching session {session_id}: {e}")
                return sessions
        except Exception as e:
            logger.error(f"Failed to get user history by SOEID: {str(e)}", exc_info=True)
            return []
    
    async def get_user_relevant_history_by_soeid(self, soeid: str, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history for a specific user by SOEID.
        
        Args:
            soeid: Source of Entity ID for the user
            query: Current query to find relevant history for
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of relevant conversation interactions for the user
        """
        try:
            async with self._lock:
                user_namespace = (soeid, "user_conversations")
                
                # Use semantic search to find relevant messages for this user
                relevant_messages = self._store.search(
                    user_namespace, 
                    query=query, 
                    limit=limit or self.max_history
                )
                
                # Convert to standard format
                history = []
                for msg in relevant_messages:
                    history.append({
                        "role": msg.value.get("role", "user"),
                        "content": msg.value.get("content", ""),
                        "timestamp": msg.value.get("timestamp", ""),
                        "soeid": msg.value.get("soeid"),
                        "session_id": msg.value.get("session_id"),
                        "metadata": msg.value.get("metadata", {})
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get user relevant history by SOEID: {str(e)}", exc_info=True)
            # Fall back to regular user history
            return await self.get_user_history_by_soeid(soeid, limit)
    
    async def add(self, 
                session_id: str = None,
                query: str = None, 
                response: str = None, 
                metadata: Optional[Dict[str, Any]] = None,
                user_id: str = None,
                messages: List[Dict[str, str]] = None) -> bool:
        """Add messages to the conversation memory.
        
        Args:
            session_id: Session identifier
            query: User query
            response: System response
            metadata: Additional metadata (can include SOEID)
            user_id: Legacy user identifier parameter
            messages: Legacy messages parameter
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle different parameter styles
            if session_id is not None and (query is not None or response is not None):
                # New style parameters
                return await self._add_interaction(session_id, query, response, metadata)
                
            elif messages and user_id:
                # Legacy style parameters
                session_id = user_id
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "user":
                        await self._add_interaction(session_id, content, "", metadata)
                    elif role == "assistant":
                        # For assistant messages without a user query, create a placeholder
                        await self._add_interaction(session_id, "", content, metadata)
                
                return True
            else:
                logger.warning("Invalid parameters provided to add method")
                return False
                
        except Exception as e:
            logger.error(f"Error adding to LangGraph memory: {str(e)}", exc_info=True)
            return False
    
    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation interactions
        """
        return await self._get_conversation_history(session_id, limit)
    
    async def get_relevant_history(self, session_id: str, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history.
        
        Args:
            session_id: Unique session identifier
            query: Current query to find relevant history for
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of relevant conversation interactions
        """
        return await self._get_relevant_history(session_id, query, limit)
    
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a user (legacy interface).
        
        Args:
            user_id: User identifier (used as session_id)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        try:
            # Get conversation history
            history = await self.get_history(user_id, limit)
            
            # Convert to legacy format
            messages = []
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            return messages
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}", exc_info=True)
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False if session not found
        """
        try:
            async with self._lock:
                namespace = (session_id, "conversation")
                
                # Get all items in the namespace
                items = self._store.search(namespace, query="")
                
                # Delete all items
                for item in items:
                    self._store.delete(namespace, item.key)
                
                logger.info(f"Cleared session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {str(e)}", exc_info=True)
            return False
    
    async def clear_user_history(self, soeid: str) -> bool:
        """Clear all conversation history for a specific user by SOEID.
        
        Args:
            soeid: Source of Entity ID for the user
            
        Returns:
            True if successful, False if user not found
        """
        try:
            async with self._lock:
                user_namespace = (soeid, "user_conversations")
                
                # Get all items in the user namespace
                items = self._store.search(user_namespace, query="")
                
                # Delete all items
                for item in items:
                    self._store.delete(user_namespace, item.key)
                
                logger.info(f"Cleared user history for SOEID {soeid}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear user history for SOEID {soeid}: {str(e)}", exc_info=True)
            return False
    
    async def add_long_term_memory(self, 
                                 namespace: tuple, 
                                 key: str, 
                                 data: Dict[str, Any]) -> bool:
        """Add long-term memory using LangGraph store.
        
        Args:
            namespace: Memory namespace (e.g., (user_id, "preferences"))
            key: Memory key
            data: Memory data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._store.put(namespace, key, data)
            logger.debug(f"Added long-term memory {key} to namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to add long-term memory: {str(e)}", exc_info=True)
            return False
    
    async def get_long_term_memory(self, namespace: tuple, key: str) -> Optional[Dict[str, Any]]:
        """Get long-term memory from LangGraph store.
        
        Args:
            namespace: Memory namespace
            key: Memory key
            
        Returns:
            Memory data if found, None otherwise
        """
        try:
            items = self._store.get(namespace, key)
            if items:
                # Handle both single item and list of items
                if isinstance(items, list) and len(items) > 0:
                    item = items[0]
                else:
                    item = items
                
                # Try different ways to access the value
                if hasattr(item, 'value'):
                    return item.value
                elif hasattr(item, 'data'):
                    return item.data
                elif isinstance(item, dict):
                    return item
                else:
                    # If it's a simple object, try to convert to dict
                    return dict(item.__dict__) if hasattr(item, '__dict__') else None
            return None
        except Exception as e:
            logger.error(f"Failed to get long-term memory: {str(e)}", exc_info=True)
            return None
    
    async def search_long_term_memory(self, 
                                    namespace: tuple, 
                                    query: str = "", 
                                    filter_dict: Optional[Dict[str, Any]] = None,
                                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search long-term memory using LangGraph store.
        
        Args:
            namespace: Memory namespace
            query: Search query
            filter_dict: Filter criteria
            limit: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        try:
            items = self._store.search(namespace, query=query, filter=filter_dict, limit=limit)
            return [item.value for item in items]
        except Exception as e:
            logger.error(f"Failed to search long-term memory: {str(e)}", exc_info=True)
            return [] 