"""Enhanced LangGraph memory implementation using checkpointers for conversation history."""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import LangGraph checkpoint components following playbook patterns
try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph import StateGraph
    LANGGRAPH_CHECKPOINT_AVAILABLE = True
    POSTGRES_CHECKPOINT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangGraph checkpoint components not available: {e}")
    LANGGRAPH_CHECKPOINT_AVAILABLE = False
    POSTGRES_CHECKPOINT_AVAILABLE = False
    InMemorySaver = None
    PostgresSaver = None
    BaseCheckpointSaver = None
    StateGraph = None

# Import LangGraph store components for long-term memory
try:
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import BaseStore
    LANGGRAPH_STORE_AVAILABLE = True
except ImportError:
    LANGGRAPH_STORE_AVAILABLE = False
    InMemoryStore = None
    BaseStore = None

try:
    from langgraph.store.postgres import PostgresStore
    POSTGRES_STORE_AVAILABLE = True
except ImportError:
    PostgresStore = None
    POSTGRES_STORE_AVAILABLE = False

from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)


class LangGraphCheckpointMemory(BaseMemory):
    """Enhanced LangGraph memory implementation following the official playbook patterns.
    
    This implementation follows LangGraph best practices:
    - Uses InMemorySaver for development and PostgresSaver for production
    - Implements proper thread-scoped memory with configurable thread_id
    - Supports memory toggle via configuration (enabled/disabled)
    - Follows the playbook patterns for checkpoint management
    - Provides fallback to NoCheckpointMemory when disabled
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LangGraph checkpoint memory system following playbook patterns.
        
        Args:
            config: Configuration dictionary for the memory system
        """
        super().__init__(config)
        
        # Configuration
        self._config = config or {}
        
        # Check if memory is enabled (following playbook pattern for disabling memory)
        if not self._config.get("enabled", True):
            logger.info("LangGraph checkpoint memory is disabled via configuration")
            return
        
        # Check if LangGraph checkpoint is available
        if not LANGGRAPH_CHECKPOINT_AVAILABLE:
            error_msg = "LangGraph checkpoint is not available. Please install langgraph with checkpoint support."
            logger.error(error_msg)
            raise MemoryError(error_msg)
        
        self._store_type = self._config.get("store_type", "in_memory")
        self._postgres_config = self._config.get("postgres", {})
        
        # Initialize checkpointer for conversation history (following playbook patterns)
        self._checkpointer = self._init_checkpointer()
        
        # Initialize store for long-term memory (if available)
        self._store = None
        if LANGGRAPH_STORE_AVAILABLE:
            self._store = self._init_store()
        
        # Session management
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"LangGraph checkpoint memory system initialized with {self._store_type} storage following playbook patterns")
    
    def _init_checkpointer(self) -> BaseCheckpointSaver:
        """Initialize the appropriate LangGraph checkpointer.
        
        Returns:
            BaseCheckpointSaver: The initialized checkpointer
        """
        try:
            if self._store_type == "in_memory":
                return InMemorySaver()
            elif self._store_type == "postgres":
                if not POSTGRES_CHECKPOINT_AVAILABLE:
                    logger.warning("PostgreSQL checkpointer is not available. Falling back to in-memory checkpointer.")
                    self._store_type = "in_memory"
                    return InMemorySaver()
                
                # Get PostgreSQL connection string
                connection_string = self._postgres_config.get("connection_string")
                if not connection_string:
                    raise MemoryError("PostgreSQL connection string is required for postgres store type")
                
                # For PostgreSQL, we need to use the context manager for each operation
                # Store the connection string for creating connections as needed
                self._postgres_connection_string = connection_string
                
                # Test the connection and set up schema
                with PostgresSaver.from_conn_string(connection_string) as postgres_saver:
                    try:
                        postgres_saver.setup()
                        logger.info("PostgreSQL database schema setup completed")
                    except Exception as setup_error:
                        logger.warning(f"PostgreSQL schema setup failed: {setup_error}")
                        # Continue anyway, tables might already exist
                    
                    logger.info(f"PostgreSQL checkpointer initialized: {type(postgres_saver)}")
                    logger.info("PostgreSQL connection tested successfully")
                
                # Return a placeholder - we'll use context managers for actual operations
                return "postgres_checkpointer"
            else:
                raise MemoryError(f"Unsupported store type: {self._store_type}")
        except Exception as e:
            error_msg = f"Failed to initialize {self._store_type} checkpointer: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg) from e
    
    def _init_store(self) -> Optional[BaseStore]:
        """Initialize the appropriate LangGraph store for long-term memory.
        
        Returns:
            BaseStore: The initialized store or None if not available
        """
        try:
            if self._store_type == "in_memory":
                return self._init_in_memory_store()
            elif self._store_type == "postgres":
                if not POSTGRES_STORE_AVAILABLE:
                    logger.warning("PostgreSQL store is not available. Using in-memory store for long-term memory.")
                    return self._init_in_memory_store()
                return self._init_postgres_store()
            else:
                logger.warning(f"Unsupported store type for long-term memory: {self._store_type}")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize store for long-term memory: {str(e)}")
            return None
    
    def _init_in_memory_store(self) -> InMemoryStore:
        """Initialize in-memory store for long-term memory.
        
        Returns:
            InMemoryStore: The in-memory store
        """
        return InMemoryStore()
    
    def _init_postgres_store(self) -> PostgresStore:
        """Initialize PostgreSQL store for long-term memory.
        
        Returns:
            PostgresStore: The PostgreSQL store
        """
        connection_string = self._postgres_config.get("connection_string")
        if not connection_string:
            raise MemoryError("PostgreSQL connection string is required for postgres store type")
        
        return PostgresStore.from_conn_string(connection_string)
    
    async def _get_thread_config(self, session_id: str) -> Dict[str, Any]:
        """Get thread configuration for a session following LangGraph patterns.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Thread configuration dictionary with proper configurable structure
        """
        return {
            "configurable": {
                "thread_id": session_id
            }
        }
    
    async def _add_message_to_checkpoint(self, 
                                       session_id: str, 
                                       role: str, 
                                       content: str, 
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to the checkpoint for a session.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message data
            message_data = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "metadata": metadata or {}
            }
            
            # Add SOEID to message if available in metadata
            if metadata and "soeid" in metadata:
                message_data["soeid"] = metadata["soeid"]
            
            # Get thread configuration
            config = await self._get_thread_config(session_id)
            
            # For checkpointers, we need to create a simple state structure
            # This is a simplified approach - in a real implementation you'd use a proper LangGraph graph
            state = {
                "messages": [message_data],
                "session_id": session_id,
                "last_updated": datetime.now().isoformat()
            }
            
            # Get existing messages from PostgreSQL first (for persistence across restarts)
            existing_messages = await self._get_messages_from_checkpoint(session_id)
            logger.debug(f"Retrieved {len(existing_messages)} existing messages from PostgreSQL for session {session_id}")
            
            # Append new message to existing messages
            all_messages = existing_messages + [message_data]
            
            # Update internal tracking for performance
            session_key = f"session_{session_id}"
            if not hasattr(self, '_session_messages'):
                self._session_messages = {}
            self._session_messages[session_key] = all_messages
            
            # Create checkpoint data following LangGraph patterns
            checkpoint_data = {
                "messages": all_messages,
                "session_id": session_id,
                "last_updated": datetime.now().isoformat()
            }
            
            # Store the checkpoint using LangGraph checkpointer patterns
            if self._store_type == "postgres":
                with PostgresSaver.from_conn_string(self._postgres_connection_string) as checkpointer:
                    # Use proper LangGraph checkpoint structure
                    checkpoint = {
                        "v": 1,
                        "ts": datetime.now().isoformat(),
                        "id": str(uuid.uuid4()),
                        "channel_values": checkpoint_data,
                        "channel_versions": {"messages": len(all_messages)},
                        "versions_seen": {}
                    }
                    checkpointer.put(config, checkpoint, {}, {})
                    logger.debug(f"Successfully stored checkpoint in PostgreSQL for session {session_id}")
            else:
                checkpoint = {
                    "v": 1,
                    "ts": datetime.now().isoformat(),
                    "id": str(uuid.uuid4()),
                    "channel_values": checkpoint_data,
                    "channel_versions": {"messages": len(all_messages)},
                    "versions_seen": {}
                }
                self._checkpointer.put(config, checkpoint, {}, {})
                logger.debug(f"Successfully stored checkpoint in in-memory storage for session {session_id}")
            
            # Update internal cache for performance (but don't rely on it for persistence)
            if not hasattr(self, '_session_messages'):
                self._session_messages = {}
            session_key = f"session_{session_id}"
            self._session_messages[session_key] = all_messages
            
            logger.info(f"Added message to checkpoint for session {session_id}, total messages: {len(all_messages)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to checkpoint: {str(e)}", exc_info=True)
            return False
    
    async def _get_messages_from_checkpoint(self, 
                                          session_id: str, 
                                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from checkpoint for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        try:
            # Always try to get from persistent storage first
            config = await self._get_thread_config(session_id)
            logger.debug(f"Getting messages for session {session_id} with config: {config}")
            
            checkpoint = None
            # Use context manager for PostgreSQL
            if self._store_type == "postgres":
                with PostgresSaver.from_conn_string(self._postgres_connection_string) as checkpointer:
                    checkpoint = checkpointer.get(config)
                    logger.debug(f"Retrieved checkpoint from PostgreSQL: {checkpoint is not None}")
            else:
                checkpoint = self._checkpointer.get(config)
                logger.debug(f"Retrieved checkpoint from in-memory: {checkpoint is not None}")
            
            # If we got data from persistent storage, return it
            if checkpoint:
                # Handle both dict and object formats
                channel_values = None
                if isinstance(checkpoint, dict):
                    channel_values = checkpoint.get('channel_values', {})
                elif hasattr(checkpoint, 'channel_values'):
                    channel_values = checkpoint.channel_values
                
                if channel_values:
                    messages = channel_values.get("messages", [])
                    if isinstance(messages, list):
                        logger.info(f"Retrieved {len(messages)} messages from persistent storage for session {session_id}")
                        
                        # Update cache for performance
                        if not hasattr(self, '_session_messages'):
                            self._session_messages = {}
                        session_key = f"session_{session_id}"
                        self._session_messages[session_key] = messages
                        
                        # Apply limit if specified
                        if limit and len(messages) > limit:
                            messages = messages[-limit:]
                        
                        return messages
            
            # No data found in persistent storage
            logger.info(f"No messages found in persistent storage for session {session_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get messages from checkpoint: {str(e)}", exc_info=True)
            return []
    
    async def _list_thread_ids(self) -> List[str]:
        """List all thread IDs from checkpointer.
        
        Returns:
            List of thread IDs
        """
        try:
            # First try to get from our internal storage
            if hasattr(self, '_simple_storage'):
                thread_ids = list(self._simple_storage.keys())
                logger.debug(f"Found {len(thread_ids)} threads in internal storage")
                return thread_ids
            
            # Fallback to checkpoint listing (using sync method since async not implemented)
            logger.warning("Checkpoint listing not available with PostgreSQL checkpointer")
            return []
            
        except Exception as e:
            logger.error(f"Failed to list thread IDs: {str(e)}", exc_info=True)
            return []
    
    async def _add_interaction(self, 
                             session_id: str,
                             query: str, 
                             response: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a single interaction to the conversation memory.
        
        Args:
            session_id: Session identifier
            query: User query
            response: System response
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                success = True
                
                # Add user message
                if query:
                    result = await self._add_message_to_checkpoint(
                        session_id, "user", query, metadata
                    )
                    if not result:
                        success = False
                
                # Add assistant message
                if response:
                    result = await self._add_message_to_checkpoint(
                        session_id, "assistant", response, metadata
                    )
                    if not result:
                        success = False
                
                # Update session tracking
                if session_id not in self._sessions:
                    self._sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat(),
                        "message_count": 0
                    }
                
                self._sessions[session_id]["last_active"] = datetime.now().isoformat()
                self._sessions[session_id]["message_count"] += (1 if query else 0) + (1 if response else 0)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to add interaction to memory: {str(e)}", exc_info=True)
            return False
    
    async def _get_conversation_history(self, 
                                      session_id: str, 
                                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        try:
            messages = await self._get_messages_from_checkpoint(session_id, limit)
            return messages
        except Exception as e:
            logger.error(f"Failed to get conversation history for session {session_id}: {str(e)}", exc_info=True)
            return []
    
    async def _get_relevant_history(self, 
                                  session_id: str, 
                                  query: str, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history.
        
        Args:
            session_id: Session identifier
            query: Current query to find relevant history for
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of relevant conversation messages
        """
        # For now, return recent history
        # In a more sophisticated implementation, you could use semantic search
        return await self._get_conversation_history(session_id, limit)
    
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a user (Memory interface method).
        
        Args:
            user_id: User identifier (treated as session_id)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            messages = await self.get_history(user_id, limit)
            # Convert to the expected format for the Memory interface
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            return formatted_messages
        except Exception as e:
            logger.error(f"Failed to get messages for user {user_id}: {str(e)}", exc_info=True)
            return []
    
    async def _add_messages_list(self, 
                               session_id: str, 
                               messages: List[Dict[str, str]], 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a list of messages to the conversation memory.
        
        Args:
            session_id: Session identifier
            messages: List of messages with 'role' and 'content' keys
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                success = True
                
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    msg_metadata = dict(metadata) if metadata else {}
                    
                    result = await self._add_message_to_checkpoint(
                        session_id, role, content, msg_metadata
                    )
                    if not result:
                        success = False
                
                # Update session tracking
                if session_id not in self._sessions:
                    self._sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat(),
                        "message_count": 0
                    }
                
                self._sessions[session_id]["last_active"] = datetime.now().isoformat()
                self._sessions[session_id]["message_count"] += len(messages)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to add messages list to memory: {str(e)}", exc_info=True)
            return False
    
    async def add(self, 
                  session_id: str = None,
                  user_id: str = None,
                  messages: List[Dict[str, str]] = None,
                  query: str = None, 
                  response: str = None, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add messages to the conversation memory.
        
        This method supports both the Memory interface and the extended RAG interface.
        
        Args:
            session_id: Session identifier (for RAG interface)
            user_id: User identifier (for Memory interface, used as session_id if session_id not provided)
            messages: List of messages to add directly (for Memory interface)
            query: User query (for RAG interface)
            response: System response (for RAG interface)
            metadata: Additional metadata (can include SOEID)
            
        Returns:
            True if successful, False otherwise
        """
        # Handle Memory interface call: add(user_id, messages)
        if user_id and messages and not session_id and not query and not response:
            session_id = user_id  # Use user_id as session_id
            return await self._add_messages_list(session_id, messages, metadata)
        
        # Handle RAG interface call with session_id
        if not session_id and user_id:
            session_id = user_id
        
        if not session_id:
            logger.error("No session_id or user_id provided")
            return False
        try:
            async with self._lock:
                success = True
                
                # Handle direct messages
                if messages:
                    for message in messages:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        msg_metadata = dict(metadata) if metadata else {}
                        msg_metadata.update(message.get("metadata", {}))
                        
                        result = await self._add_message_to_checkpoint(
                            session_id, role, content, msg_metadata
                        )
                        if not result:
                            success = False
                
                # Handle query/response pairs
                else:
                    if query:
                        result = await self._add_message_to_checkpoint(
                            session_id, "user", query, metadata
                        )
                        if not result:
                            success = False
                    
                    if response:
                        result = await self._add_message_to_checkpoint(
                            session_id, "assistant", response, metadata
                        )
                        if not result:
                            success = False
                
                # Update session tracking
                if session_id not in self._sessions:
                    self._sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat(),
                        "message_count": 0
                    }
                
                self._sessions[session_id]["last_active"] = datetime.now().isoformat()
                self._sessions[session_id]["message_count"] += len(messages) if messages else (1 if query else 0) + (1 if response else 0)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to add messages to memory: {str(e)}", exc_info=True)
            return False
    
    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        try:
            messages = await self._get_messages_from_checkpoint(session_id, limit)
            return messages
        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {str(e)}", exc_info=True)
            return []
    
    async def get_session_history_by_soeid(self, session_id: str, soeid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session filtered by SOEID.
        
        Args:
            session_id: Session identifier
            soeid: Source of Entity ID for the user
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages for the specific user
        """
        try:
            # Get all messages for the session
            all_messages = await self._get_messages_from_checkpoint(session_id, limit)
            
            # Filter messages by SOEID
            filtered_messages = []
            for msg in all_messages:
                msg_soeid = msg.get("metadata", {}).get("soeid")
                if msg_soeid == soeid:
                    filtered_messages.append(msg)
                else:
                    logger.debug(f"Filtering out message with SOEID '{msg_soeid}' (looking for '{soeid}')")
            
            logger.debug(f"Filtered session {session_id} history: {len(all_messages)} total -> {len(filtered_messages)} for SOEID {soeid}")
            return filtered_messages
            
        except Exception as e:
            logger.error(f"Failed to get session history for session {session_id} and SOEID {soeid}: {str(e)}", exc_info=True)
            return []
    
    async def get_relevant_history(self, 
                                 session_id: str, 
                                 query: str, 
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get query-relevant conversation history.
        
        Args:
            session_id: Session identifier
            query: Current query to find relevant history for
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of relevant conversation messages
        """
        # For now, return recent history
        # In a more sophisticated implementation, you could use semantic search
        return await self.get_history(session_id, limit)
    
    async def get_user_history_by_soeid(self, soeid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all sessions (with all messages) where any message in the session has the given SOEID.
        
        Args:
            soeid: Source of Entity ID for the user
            limit: Maximum number of sessions to return
            
        Returns:
            List of sessions with messages for the user
        """
        try:
            # Get all thread IDs
            thread_ids = await self._list_thread_ids()
            
            user_sessions = []
            logger.debug(f"Searching for SOEID '{soeid}' across {len(thread_ids)} threads")
            
            for thread_id in thread_ids:
                messages = await self._get_messages_from_checkpoint(thread_id)
                logger.debug(f"Thread {thread_id}: Found {len(messages)} messages")
                
                # Check if any message in this session has the SOEID
                session_has_soeid = False
                for message in messages:
                    msg_soeid = message.get("soeid")
                    logger.debug(f"  Message SOEID: '{msg_soeid}' vs target: '{soeid}'")
                    if msg_soeid == soeid:
                        session_has_soeid = True
                        break
                
                logger.debug(f"Thread {thread_id}: Has SOEID {soeid}? {session_has_soeid}")
                
                if session_has_soeid:
                    # Ensure all messages have the SOEID field
                    for message in messages:
                        if "soeid" not in message:
                            message["soeid"] = soeid
                    
                    user_sessions.append({
                        "session_id": thread_id,
                        "messages": messages,
                        "message_count": len(messages),
                        "created_at": messages[0].get("timestamp") if messages else None,
                        "last_active": messages[-1].get("timestamp") if messages else None
                    })
            
            # Sort by last active (most recent first)
            user_sessions.sort(key=lambda x: x.get("last_active", ""), reverse=True)
            
            # Apply limit if specified
            if limit and len(user_sessions) > limit:
                user_sessions = user_sessions[:limit]
            
            return user_sessions
            
        except Exception as e:
            logger.error(f"Failed to get user history by SOEID {soeid}: {str(e)}", exc_info=True)
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                # Delete thread from checkpointer
                config = await self._get_thread_config(session_id)
                if self._store_type == "postgres":
                    with PostgresSaver.from_conn_string(self._postgres_connection_string) as checkpointer:
                        checkpointer.delete_thread(config["configurable"]["thread_id"])
                else:
                    self._checkpointer.delete_thread(config["configurable"]["thread_id"])
                
                # Remove from session tracking
                if session_id in self._sessions:
                    del self._sessions[session_id]
                
                # Clear from internal caches
                if hasattr(self, '_simple_storage') and session_id in self._simple_storage:
                    del self._simple_storage[session_id]
                
                if hasattr(self, '_session_messages'):
                    session_key = f"session_{session_id}"
                    if session_key in self._session_messages:
                        del self._session_messages[session_key]
                
                logger.info(f"Cleared session {session_id} from checkpointer and caches")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {str(e)}", exc_info=True)
            return False
    
    async def clear_user_history(self, soeid: str) -> bool:
        """Clear all conversation history for a specific user by SOEID.
        
        Args:
            soeid: Source of Entity ID for the user
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._lock:
                # Get all sessions for the user
                user_sessions = await self.get_user_history_by_soeid(soeid)
                
                # Clear each session
                success = True
                for session in user_sessions:
                    session_id = session["session_id"]
                    result = await self.clear_session(session_id)
                    if not result:
                        success = False
                
                logger.info(f"Cleared user history for SOEID {soeid}")
                return success
                
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
        if not self._store:
            logger.warning("Long-term memory store is not available")
            return False
        
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
        if not self._store:
            logger.warning("Long-term memory store is not available")
            return None
        
        try:
            items = self._store.get(namespace, key)
            if items:
                if isinstance(items, list) and len(items) > 0:
                    item = items[0]
                else:
                    item = items
                
                if hasattr(item, 'value'):
                    return item.value
                elif hasattr(item, 'data'):
                    return item.data
                elif isinstance(item, dict):
                    return item
                else:
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
        if not self._store:
            logger.warning("Long-term memory store is not available")
            return []
        
        try:
            items = self._store.search(namespace, query=query, filter=filter_dict, limit=limit)
            return [item.value for item in items]
        except Exception as e:
            logger.error(f"Failed to search long-term memory: {str(e)}", exc_info=True)
            return []
    
    async def get_chat_history_by_soeid_and_date(self, 
                                               soeid: str, 
                                               days: int = 7, 
                                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat history for a SOEID within the specified number of days.
        
        Args:
            soeid: Source of Entity ID for the user
            days: Number of days to look back (default: 7)
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages from the specified date range
        """
        try:
            # Calculate the cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get all history for the SOEID
            all_history = await self.get_user_history_by_soeid(soeid)
            
            # Filter messages by date
            filtered_messages = []
            for message in all_history:
                timestamp_str = message.get("timestamp")
                if timestamp_str:
                    try:
                        # Parse timestamp (handle both ISO format and other formats)
                        if timestamp_str.endswith('Z'):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        
                        # Only include messages within the date range
                        if timestamp >= cutoff_date:
                            filtered_messages.append(message)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse timestamp {timestamp_str}: {str(e)}")
                        # Include message if we can't parse timestamp (better to include than exclude)
                        filtered_messages.append(message)
                else:
                    # Include messages without timestamps (better to include than exclude)
                    filtered_messages.append(message)
            
            # Sort by timestamp (newest first)
            filtered_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Apply limit if specified
            if limit and len(filtered_messages) > limit:
                filtered_messages = filtered_messages[:limit]
            
            logger.debug(f"Retrieved {len(filtered_messages)} messages for SOEID {soeid} within {days} days")
            return filtered_messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history by SOEID and date: {str(e)}", exc_info=True)
            return []
    
    def cleanup(self):
        """Clean up resources."""
        # No persistent connections to clean up with the new approach
        if hasattr(self, '_postgres_connection_string'):
            logger.info("PostgreSQL memory cleanup - no persistent connections to close")
        
        # Clear any cached data (but keep persistent data in PostgreSQL)
        if hasattr(self, '_session_messages'):
            self._session_messages.clear()
            logger.debug("Cleared session message cache")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
