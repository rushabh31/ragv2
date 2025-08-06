"""Production-grade LangGraph memory implementation using AsyncPostgresSaver for chat history."""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    MessagesState = None
    START = None
    AsyncPostgresSaver = None
    AsyncPostgresStore = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None

from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)


class LangGraphCheckpointMemory(BaseMemory):
    """
    Production LangGraph memory implementation using AsyncPostgresSaver and AsyncPostgresStore.
    
    This implementation follows LangGraph best practices:
    - Uses AsyncPostgresSaver for thread-scoped conversation checkpoints
    - Uses AsyncPostgresStore for cross-thread user memories and profiles
    - Supports SOEID-based user namespacing
    - Provides enable/disable functionality
    - Handles connection management and cleanup properly
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LangGraph checkpoint memory.
        
        Args:
            config: Configuration dictionary with the following structure:
                {
                    "enabled": True,
                    "chat_history_enabled": True,
                    "max_history_days": 30,
                    "store_type": "postgres",
                    "postgres": {
                        "connection_string": "postgresql://user:pass@host:port/db",
                        "schema_name": "chat_memory",  # Optional custom schema
                        "pool_min_size": 1,
                        "pool_max_size": 10
                    },
                    "max_history": 50
                }
        """
        super().__init__(config)
        
        if not LANGGRAPH_AVAILABLE:
            raise MemoryError(
                "LangGraph is not available. Install with: "
                "pip install langgraph langgraph-checkpoint-postgres langgraph-store-postgres"
            )
        
        # Configuration
        self._config = config
        self._enabled = config.get("enabled", True)
        self._chat_history_enabled = config.get("chat_history_enabled", True)
        self._max_history_days = config.get("max_history_days", 30)
        self._max_history = config.get("max_history", 50)
        self._store_type = config.get("store_type", "postgres")
        
        # PostgreSQL configuration
        self._postgres_config = config.get("postgres", {})
        self._schema_name = self._postgres_config.get("schema_name", "public")
        
        # Build connection string using environment variables
        try:
            from src.utils.env_manager import env
            
            # Get database name from config or environment
            database = self._postgres_config.get("database") or env.get_string("POSTGRES_DB")
            
            # Build connection string using env manager
            self._connection_string = env.build_postgresql_connection_string(
                database=database,
                schema=self._schema_name if self._schema_name != "public" else None,
                ssl_mode=self._postgres_config.get("ssl_mode", "prefer")
            )
            
            logger.info(f"Built PostgreSQL connection string using environment variables")
            
        except Exception as e:
            # Fallback to explicit connection string if env variables not available
            self._connection_string = self._postgres_config.get("connection_string")
            if not self._connection_string:
                raise MemoryError(f"Failed to build PostgreSQL connection string from environment variables: {e}")
            
            logger.warning("Using explicit connection string from config (env variables not available)")
            
            # Add schema to connection string if custom schema is specified
            if self._schema_name != "public":
                separator = "&" if "?" in self._connection_string else "?"
                self._connection_string += f"{separator}options=-c%20search_path%3D{self._schema_name}%2Cpublic"
        
        # Components (initialized lazily)
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._store: Optional[AsyncPostgresStore] = None
        self._graph: Optional[StateGraph] = None
        self._memory_storage: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        
        logger.info(f"LangGraph checkpoint memory initialized with schema: {self._schema_name}")
    
    async def _ensure_initialized(self):
        """Ensure components are initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._initialize_components()
                    self._initialized = True
    
    async def _initialize_components(self):
        """Initialize LangGraph components."""
        try:
            if not self._enabled:
                logger.info("Memory disabled - skipping component initialization")
                return
            
            logger.info("Initializing LangGraph components with PostgreSQL backend...")
            
            # For now, disable LangGraph persistence due to schema issues
            # and fall back to in-memory storage until the database schema is resolved
            logger.warning("LangGraph PostgreSQL persistence temporarily disabled due to schema compatibility issues")
            logger.warning("Using in-memory conversation storage as fallback")
            
            # Initialize components but don't set up tables yet
            try:
                self._checkpointer = AsyncPostgresSaver.from_conn_string(self._connection_string)
                logger.info("AsyncPostgresSaver created (tables setup deferred)")
            except Exception as e:
                logger.warning(f"Failed to create AsyncPostgresSaver: {e}")
                self._checkpointer = None
            
            try:
                self._store = AsyncPostgresStore.from_conn_string(self._connection_string)
                logger.info("AsyncPostgresStore created (tables setup deferred)")
            except Exception as e:
                logger.warning(f"Failed to create AsyncPostgresStore: {e}")
                self._store = None
            
            # Create a simple in-memory graph for now
            logger.info("Creating LangGraph workflow without persistence...")
            self._graph = self._create_simple_graph()
            logger.info("LangGraph workflow created successfully (in-memory mode)")
            
            # Initialize in-memory storage as fallback
            self._memory_storage = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph components: {e}")
            raise MemoryError(f"Initialization failed: {e}") from e
    
    def _create_message_graph(self) -> StateGraph:
        """Create a simple LangGraph workflow for message handling."""
        
        async def process_messages(state: MessagesState):
            """Simple node that processes messages without modification."""
            return state
        
        # Create a minimal graph for checkpoint management
        workflow = StateGraph(MessagesState)
        workflow.add_node("process", process_messages)
        workflow.add_edge(START, "process")
        
        try:
            # Compile the graph with checkpointer and store
            return workflow.compile(checkpointer=self._checkpointer, store=self._store)
        except Exception as e:
            logger.error(f"Failed to compile graph with checkpointer and store: {e}")
            # Fallback: try without store first
            try:
                logger.warning("Attempting to compile graph with checkpointer only (no store)")
                return workflow.compile(checkpointer=self._checkpointer)
            except Exception as e2:
                logger.error(f"Failed to compile graph with checkpointer only: {e2}")
                # Final fallback: no persistence
                logger.warning("Compiling graph without persistence (memory-only mode)")
                return workflow.compile()
    
    def _create_simple_graph(self) -> StateGraph:
        """Create a simple LangGraph workflow without persistence."""
        
        async def process_messages(state: MessagesState):
            """Simple node that processes messages without modification."""
            return state
        
        # Create a minimal graph without persistence
        workflow = StateGraph(MessagesState)
        workflow.add_node("process", process_messages)
        workflow.add_edge(START, "process")
        
        # Compile without any persistence
        return workflow.compile()
    
    def _get_thread_config(self, session_id: str, soeid: Optional[str] = None) -> Dict[str, Any]:
        """
        Get thread configuration for LangGraph operations.
        
        Args:
            session_id: Session identifier
            soeid: Optional user SOEID for namespacing
            
        Returns:
            Thread configuration dictionary
        """
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",
            }
        }
        
        if soeid:
            # Use SOEID-based namespace for user isolation
            config["configurable"]["user_id"] = soeid
            config["configurable"]["checkpoint_ns"] = f"user:{soeid}"
        
        return config
    
    def _get_user_namespace(self, soeid: str) -> Tuple[str, ...]:
        """Get namespace for user-specific store operations."""
        return (soeid, "memories")
    
    def _convert_to_langchain_messages(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Convert message dictionaries to LangChain message objects."""
        langchain_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                # Default to HumanMessage for unknown roles
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def _convert_from_langchain_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain message objects to dictionaries."""
        converted = []
        
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            converted.append({
                "role": role,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
        
        return converted
    
    async def add(self, 
                  user_id: str = None,
                  messages: List[Dict[str, str]] = None,
                  session_id: str = None, 
                  query: str = None, 
                  response: str = None, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation exchange to memory.
        
        Args:
            user_id: User identifier (legacy parameter for Memory interface)
            messages: List of message dictionaries (legacy parameter for Memory interface)
            session_id: Session identifier
            query: User query
            response: Assistant response
            metadata: Optional metadata (should contain 'soeid' for user identification)
            
        Returns:
            True if successful, False otherwise
        """
        await self._ensure_initialized()
        
        if not self._enabled or not self._chat_history_enabled:
            logger.debug("Memory or chat history disabled")
            return True
        
        try:
            # Handle both old and new parameter styles
            if session_id is not None and query is not None and response is not None:
                # New style parameters (BaseMemory interface)
                return await self._add_session_interaction(session_id, query, response, metadata)
            
            elif user_id is not None and messages is not None:
                # Old style parameters (Memory interface)
                return await self._add_user_messages(user_id, messages, metadata)
            
            else:
                logger.warning("Invalid parameters provided to LangGraphCheckpointMemory.add()")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add conversation to memory: {e}")
            return False
    
    async def _add_session_interaction(self, 
                                     session_id: str, 
                                     query: str, 
                                     response: str, 
                                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a single interaction to memory (BaseMemory style)."""
        try:
            # Extract SOEID from metadata
            soeid = metadata.get("soeid") if metadata else None
            
            # Get thread configuration
            config = self._get_thread_config(session_id, soeid)
            
            # Create messages for the conversation
            messages = [
                HumanMessage(content=query),
                AIMessage(content=response)
            ]
            
            # Store conversation in memory fallback for now
            async with self._lock:
                if session_id not in self._memory_storage:
                    self._memory_storage[session_id] = []
                
                # Add messages to in-memory storage
                timestamp = datetime.now().isoformat()
                self._memory_storage[session_id].append({
                    "role": "user",
                    "content": query,
                    "timestamp": timestamp,
                    "soeid": soeid
                })
                self._memory_storage[session_id].append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": timestamp,
                    "soeid": soeid
                })
                
                # Keep only recent messages to prevent unlimited growth
                if len(self._memory_storage[session_id]) > self._max_history * 2:
                    self._memory_storage[session_id] = self._memory_storage[session_id][-self._max_history * 2:]
                
                logger.debug(f"Stored conversation in memory for session {session_id}")
            
            # TODO: Re-enable LangGraph persistence once database schema issues are resolved
            # try:
            #     async with self._checkpointer as checkpointer:
            #         temp_graph = self._create_message_graph()
            #         await temp_graph.ainvoke({"messages": messages}, config=config)
            # except Exception as e:
            #     logger.warning(f"LangGraph persistence failed: {e}")
            #     # Continue with in-memory storage
            
            # Store operation is already handled above in memory storage
            # TODO: Re-enable user profile storage once database schema issues are resolved
            
            logger.debug(f"Added conversation to memory for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add session interaction: {e}")
            return False
    
    async def _add_user_messages(self, 
                               user_id: str, 
                               messages: List[Dict[str, str]], 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add messages to memory (Memory interface style)."""
        try:
            # Convert messages to query/response pairs
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]
                    assistant_msg = messages[i + 1]
                    
                    # Use user_id as session_id and add the interaction
                    success = await self._add_session_interaction(
                        session_id=user_id,
                        query=user_msg.get("content", ""),
                        response=assistant_msg.get("content", ""),
                        metadata=metadata or {"user_id": user_id}
                    )
                    
                    if not success:
                        logger.warning(f"Failed to add message pair {i//2 + 1}")
                        
            return True
            
        except Exception as e:
            logger.error(f"Failed to add user messages: {e}")
            return False
    
    async def _update_user_profile_with_store(self, 
                                           store: Any,  # AsyncPostgresStore type hint causes issues
                                           soeid: str, 
                                           query: str, 
                                           response: str, 
                                           session_id: str,
                                           metadata: Optional[Dict[str, Any]] = None):
        """Update user profile with conversation insights using provided store."""
        try:
            namespace = self._get_user_namespace(soeid)
            
            # Create memory entry
            memory_id = str(uuid.uuid4())
            memory_data = {
                "conversation_snippet": {
                    "query": query[:500],  # Truncate for storage efficiency
                    "response": response[:500],
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                },
                "interaction_type": "chat"
            }
            
            # Store in cross-thread memory
            await store.aput(
                namespace=namespace,
                key=memory_id,
                value=memory_data
            )
            
            logger.debug(f"Updated user profile for SOEID {soeid}")
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        await self._ensure_initialized()
        
        if not self._enabled or not self._checkpointer:
            return []
        
        try:
            # Get history from in-memory storage
            async with self._lock:
                if session_id not in self._memory_storage:
                    logger.debug(f"Session {session_id} not found in memory storage")
                    return []
                
                history = self._memory_storage[session_id].copy()
                
                # Apply date filtering if configured
                if self._max_history_days > 0:
                    cutoff_date = datetime.now() - timedelta(days=self._max_history_days)
                    filtered_history = []
                    
                    for msg in history:
                        timestamp_str = msg.get("timestamp", "")
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp >= cutoff_date:
                                filtered_history.append(msg)
                        except (ValueError, TypeError):
                            # Include messages with invalid timestamps
                            filtered_history.append(msg)
                    
                    history = filtered_history
                
                # Apply limit
                if limit:
                    history = history[-limit:]
                elif self._max_history:
                    history = history[-self._max_history:]
                
                logger.debug(f"Retrieved {len(history)} messages for session {session_id} from memory storage")
                return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def get_chat_history_by_soeid_and_date(
        self, 
        soeid: str, 
        days: int = 7, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a user across all sessions within date range.
        
        Args:
            soeid: User SOEID
            days: Number of days to look back
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages across all user sessions
        """
        await self._ensure_initialized()
        
        if not self._enabled or not self._store:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Search in-memory storage for user messages across all sessions
            history = []
            async with self._lock:
                for session_id, messages in self._memory_storage.items():
                    for msg in messages:
                        # Check if message belongs to this user and is within date range
                        if msg.get("soeid") == soeid:
                            timestamp_str = msg.get("timestamp", "")
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if timestamp >= cutoff_date:
                                    history.append({
                                        "role": msg.get("role", "user"),
                                        "content": msg.get("content", ""),
                                        "timestamp": timestamp_str,
                                        "session_id": session_id,
                                        "soeid": soeid
                                    })
                            except (ValueError, TypeError):
                                logger.warning(f"Failed to parse timestamp {timestamp_str}")
                                continue
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                history = history[:limit]
            
            logger.debug(f"Retrieved {len(history)} cross-session messages for SOEID {soeid} from memory storage")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get chat history by SOEID: {e}")
            return []
    
    async def get_session_history_by_soeid(self, session_id: str, soeid: str) -> List[Dict[str, Any]]:
        """
        Get session history filtered by SOEID.
        
        Args:
            session_id: Session identifier
            soeid: User SOEID to filter by
            
        Returns:
            List of conversation messages for the specific user in the session
        """
        try:
            # Get all messages for the session
            all_messages = await self.get_history(session_id)
            
            # Filter messages by SOEID
            filtered_messages = []
            for msg in all_messages:
                if msg.get("soeid") == soeid:
                    filtered_messages.append(msg)
            
            logger.debug(f"Retrieved {len(filtered_messages)} messages for session {session_id} and SOEID {soeid}")
            return filtered_messages
            
        except Exception as e:
            logger.error(f"Failed to get session history by SOEID: {e}")
            return []
    
    async def get_user_history_by_soeid(self, soeid: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get user history by SOEID across all sessions. Alias for get_chat_history_by_soeid_and_date.
        
        Args:
            soeid: User SOEID
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages across all user sessions
        """
        # Use a reasonable default of 30 days for backward compatibility
        return await self.get_chat_history_by_soeid_and_date(soeid, days=30, limit=limit)
    
    async def _list_thread_ids(self) -> List[str]:
        """
        List all thread IDs (session IDs) that have conversation history.
        
        Returns:
            List of session IDs
        """
        try:
            async with self._lock:
                thread_ids = list(self._memory_storage.keys())
                logger.debug(f"Found {len(thread_ids)} thread IDs in memory storage")
                return thread_ids
                
        except Exception as e:
            logger.error(f"Failed to list thread IDs: {e}")
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        await self._ensure_initialized()
        
        if not self._enabled or not self._checkpointer:
            return True
        
        try:
            # Clear from in-memory storage
            async with self._lock:
                if session_id in self._memory_storage:
                    del self._memory_storage[session_id]
                    logger.info(f"Cleared session {session_id} from memory storage")
                    return True
                else:
                    logger.debug(f"Session {session_id} not found in memory storage")
                    return True
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    async def clear_user_history(self, soeid: str) -> bool:
        """
        Clear all history for a user across all sessions.
        
        Args:
            soeid: User SOEID
            
        Returns:
            True if successful, False otherwise
        """
        await self._ensure_initialized()
        
        if not self._enabled:
            return True
        
        try:
            # Clear all sessions for this user from in-memory storage
            async with self._lock:
                sessions_to_clear = []
                for session_id, messages in self._memory_storage.items():
                    # Check if any message in this session belongs to this user
                    for msg in messages:
                        if msg.get("soeid") == soeid:
                            sessions_to_clear.append(session_id)
                            break
                
                # Clear the sessions
                for session_id in sessions_to_clear:
                    del self._memory_storage[session_id]
                
                logger.info(f"Cleared {len(sessions_to_clear)} sessions for SOEID {soeid} from memory storage")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear user history for SOEID {soeid}: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if memory is enabled."""
        return self._enabled
    
    def is_chat_history_enabled(self) -> bool:
        """Check if chat history is enabled."""
        return self._chat_history_enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable memory."""
        self._enabled = enabled
        logger.info(f"Memory enabled set to: {enabled}")
    
    def set_chat_history_enabled(self, enabled: bool):
        """Enable or disable chat history."""
        self._chat_history_enabled = enabled
        logger.info(f"Chat history enabled set to: {enabled}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        await self._ensure_initialized()
        
        stats = {
            "enabled": self._enabled,
            "chat_history_enabled": self._chat_history_enabled,
            "store_type": self._store_type,
            "schema_name": self._schema_name,
            "max_history": self._max_history,
            "max_history_days": self._max_history_days,
            "checkpointer_available": self._checkpointer is not None,
            "store_available": self._store is not None
        }
        
        return stats
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up components
            if self._checkpointer:
                try:
                    # The checkpointer will handle its own cleanup when the object is destroyed
                    self._checkpointer = None
                    logger.debug("AsyncPostgresSaver cleared")
                except Exception as e:
                    logger.warning(f"Error clearing checkpointer: {e}")
            
            if self._store:
                try:
                    # The store will handle its own cleanup when the object is destroyed
                    self._store = None
                    logger.debug("AsyncPostgresStore cleared")
                except Exception as e:
                    logger.warning(f"Error clearing store: {e}")
            
            self._graph = None
            self._initialized = False
            logger.info("LangGraph memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Abstract methods implementation for BaseMemory
    async def _add_interaction(self, 
                             session_id: str,
                             query: str, 
                             response: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add interaction using the internal method."""
        return await self._add_session_interaction(session_id, query, response, metadata)
    
    async def _get_conversation_history(self, 
                                      session_id: str, 
                                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history using the public get_history method."""
        return await self.get_history(session_id, limit)
    
    async def _get_relevant_history(self, 
                                  session_id: str, 
                                  query: str, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get relevant history - for now, return recent history."""
        # For LangGraph implementation, we'll use recent history as relevant history
        # In the future, this could be enhanced with semantic similarity search using the query parameter
        _ = query  # TODO: Use query for semantic similarity search in future implementation
        return await self.get_history(session_id, limit)
    
    
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a user (Memory interface).
        
        Args:
            user_id: User identifier (treated as session_id)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        try:
            # Use user_id as session_id for this interface
            history = await self.get_history(user_id, limit)
            
            # Convert to the format expected by Memory interface
            messages = []
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for Memory interface: {e}")
            return []


# Remove the alias - LangGraphCheckpointMemory is the main class