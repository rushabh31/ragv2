"""Production-grade LangGraph memory implementation using AsyncPostgresSaver for chat history."""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# PostgreSQL imports
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

# LangGraph components not needed for direct PostgreSQL implementation
LANGGRAPH_AVAILABLE = False

from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)


class LangGraphCheckpointMemory(BaseMemory):
    """
    PostgreSQL-based conversation memory implementation.
    
    This implementation provides:
    - Direct PostgreSQL storage for conversation history
    - Session-based conversation management
    - SOEID-based user namespacing
    - Cross-session user history tracking
    - Proper database schema with indexes for performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL checkpoint memory.
        
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
        
        if not ASYNCPG_AVAILABLE:
            raise MemoryError(
                "asyncpg is not available. Install with: pip install asyncpg"
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
        self._pool_min_size = self._postgres_config.get("pool_min_size", 1)
        self._pool_max_size = self._postgres_config.get("pool_max_size", 10)
        
        # Build connection string using environment variables (production-grade)
        try:
            from src.utils.env_manager import env
            
            # Get database name from config or environment (prefer environment)
            database = self._postgres_config.get("database")  # Config can override if needed
            
            # Build connection string using env manager with PGVECTOR_URL and secrets
            self._connection_string = env.build_postgresql_connection_string(
                database=database,
                schema=self._schema_name if self._schema_name != "public" else None,
                ssl_mode=self._postgres_config.get("ssl_mode", "require")  # Production default: require SSL
            )
            
            logger.info("Successfully built PostgreSQL connection string from environment variables and YAML secrets")
            
        except Exception as e:
            # In production, we should not have hardcoded connection strings
            logger.error(f"Failed to build PostgreSQL connection string from environment: {e}")
            raise MemoryError(
                f"Failed to build PostgreSQL connection string from environment variables: {e}. "
                "Ensure PGVECTOR_URL is set and PostgreSQL credentials are available in YAML secrets file."
            )
        
        # Database components
        self._pool: Optional[Any] = None  # asyncpg.Pool type hint causes issues
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Table names with schema prefix
        schema_prefix = f'"{self._schema_name}".' if self._schema_name != "public" else ""
        self._conversations_table = f"{schema_prefix}chat_conversations"
        self._messages_table = f"{schema_prefix}chat_messages"
        self._user_sessions_table = f"{schema_prefix}chat_user_sessions"
        
        logger.info(f"LangGraph checkpoint memory initialized with schema: {self._schema_name}")
    
    async def _ensure_initialized(self):
        """Ensure components are initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._initialize_components()
                    self._initialized = True
    
    async def _initialize_components(self):
        """Initialize PostgreSQL components and create tables."""
        try:
            if not self._enabled:
                logger.info("Memory disabled - skipping component initialization")
                return
            
            logger.info("Initializing PostgreSQL memory backend...")
            
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                command_timeout=60
            )
            logger.info(f"PostgreSQL connection pool created (min: {self._pool_min_size}, max: {self._pool_max_size})")
            
            # Create schema if needed
            await self._create_schema()
            
            # Create tables and indexes
            await self._create_tables()
            
            logger.info("PostgreSQL memory backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL components: {e}")
            raise MemoryError(f"PostgreSQL initialization failed: {e}") from e
    
    async def _create_schema(self):
        """Create the database schema if it doesn't exist."""
        if self._schema_name == "public":
            return  # No need to create public schema
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema_name}"')
                logger.info(f"Schema '{self._schema_name}' created or verified")
        except Exception as e:
            logger.error(f"Failed to create schema '{self._schema_name}': {e}")
            raise
    
    async def _create_tables(self):
        """Create all required tables and indexes."""
        try:
            async with self._pool.acquire() as conn:
                # Create conversations table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._conversations_table} (
                        session_id VARCHAR(255) PRIMARY KEY,
                        soeid VARCHAR(100),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)
                
                # Create messages table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._messages_table} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id VARCHAR(255) NOT NULL,
                        soeid VARCHAR(100),
                        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        FOREIGN KEY (session_id) REFERENCES {self._conversations_table}(session_id) ON DELETE CASCADE
                    )
                """)
                
                # Create user sessions mapping table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._user_sessions_table} (
                        soeid VARCHAR(100) NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        PRIMARY KEY (soeid, session_id),
                        FOREIGN KEY (session_id) REFERENCES {self._conversations_table}(session_id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                await self._create_indexes(conn)
                
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def _create_indexes(self, conn):
        """Create database indexes for performance."""
        try:
            # Indexes for messages table
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_messages_session_id ON {self._messages_table}(session_id)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_messages_soeid ON {self._messages_table}(soeid)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_messages_timestamp ON {self._messages_table}(timestamp DESC)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_messages_soeid_timestamp ON {self._messages_table}(soeid, timestamp DESC)")
            
            # Indexes for conversations table
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_conversations_soeid ON {self._conversations_table}(soeid)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_conversations_updated_at ON {self._conversations_table}(updated_at DESC)")
            
            # Indexes for user sessions table
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_user_sessions_soeid ON {self._user_sessions_table}(soeid)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema_name}_user_sessions_last_activity ON {self._user_sessions_table}(last_activity DESC)")
            
            logger.debug("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some database indexes: {e}")
            # Don't raise - indexes are for performance, not critical
    
    
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
        """Add a single interaction to PostgreSQL database."""
        try:
            # Extract SOEID from metadata
            soeid = metadata.get("soeid") if metadata else None
            
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Ensure conversation record exists
                    await conn.execute(f"""
                        INSERT INTO {self._conversations_table} (session_id, soeid, metadata)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (session_id) DO UPDATE SET 
                            updated_at = NOW(),
                            soeid = COALESCE(EXCLUDED.soeid, {self._conversations_table}.soeid),
                            metadata = {self._conversations_table}.metadata || EXCLUDED.metadata
                    """, session_id, soeid, json.dumps(metadata or {}))
                    
                    # Insert user message
                    await conn.execute(f"""
                        INSERT INTO {self._messages_table} (session_id, soeid, role, content, metadata)
                        VALUES ($1, $2, 'user', $3, $4)
                    """, session_id, soeid, query, json.dumps(metadata or {}))
                    
                    # Insert assistant message
                    await conn.execute(f"""
                        INSERT INTO {self._messages_table} (session_id, soeid, role, content, metadata)
                        VALUES ($1, $2, 'assistant', $3, $4)
                    """, session_id, soeid, response, json.dumps(metadata or {}))
                    
                    # Update user sessions mapping if SOEID provided
                    if soeid:
                        await conn.execute(f"""
                            INSERT INTO {self._user_sessions_table} (soeid, session_id)
                            VALUES ($1, $2)
                            ON CONFLICT (soeid, session_id) DO UPDATE SET
                                last_activity = NOW()
                        """, soeid, session_id)
                    
                    logger.debug(f"Stored conversation in PostgreSQL for session {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add session interaction to PostgreSQL: {e}")
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
        
        if not self._enabled or not self._pool:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                # Build query with optional date filtering
                conditions = ["session_id = $1"]
                params = [session_id]
                
                if self._max_history_days > 0:
                    cutoff_date = datetime.now() - timedelta(days=self._max_history_days)
                    conditions.append("timestamp >= $2")
                    params.append(cutoff_date)
                
                where_clause = " AND ".join(conditions)
                limit_clause = f"LIMIT {limit}" if limit else (f"LIMIT {self._max_history}" if self._max_history else "")
                
                query = f"""
                    SELECT role, content, timestamp, soeid, metadata 
                    FROM {self._messages_table}
                    WHERE {where_clause}
                    ORDER BY timestamp ASC
                    {limit_clause}
                """
                
                rows = await conn.fetch(query, *params)
                
                history = []
                for row in rows:
                    history.append({
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"].isoformat(),
                        "soeid": row["soeid"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })
                
                logger.debug(f"Retrieved {len(history)} messages for session {session_id} from PostgreSQL")
                return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history from PostgreSQL: {e}")
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
        
        if not self._enabled or not self._pool:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with self._pool.acquire() as conn:
                query = f"""
                    SELECT session_id, role, content, timestamp, metadata 
                    FROM {self._messages_table}
                    WHERE soeid = $1 AND timestamp >= $2
                    ORDER BY timestamp DESC
                    {f"LIMIT {limit}" if limit else f"LIMIT {self._max_history}" if self._max_history else ""}
                """
                
                rows = await conn.fetch(query, soeid, cutoff_date)
                
                history = []
                for row in rows:
                    history.append({
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"].isoformat(),
                        "session_id": row["session_id"],
                        "soeid": soeid,
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })
                
                logger.debug(f"Retrieved {len(history)} cross-session messages for SOEID {soeid} from PostgreSQL")
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
            await self._ensure_initialized()
            
            if not self._enabled or not self._pool:
                return []
            
            async with self._pool.acquire() as conn:
                query = f"SELECT DISTINCT session_id FROM {self._conversations_table} ORDER BY updated_at DESC"
                rows = await conn.fetch(query)
                thread_ids = [row["session_id"] for row in rows]
                
                logger.debug(f"Found {len(thread_ids)} thread IDs in PostgreSQL")
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
        
        if not self._enabled or not self._pool:
            return True
        
        try:
            async with self._pool.acquire() as conn:
                # Delete the conversation (messages will be deleted by CASCADE)
                result = await conn.execute(f"DELETE FROM {self._conversations_table} WHERE session_id = $1", session_id)
                
                if result == "DELETE 1":
                    logger.info(f"Cleared session {session_id} from PostgreSQL")
                else:
                    logger.debug(f"Session {session_id} not found in PostgreSQL")
                
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
            async with self._pool.acquire() as conn:
                # Delete all conversations for this user (messages will be deleted by CASCADE)
                result = await conn.execute(f"DELETE FROM {self._conversations_table} WHERE soeid = $1", soeid)
                
                # Extract the number of deleted rows
                deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
                
                logger.info(f"Cleared {deleted_count} sessions for SOEID {soeid} from PostgreSQL")
            
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
            # Close the PostgreSQL connection pool
            if self._pool:
                try:
                    await self._pool.close()
                    self._pool = None
                    logger.debug("PostgreSQL connection pool closed")
                except Exception as e:
                    logger.warning(f"Error closing PostgreSQL pool: {e}")
            
            self._initialized = False
            logger.info("PostgreSQL memory cleanup completed")
            
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