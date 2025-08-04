"""Production-grade LangGraph memory implementation using AsyncPostgresSaver with SOEID profiles."""

import logging
import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncpg

# Import our custom embedding models and factories
try:
    from ...models.embedding.vertex_embedding import VertexEmbeddingAI
    from ...models.embedding.openai_embedding import OpenAIEmbeddingAI
    from ...models.embedding.azure_openai_embedding import AzureOpenAIEmbeddingAI
    from ...models.embedding.embedding_factory import EmbeddingModelFactory
    CUSTOM_EMBEDDINGS_AVAILABLE = True
except ImportError:
    CUSTOM_EMBEDDINGS_AVAILABLE = False
    VertexEmbeddingAI = None
    OpenAIEmbeddingAI = None
    AzureOpenAIEmbeddingAI = None
    EmbeddingModelFactory = None

# Import LangGraph checkpoint components
try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    LANGGRAPH_CHECKPOINT_AVAILABLE = True
except ImportError:
    LANGGRAPH_CHECKPOINT_AVAILABLE = False
    InMemorySaver = None
    BaseCheckpointSaver = None
    Checkpoint = None
    JsonPlusSerializer = None

# Import async PostgreSQL checkpoint saver
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    ASYNC_POSTGRES_CHECKPOINT_AVAILABLE = True
except ImportError:
    AsyncPostgresSaver = None
    ASYNC_POSTGRES_CHECKPOINT_AVAILABLE = False

# Import LangGraph store components for cross-thread memory
try:
    from langgraph.store.memory import InMemoryStore
    from langgraph.store.base import BaseStore, Item
    LANGGRAPH_STORE_AVAILABLE = True
except ImportError:
    LANGGRAPH_STORE_AVAILABLE = False
    InMemoryStore = None
    BaseStore = None
    Item = None

try:
    from langgraph.store.postgres.aio import AsyncPostgresStore
    POSTGRES_STORE_AVAILABLE = True
except ImportError:
    AsyncPostgresStore = None
    POSTGRES_STORE_AVAILABLE = False
    POSTGRES_STORE_AVAILABLE = False

from ..base_memory import BaseMemory

logger = logging.getLogger(__name__)


class ProductionLangGraphMemory(BaseMemory):
    """
    Production-grade LangGraph memory implementation with AsyncPostgresSaver and AsyncPostgresStore.
    
    Features:
    - AsyncPostgresSaver for thread-scoped conversation history
    - AsyncPostgresStore for cross-thread user profiles and long-term memory
    - SOEID-based user namespacing
    - Proper checkpoint and state management
    - Production-ready error handling and connection management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production LangGraph memory system.
        
        Args:
            config: Configuration dictionary with the following structure:
                {
                    "store_type": "postgres",  # Required for production
                    "postgres": {
                        "connection_string": "postgresql://user:pass@host:port/db",
                        "pool_size": 10,
                        "max_overflow": 20
                    },
                    "checkpointer": {
                        "serde": {
                            "pickle_fallback": True,
                            "encryption_key": "optional-aes-key"
                        }
                    },
                    "store": {
                        "index": {
                            "embed": "openai:text-embedding-3-small",  # Optional for semantic search
                            "dims": 1536,
                            "fields": ["content", "metadata"]
                        }
                    },
                    "memory_enabled": True,
                    "max_history": 50
                }
        """
        super().__init__(config)
        
        # Configuration
        self._store_type = config.get("store_type", "postgres")
        self._postgres_config = config.get("postgres", {})
        self._checkpointer_config = config.get("checkpointer", {})
        self._store_config = config.get("store", {})
        self._memory_enabled = config.get("memory_enabled", True)
        self._max_history = config.get("max_history", 50)
        
        # Connection management
        self._connection_string = self._postgres_config.get("connection_string")
        self._pool_size = self._postgres_config.get("pool_size", 10)
        self._max_overflow = self._postgres_config.get("max_overflow", 20)
        
        # Components
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._store: Optional[AsyncPostgresStore] = None
        self._serializer = None
        
        # Internal state
        self._initialized = False
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized ProductionLangGraphMemory with store_type: {self._store_type}")
    
    async def _ensure_initialized(self):
        """Ensure the memory system is properly initialized."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._initialize_components()
                    self._initialized = True
    
    async def _initialize_components(self):
        """Initialize checkpointer and store components."""
        try:
            if not self._memory_enabled:
                logger.info("Memory disabled - using no-op components")
                return
            
            if self._store_type != "postgres":
                raise ValueError("Production implementation requires postgres store_type")
            
            if not self._connection_string:
                raise ValueError("PostgreSQL connection string is required for production")
            
            # Validate dependencies
            if not ASYNC_POSTGRES_CHECKPOINT_AVAILABLE:
                raise ImportError(
                    "AsyncPostgresSaver not available. Install with: "
                    "pip install langgraph-checkpoint-postgres"
                )
            
            if not POSTGRES_STORE_AVAILABLE:
                raise ImportError(
                    "AsyncPostgresStore not available. Install with: "
                    "pip install langgraph-store-postgres"
                )
            
            # Initialize serializer with optional encryption
            serde_config = self._checkpointer_config.get("serde", {})
            if serde_config.get("encryption_key"):
                from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
                self._serializer = EncryptedSerializer.from_pycryptodome_aes(
                    key=serde_config["encryption_key"]
                )
                logger.info("Initialized encrypted serializer")
            else:
                self._serializer = JsonPlusSerializer(
                    pickle_fallback=serde_config.get("pickle_fallback", True)
                )
                logger.info("Initialized JSON+ serializer with pickle fallback")
            
            # Initialize AsyncPostgresSaver for checkpointing
            self._checkpointer = AsyncPostgresSaver.from_conn_string(
                self._connection_string,
                serde=self._serializer
            )
            
            # Setup checkpointer tables
            await self._checkpointer.setup()
            logger.info("AsyncPostgresSaver initialized and setup completed")
            
            # Initialize AsyncPostgresStore for cross-thread memory
            store_index_config = self._store_config.get("index")
            embeddings_config = self._store_config.get("embeddings")
            
            if store_index_config or embeddings_config:
                # Initialize with semantic search capabilities
                store_kwargs = {}
                
                if store_index_config:
                    store_kwargs["index"] = store_index_config
                
                if embeddings_config:
                    # Configure VertexAI embeddings for semantic search
                    embeddings_model = await self._create_embeddings_model(embeddings_config)
                    if embeddings_model:
                        store_kwargs["embeddings"] = embeddings_model
                        logger.info(f"Configured embeddings: {embeddings_config.get('provider', 'unknown')}")
                
                self._store = AsyncPostgresStore.from_conn_string(
                    self._connection_string,
                    **store_kwargs
                )
                logger.info("AsyncPostgresStore initialized with semantic search and embeddings")
            else:
                # Initialize without semantic search
                self._store = AsyncPostgresStore.from_conn_string(
                    self._connection_string
                )
                logger.info("AsyncPostgresStore initialized without semantic search")
            
            # Setup store tables
            await self._store.setup()
            logger.info("AsyncPostgresStore setup completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph components: {e}")
            raise
    
    async def _create_embeddings_model(self, embeddings_config: Dict[str, Any]):
        """
        Create embeddings model for LangGraph store semantic search using our custom models.
        
        Args:
            embeddings_config: Configuration dictionary with provider and model settings
            
        Returns:
            Embeddings model instance or None if creation fails
        """
        try:
            if not CUSTOM_EMBEDDINGS_AVAILABLE:
                logger.warning("Custom embedding models not available - check imports")
                return None
                
            provider = embeddings_config.get("provider", "vertex_ai")
            model_name = embeddings_config.get("model", "text-embedding-004")
            
            # Create embeddings model using our custom models with universal auth
            if provider == "vertex_ai" or provider == "google_vertexai":
                # Use our custom VertexEmbeddingAI with universal auth token service
                vertex_config = embeddings_config.get("config", {})
                
                # Create VertexAI embeddings model with our auth system
                embeddings_model = VertexEmbeddingAI(
                    model_name=model_name,
                    **vertex_config
                )
                
                # Test authentication and initialize model
                try:
                    token = embeddings_model.get_coin_token()
                    if token:
                        # Initialize the model with proper authentication
                        await embeddings_model._init_model()
                        logger.info(f"Created and authenticated VertexAI embeddings model: {model_name}")
                        return embeddings_model
                    else:
                        logger.warning("VertexAI embeddings authentication failed - no token")
                        return None
                except Exception as auth_error:
                    logger.error(f"VertexAI embeddings auth/init error: {auth_error}")
                    return None
                
            elif provider == "openai" or provider == "openai_universal":
                # Use our custom OpenAIEmbeddingAI with universal auth
                openai_config = embeddings_config.get("config", {})
                
                embeddings_model = OpenAIEmbeddingAI(
                    model=model_name,
                    **openai_config
                )
                
                # Test authentication and initialize model
                try:
                    token = embeddings_model.get_coin_token()
                    if token:
                        # OpenAI models don't need explicit async init like Vertex AI
                        logger.info(f"Created and authenticated OpenAI embeddings model: {model_name}")
                        return embeddings_model
                    else:
                        logger.warning("OpenAI embeddings authentication failed - no token")
                        return None
                except Exception as auth_error:
                    logger.error(f"OpenAI embeddings auth error: {auth_error}")
                    return None
                
            elif provider == "azure_openai":
                # Use our custom AzureOpenAIEmbeddingAI with universal auth
                azure_config = embeddings_config.get("config", {})
                
                embeddings_model = AzureOpenAIEmbeddingAI(
                    model=model_name,
                    **azure_config
                )
                
                # Test authentication
                try:
                    token = embeddings_model.get_coin_token()
                    if token:
                        logger.info(f"Created Azure OpenAI embeddings model with universal auth: {model_name}")
                        return embeddings_model
                    else:
                        logger.warning("Azure OpenAI embeddings authentication failed")
                        return None
                except Exception as auth_error:
                    logger.error(f"Azure OpenAI embeddings auth error: {auth_error}")
                    return None
                
            else:
                # Try using the embedding factory as fallback
                try:
                    embeddings_model = EmbeddingModelFactory.create_model(
                        provider,
                        model_name,
                        **embeddings_config.get("config", {})
                    )
                    
                    if embeddings_model and hasattr(embeddings_model, 'get_coin_token'):
                        token = embeddings_model.get_coin_token()
                        if token:
                            logger.info(f"Created {provider} embeddings model via factory with universal auth: {model_name}")
                            return embeddings_model
                    
                    logger.warning(f"Unsupported embeddings provider or auth failed: {provider}")
                    return None
                    
                except Exception as factory_error:
                    logger.error(f"Factory creation failed for {provider}: {factory_error}")
                    return None
                
        except Exception as e:
            logger.error(f"Failed to create custom embeddings model: {e}")
            return None
    
    def _get_thread_config(self, session_id: str, soeid: Optional[str] = None) -> Dict[str, Any]:
        """
        Get thread configuration with proper namespacing.
        
        Args:
            session_id: Session/thread identifier
            soeid: User SOEID for namespacing
            
        Returns:
            Configuration dictionary for LangGraph operations
        """
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",  # Default namespace
            }
        }
        
        if soeid:
            config["configurable"]["user_id"] = soeid
            # Use SOEID-based namespace for user-specific threading
            config["configurable"]["checkpoint_ns"] = f"user:{soeid}"
        
        return config
    
    def _get_user_namespace(self, soeid: str) -> Tuple[str, ...]:
        """
        Get user namespace for store operations.
        
        Args:
            soeid: User SOEID
            
        Returns:
            Namespace tuple for store operations
        """
        return (soeid, "memories")
    
    async def add(self, session_id: str, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message exchange to the conversation history.
        
        Args:
            session_id: Session identifier
            query: User query
            response: Assistant response
            metadata: Optional metadata including SOEID
        """
        await self._ensure_initialized()
        
        if not self._memory_enabled or not self._checkpointer:
            logger.debug("Memory disabled or checkpointer not available")
            return
        
        try:
            # Extract SOEID from metadata
            soeid = metadata.get("soeid") if metadata else None
            
            # Get thread configuration
            config = self._get_thread_config(session_id, soeid)
            
            # Create message objects
            messages = [
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id,
                    "soeid": soeid
                },
                {
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id,
                    "soeid": soeid
                }
            ]
            
            # Store in checkpointer (thread-scoped)
            checkpoint_data = {
                "messages": messages,
                "metadata": metadata or {}
            }
            
            # Create checkpoint
            checkpoint = Checkpoint(
                v=1,
                ts=datetime.utcnow().isoformat(),
                id=str(uuid.uuid4()),
                channel_values=checkpoint_data,
                channel_versions={},
                versions_seen={}
            )
            
            # Save checkpoint
            await self._checkpointer.aput(
                config=config,
                checkpoint=checkpoint,
                metadata={"source": "conversation", "step": 1},
                new_versions={}
            )
            
            # Also store user profile information in cross-thread store if SOEID provided
            if soeid and self._store:
                await self._update_user_profile(soeid, query, response, session_id)
            
            logger.debug(f"Added conversation to memory for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conversation to memory: {e}")
            raise
    
    async def _update_user_profile(self, soeid: str, query: str, response: str, session_id: str):
        """
        Update user profile with conversation insights.
        
        Args:
            soeid: User SOEID
            query: User query
            response: Assistant response
            session_id: Session identifier
        """
        try:
            namespace = self._get_user_namespace(soeid)
            
            # Create memory entry
            memory_id = str(uuid.uuid4())
            memory_data = {
                "conversation_snippet": {
                    "query": query[:500],  # Truncate for storage
                    "response": response[:500],
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "topics": self._extract_topics(query),
                "interaction_type": "chat"
            }
            
            # Store in cross-thread memory
            await self._store.aput(
                namespace=namespace,
                key=memory_id,
                value=memory_data
            )
            
            logger.debug(f"Updated user profile for SOEID {soeid}")
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text (simple keyword extraction).
        In production, this could use NLP models.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted topics/keywords
        """
        # Simple keyword extraction - in production use proper NLP
        import re
        words = re.findall(r'\b\w{4,}\b', text.lower())
        return list(set(words[:10]))  # Return unique words, max 10
    
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
        
        if not self._memory_enabled or not self._checkpointer:
            logger.debug("Memory disabled or checkpointer not available")
            return []
        
        try:
            # Get thread configuration
            config = self._get_thread_config(session_id)
            
            # Get state history
            history = []
            async for state_snapshot in self._checkpointer.alist(config):
                if state_snapshot.values and "messages" in state_snapshot.values:
                    messages = state_snapshot.values["messages"]
                    history.extend(messages)
            
            # Sort by timestamp and apply limit
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                history = history[:limit]
            elif self._max_history:
                history = history[:self._max_history]
            
            logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
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
        
        if not self._memory_enabled or not self._store:
            logger.debug("Memory disabled or store not available")
            return []
        
        try:
            namespace = self._get_user_namespace(soeid)
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Search user memories
            memories = await self._store.asearch(
                namespace=namespace,
                limit=limit or self._max_history
            )
            
            # Filter by date and extract conversations
            history = []
            for memory in memories:
                memory_data = memory.value
                if "conversation_snippet" in memory_data:
                    snippet = memory_data["conversation_snippet"]
                    timestamp_str = snippet.get("timestamp", "")
                    
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= cutoff_date:
                            history.append({
                                "role": "user",
                                "content": snippet["query"],
                                "timestamp": timestamp_str,
                                "session_id": snippet["session_id"],
                                "soeid": soeid
                            })
                            history.append({
                                "role": "assistant",
                                "content": snippet["response"],
                                "timestamp": timestamp_str,
                                "session_id": snippet["session_id"],
                                "soeid": soeid
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
                        continue
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            logger.debug(f"Retrieved {len(history)} cross-session messages for SOEID {soeid}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get chat history by SOEID: {e}")
            return []
    
    async def clear_session(self, session_id: str):
        """
        Clear conversation history for a specific session.
        
        Args:
            session_id: Session identifier to clear
        """
        await self._ensure_initialized()
        
        if not self._memory_enabled or not self._checkpointer:
            logger.debug("Memory disabled or checkpointer not available")
            return
        
        try:
            # Get thread configuration
            config = self._get_thread_config(session_id)
            
            # Delete thread
            await self._checkpointer.adelete_thread(config["configurable"]["thread_id"])
            
            logger.info(f"Cleared session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            raise
    
    async def clear_user_history(self, soeid: str):
        """
        Clear all history for a user across all sessions.
        
        Args:
            soeid: User SOEID
        """
        await self._ensure_initialized()
        
        if not self._memory_enabled:
            logger.debug("Memory disabled")
            return
        
        try:
            # Clear cross-thread memories
            if self._store:
                namespace = self._get_user_namespace(soeid)
                
                # Get all memories for user
                memories = await self._store.asearch(namespace=namespace)
                
                # Delete each memory
                for memory in memories:
                    await self._store.adelete(namespace=namespace, key=memory.key)
                
                logger.info(f"Cleared cross-thread memories for SOEID {soeid}")
            
            # Note: Clearing thread-scoped memories would require knowing all session IDs
            # This could be implemented by maintaining a user->sessions mapping
            
        except Exception as e:
            logger.error(f"Failed to clear user history for SOEID {soeid}: {e}")
            raise
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        await self._ensure_initialized()
        
        stats = {
            "memory_enabled": self._memory_enabled,
            "store_type": self._store_type,
            "checkpointer_available": self._checkpointer is not None,
            "store_available": self._store is not None,
            "max_history": self._max_history
        }
        
        try:
            if self._checkpointer:
                # Count total checkpoints (approximate)
                checkpoint_count = 0
                async for _ in self._checkpointer.alist({"configurable": {"thread_id": "*"}}):
                    checkpoint_count += 1
                stats["total_checkpoints"] = checkpoint_count
            
            if self._store:
                # Count total memories (approximate)
                memory_count = 0
                memories = await self._store.asearch(namespace=("*",))
                stats["total_memories"] = len(memories)
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self._checkpointer:
                # AsyncPostgresSaver handles its own connection cleanup
                pass
            
            if self._store:
                # AsyncPostgresStore handles its own connection cleanup
                pass
            
            self._initialized = False
            logger.info("LangGraph memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, '_connection_pool') and self._connection_pool:
            # Note: We can't call async cleanup in __del__, so just log
            logger.warning("Memory object deleted with active connection pool - call cleanup() explicitly")

# Alias for backward compatibility
LangGraphCheckpointMemory = ProductionLangGraphMemory
