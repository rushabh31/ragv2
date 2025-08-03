import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

# Import environment manager for configuration
from src.utils.env_manager import env
from src.rag.shared.utils.config_manager import ConfigManager
from src.rag.core.exceptions.exceptions import GenerationError, RetrievalError, RerankerError
from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever
from src.rag.chatbot.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.rag.chatbot.rerankers.custom_reranker import CustomReranker
from src.models.generation.model_factory import GenerationModelFactory

from src.rag.chatbot.workflow.workflow_manager import WorkflowManager
from src.rag.shared.models.schema import FullDocument, ChatMessage, ChatSession
from src.rag.shared.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service for handling chat interactions."""
    
    # Class-level singleton instances
    _memory_instance = None
    _memory_lock = asyncio.Lock()
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the chatbot service.
        
        Args:
            config: Configuration dictionary
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        self.sessions: Dict[str, ChatSession] = {}
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.config.get("cache", {}))
        
        # Initialize workflow manager
        self._workflow_manager = WorkflowManager(self.config)
        
        # Components will be initialized lazily
        self._retriever = None
        self._reranker = None
        self._generator = None
        self._memory = None
    
    async def _get_memory_instance(self):
        """Get or create the singleton memory instance.
        
        Returns:
            Memory instance
        """
        async with ChatbotService._memory_lock:
            if ChatbotService._memory_instance is None:
                from src.rag.chatbot.memory.memory_factory import MemoryFactory
                memory_config = self.config_manager.get_section("chatbot.memory", {})
                logger.info(f"Memory config being used: {memory_config}")
                ChatbotService._memory_instance = MemoryFactory.create_memory(memory_config)
                logger.info("Created singleton memory instance")
            return ChatbotService._memory_instance
    
    async def reset_memory_reference(self):
        """Reset the local memory reference (useful when singleton is reset)."""
        self._memory = None
    
    @classmethod
    async def reset_singleton_memory(cls):
        """Reset the singleton memory instance (useful for testing or reconfiguration)."""
        async with cls._memory_lock:
            if cls._memory_instance is not None:
                logger.info("Resetting singleton memory instance")
                # Call cleanup if the memory instance supports it
                if hasattr(cls._memory_instance, 'cleanup'):
                    try:
                        await cls._memory_instance.cleanup()
                        logger.info("Memory instance cleanup completed")
                    except Exception as e:
                        logger.error(f"Error during memory cleanup: {e}")
                cls._memory_instance = None
    
    async def _init_components(self):
        """Initialize components lazily."""
        # Only initialize if not already done
        if self._retriever is None:
            retriever_config = self.config_manager.get_section("chatbot.retrieval", {})
            self._retriever = VectorRetriever(retriever_config)
        
        if self._reranker is None:
            reranker_config = self.config_manager.get_section("chatbot.reranking", {})
            reranker_type = reranker_config.get("type", "cross_encoder")
            
            if reranker_type == "custom":
                self._reranker = CustomReranker(reranker_config)
            else:
                self._reranker = CrossEncoderReranker(reranker_config)
        
        if self._generator is None:
            # Use factory to create generator - it will automatically read config from YAML
            self._generator = GenerationModelFactory.create_model()
        
        if self._memory is None:
            # Use the singleton memory instance
            self._memory = await self._get_memory_instance()
    
    async def _get_or_create_session(self, session_id: Optional[str], user_id: str) -> str:
        """Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID
            user_id: User identifier
            
        Returns:
            Session ID
        """
        if session_id and session_id in self.sessions:
            # Update last active timestamp
            self.sessions[session_id].last_active = datetime.now()
            return session_id
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        now = datetime.now()
        self.sessions[new_session_id] = ChatSession(
            session_id=new_session_id,
            user_id=user_id,
            created_at=now,
            last_active=now
        )
        return new_session_id
    
    async def process_chat(self, 
                         user_id: str,
                         query: str,
                         session_id: Optional[str] = None,
                         use_retrieval: bool = True,
                         use_history: bool = True,
                         use_chat_history: bool = False,
                         chat_history_days: int = 7,
                         enable_memory: Optional[bool] = None,
                         enable_chat_history: Optional[bool] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a chat query and generate a response.
        
        Args:
            user_id: User identifier
            query: User query text
            session_id: Optional session identifier
            use_retrieval: Whether to use document retrieval
            use_history: Whether to use conversation history
            use_chat_history: Whether to include chat history from other sessions by SOEID
            chat_history_days: Number of days of chat history to include (1-365)
            enable_memory: Override memory enabled setting for this request
            enable_chat_history: Override chat history enabled setting for this request
            metadata: Additional metadata for the query
            
        Returns:
            Dictionary with response and metadata
        """
        # Get or create session
        session_id = await self._get_or_create_session(session_id, user_id)
        
        # Initialize components to access memory for enable/disable
        await self._init_components()
        
        # Handle memory enable/disable overrides
        if enable_memory is not None and hasattr(self._memory, 'set_enabled'):
            self._memory.set_enabled(enable_memory)
        
        if enable_chat_history is not None and hasattr(self._memory, 'set_chat_history_enabled'):
            self._memory.set_chat_history_enabled(enable_chat_history)
        
        try:
            # Use the workflow manager to process the query using LangGraph
            workflow_params = {
                "use_retrieval": use_retrieval,
                "use_history": use_history,
                "use_chat_history": use_chat_history,
                "chat_history_days": chat_history_days,
                "enable_memory": enable_memory,
                "enable_chat_history": enable_chat_history,
                "metadata": metadata or {}
            }
            
            logger.info(f"Processing chat with workflow manager for session {session_id}")
            result = await self._workflow_manager.process_query(
                user_id=user_id,
                query=query,
                session_id=session_id,
                workflow_params=workflow_params
            )
            
            # Convert the workflow result to the expected response format
            response_data = {
                "session_id": session_id,
                "query": query,
                "response": result.get("response", "I'm sorry, I couldn't generate a response."),
                "created_at": datetime.now(),
                "retrieved_documents": result.get("retrieved_documents", []),
                "metadata": {
                    **(metadata or {}),
                    "workflow_run_id": result.get("workflow_run_id"),
                    "metrics": result.get("metrics", {})
                }
            }
            
            # If there was an error, include it in the response
            if "error" in result:
                response_data["metadata"]["error"] = result["error"]
            
            return response_data
            
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
            # Return error response
            error_message = "I'm sorry, I encountered an error processing your request."
            
            if isinstance(e, RetrievalError):
                error_message = "I'm having trouble retrieving relevant information. Please try again."
            elif isinstance(e, GenerationError):
                error_message = "I'm having trouble generating a response. Please try again."
            
            return {
                "session_id": session_id,
                "query": query,
                "response": error_message,
                "created_at": datetime.now(),
                "retrieved_documents": [],
                "metadata": {
                    **(metadata or {}),
                    "error": str(e)
                }
            }
    
    async def get_session_history(self, session_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            Dictionary with session history
        """
        await self._init_components()
        
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found in active sessions")
            # Still try to get from memory as the session might exist there
            # but not be active anymore
        
        try:
            raw_messages = await self._memory.get_history(session_id, limit)
            logger.debug(f"Retrieved {len(raw_messages)} messages from memory for session {session_id}")
            
            # Convert the raw message dictionaries to ChatMessage format
            formatted_messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp")
                } 
                for msg in raw_messages
            ]
            
            return {
                "session_id": session_id,
                "messages": formatted_messages,
                "metadata": {}
            }
        except Exception as e:
            logger.error(f"Error retrieving session history: {str(e)}", exc_info=True)
            return {
                "session_id": session_id,
                "messages": [],
                "metadata": {"error": str(e)}
            }
    
    async def get_user_history_by_soeid(self, soeid: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get all conversation history for a user by SOEID across all sessions.
        
        Args:
            soeid: Source of Entity ID for the user
            limit: Maximum number of messages to return
            
        Returns:
            Dictionary with user history across all sessions
        """
        await self._init_components()
        
        try:
            # Get user history by SOEID
            raw_messages = await self._memory.get_user_history_by_soeid(soeid, limit)
            logger.debug(f"Retrieved {len(raw_messages)} messages from memory for user {soeid}")
            
            # Convert the raw message dictionaries to UserHistoryMessage format
            formatted_messages = []
            session_ids = set()
            
            for msg in raw_messages:
                session_id = msg.get("session_id", "unknown")
                session_ids.add(session_id)
                
                formatted_messages.append({
                    "session_id": session_id,
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp"),
                    "metadata": msg.get("metadata", {})
                })
            
            return {
                "soeid": soeid,
                "messages": formatted_messages,
                "total_messages": len(formatted_messages),
                "total_sessions": len(session_ids),
                "metadata": {
                    "session_ids": list(session_ids),
                    "retrieved_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error retrieving user history by SOEID: {str(e)}", exc_info=True)
            return {
                "soeid": soeid,
                "messages": [],
                "total_messages": 0,
                "total_sessions": 0,
                "metadata": {"error": str(e)}
            }
    
    async def list_sessions(self, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """List active sessions for a user.
        
        Args:
            user_id: User identifier
            page: Page number
            page_size: Number of sessions per page
            
        Returns:
            Dictionary with sessions and pagination info
        """
        # Filter sessions by user_id
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        
        # Sort by last active timestamp, most recent first
        user_sessions.sort(key=lambda s: s.last_active, reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        paginated_sessions = user_sessions[start:end]
        
        return {
            "sessions": paginated_sessions,
            "total": len(user_sessions),
            "page": page,
            "page_size": page_size
        }
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session and its history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        await self._init_components()
        
        if session_id not in self.sessions:
            return False
        
        try:
            # Clear memory
            await self._memory.clear_session(session_id)
            # Remove session
            del self.sessions[session_id]
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}", exc_info=True)
            return False
    
    async def submit_feedback(self, 
                            session_id: str, 
                            message_id: str, 
                            feedback_type: str, 
                            score: Optional[float] = None, 
                            comment: Optional[str] = None) -> bool:
        """Submit feedback for a chat response.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            feedback_type: Type of feedback (thumbs_up, thumbs_down, etc.)
            score: Optional numerical score
            comment: Optional text comment
            
        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would store feedback in a database
        # For now, just log the feedback
        logger.info(f"Feedback received - session: {session_id}, message: {message_id}, " 
                   f"type: {feedback_type}, score: {score}, comment: {comment}")
        return True

    async def get_all_history_by_soeid(self, soeid: str) -> Dict[str, Any]:
        """Get all chat history for a SOEID, grouped by session, with soeid in every message."""
        await self._init_components()
        try:
            # Get all sessions (with all messages) for the SOEID
            sessions = await self._memory.get_user_history_by_soeid(soeid)
            total_sessions = len(sessions)
            total_messages = sum(len(s["messages"]) for s in sessions)
            return {
                "soeid": soeid,
                "sessions": sessions,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "metadata": {"retrieved_at": datetime.now().isoformat()}
            }
        except Exception as e:
            logger.error(f"Error retrieving all history by SOEID: {str(e)}", exc_info=True)
            return {
                "soeid": soeid,
                "sessions": [],
                "total_sessions": 0,
                "total_messages": 0,
                "metadata": {"error": str(e)}
            }

    async def get_session_history_with_soeid(self, session_id: str) -> Dict[str, Any]:
        """Get all chat history for a session_id, including soeid in every message."""
        await self._init_components()
        try:
            raw_messages = await self._memory.get_history(session_id)
            # Try to get soeid from the first message with soeid, or from metadata
            soeid = None
            for msg in raw_messages:
                if msg.get("soeid"):
                    soeid = msg["soeid"]
                    break
            if soeid is None:
                soeid = ""
            chat_messages = [
                {"role": msg["role"], "content": msg["content"], "timestamp": msg.get("timestamp"), "soeid": msg.get("soeid", soeid)}
                for msg in raw_messages
            ]
            return {
                "session_id": session_id,
                "soeid": soeid,
                "messages": chat_messages,
                "metadata": {"retrieved_at": datetime.now().isoformat()}
            }
        except Exception as e:
            logger.error(f"Error retrieving session history with soeid: {str(e)}", exc_info=True)
            return {
                "session_id": session_id,
                "soeid": "",
                "messages": [],
                "metadata": {"error": str(e)}
            }

    async def delete_history_by_soeid(self, soeid: str) -> bool:
        await self._init_components()
        return await self._memory.clear_user_history(soeid)

    async def delete_history_by_session_id(self, session_id: str) -> bool:
        await self._init_components()
        return await self._memory.clear_session(session_id)
    
    async def get_sessions_for_soeid(self, soeid: str) -> Dict[str, Any]:
        """Get all sessions for a specific SOEID with metadata."""
        await self._init_components()
        try:
            # Get all sessions for the SOEID
            sessions_data = await self._memory.get_user_history_by_soeid(soeid)
            
            # Extract session metadata
            sessions = []
            for session_data in sessions_data:
                session_info = {
                    "session_id": session_data["session_id"],
                    "message_count": session_data["message_count"],
                    "created_at": session_data.get("created_at"),
                    "last_active": session_data.get("last_active")
                }
                sessions.append(session_info)
            
            return {
                "soeid": soeid,
                "sessions": sessions,
                "total_sessions": len(sessions),
                "metadata": {"retrieved_at": datetime.now().isoformat()}
            }
        except Exception as e:
            logger.error(f"Error retrieving sessions for SOEID: {str(e)}", exc_info=True)
            return {
                "soeid": soeid,
                "sessions": [],
                "total_sessions": 0,
                "metadata": {"error": str(e)}
            }
    
    async def get_all_threads(self) -> Dict[str, Any]:
        """Get all threads (sessions) in the system."""
        await self._init_components()
        try:
            # Check if the memory implementation supports listing threads
            if hasattr(self._memory, '_list_thread_ids'):
                thread_ids = await self._memory._list_thread_ids()
                
                threads = []
                for thread_id in thread_ids:
                    # Get basic info for each thread
                    messages = await self._memory.get_history(thread_id, limit=1)
                    if messages:
                        thread_info = {
                            "thread_id": thread_id,
                            "session_id": thread_id,  # Same as thread_id in our case
                            "soeid": messages[0].get("soeid", "unknown"),
                            "last_message_time": messages[0].get("timestamp"),
                            "has_messages": len(messages) > 0
                        }
                        threads.append(thread_info)
                
                return {
                    "threads": threads,
                    "total_threads": len(threads),
                    "metadata": {"retrieved_at": datetime.now().isoformat()}
                }
            else:
                # Fallback for memory implementations that don't support thread listing
                return {
                    "threads": [],
                    "total_threads": 0,
                    "metadata": {"error": "Thread listing not supported by current memory implementation"}
                }
        except Exception as e:
            logger.error(f"Error retrieving all threads: {str(e)}", exc_info=True)
            return {
                "threads": [],
                "total_threads": 0,
                "metadata": {"error": str(e)}
            }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        await self._init_components()
        try:
            # Get basic memory info
            memory_type = type(self._memory).__name__
            store_type = getattr(self._memory, '_store_type', 'unknown')
            memory_enabled = getattr(self._memory, 'is_enabled', lambda: True)()
            chat_history_enabled = getattr(self._memory, 'is_chat_history_enabled', lambda: True)()
            
            # Initialize counters
            total_sessions = 0
            total_messages = 0
            unique_soeids = set()
            oldest_session = None
            newest_session = None
            
            # Check if we can get thread listing
            if hasattr(self._memory, '_list_thread_ids'):
                thread_ids = await self._memory._list_thread_ids()
                total_sessions = len(thread_ids)
                
                # Analyze each thread
                for thread_id in thread_ids:
                    messages = await self._memory.get_history(thread_id)
                    total_messages += len(messages)
                    
                    for message in messages:
                        # Collect SOEIDs
                        soeid = message.get("soeid")
                        if soeid:
                            unique_soeids.add(soeid)
                        
                        # Track session timestamps
                        timestamp_str = message.get("timestamp")
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if oldest_session is None or timestamp < oldest_session:
                                    oldest_session = timestamp
                                if newest_session is None or timestamp > newest_session:
                                    newest_session = timestamp
                            except Exception:
                                continue
            
            return {
                "memory_type": memory_type,
                "store_type": store_type,
                "memory_enabled": memory_enabled,
                "chat_history_enabled": chat_history_enabled,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "unique_soeids": len(unique_soeids),
                "oldest_session": oldest_session,
                "newest_session": newest_session,
                "metadata": {"retrieved_at": datetime.now().isoformat()}
            }
        except Exception as e:
            logger.error(f"Error retrieving memory stats: {str(e)}", exc_info=True)
            return {
                "memory_type": "unknown",
                "store_type": "unknown",
                "memory_enabled": False,
                "chat_history_enabled": False,
                "total_sessions": 0,
                "total_messages": 0,
                "unique_soeids": 0,
                "oldest_session": None,
                "newest_session": None,
                "metadata": {"error": str(e)}
            }
    
    async def set_memory_enabled(self, enabled: bool) -> bool:
        """Enable or disable memory system.
        
        Args:
            enabled: Whether to enable memory
            
        Returns:
            True if successful, False otherwise
        """
        await self._init_components()
        try:
            if hasattr(self._memory, 'set_enabled'):
                self._memory.set_enabled(enabled)
                logger.info(f"Memory enabled set to: {enabled}")
                return True
            else:
                logger.warning("Memory implementation does not support enable/disable")
                return False
        except Exception as e:
            logger.error(f"Failed to set memory enabled: {str(e)}", exc_info=True)
            return False
    
    async def set_chat_history_enabled(self, enabled: bool) -> bool:
        """Enable or disable chat history.
        
        Args:
            enabled: Whether to enable chat history
            
        Returns:
            True if successful, False otherwise
        """
        await self._init_components()
        try:
            if hasattr(self._memory, 'set_chat_history_enabled'):
                self._memory.set_chat_history_enabled(enabled)
                logger.info(f"Chat history enabled set to: {enabled}")
                return True
            else:
                logger.warning("Memory implementation does not support chat history enable/disable")
                return False
        except Exception as e:
            logger.error(f"Failed to set chat history enabled: {str(e)}", exc_info=True)
            return False
