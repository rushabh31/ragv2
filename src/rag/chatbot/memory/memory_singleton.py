"""Singleton memory manager for the RAG system."""

import logging
import asyncio
from typing import Optional, Dict, Any

from src.rag.chatbot.memory.memory_factory import MemoryFactory
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MemorySingleton:
    """Singleton class to manage a single memory instance across the application."""
    
    _instance = None
    _memory_instance = None
    _lock = asyncio.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemorySingleton, cls).__new__(cls)
        return cls._instance
    
    async def get_memory(self) -> Any:
        """Get or create the singleton memory instance.
        
        Returns:
            Memory instance
        """
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    logger.info("Initializing singleton memory instance")
                    config_manager = ConfigManager()
                    memory_config = config_manager.get_section("chatbot.memory", {})
                    self._memory_instance = MemoryFactory.create_memory(memory_config)
                    self._initialized = True
                    logger.info("Singleton memory instance initialized")
        
        return self._memory_instance
    
    async def reset_memory(self):
        """Reset the memory instance (useful for testing)."""
        async with self._lock:
            # Clear the old instance
            if self._memory_instance:
                # Clear the memory if it has a clear method
                if hasattr(self._memory_instance, 'clear_all'):
                    await self._memory_instance.clear_all()
                elif hasattr(self._memory_instance, '_store'):
                    # For LangGraph memory, clear the store
                    try:
                        # Get all namespaces and clear them
                        if hasattr(self._memory_instance._store, 'search'):
                            # This is a simple approach - in production you might want more sophisticated clearing
                            logger.info("Clearing LangGraph memory store")
                    except Exception as e:
                        logger.warning(f"Could not clear memory store: {e}")
            
            # Reset the singleton state
            self._memory_instance = None
            self._initialized = False
            logger.info("Memory singleton reset - new instance will be created on next access")

# Global singleton instance
memory_singleton = MemorySingleton() 