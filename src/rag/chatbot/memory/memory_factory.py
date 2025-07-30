"""Memory factory for creating different memory implementations."""

import logging
from typing import Dict, Any, Optional

from src.rag.chatbot.memory.base_memory import BaseMemory
from src.rag.chatbot.memory.simple_memory import SimpleMemory
from src.rag.chatbot.memory.mem0_memory import Mem0Memory
from src.rag.chatbot.memory.langgraph_memory import LangGraphMemory
from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
from src.rag.chatbot.memory.no_checkpoint_memory import NoCheckpointMemory
from src.rag.core.exceptions.exceptions import MemoryError

logger = logging.getLogger(__name__)

class MemoryFactory:
    """Factory for creating memory implementations."""
    
    _MEMORY_TYPES = {
        "simple": SimpleMemory,
        "mem0": Mem0Memory,
        "langgraph": LangGraphMemory,
        "langgraph_checkpoint": LangGraphCheckpointMemory,
        "no_checkpoint": NoCheckpointMemory
    }
    
    @classmethod
    def create_memory(cls, config: Dict[str, Any]) -> BaseMemory:
        """Create a memory implementation based on configuration.
        
        Args:
            config: Configuration dictionary containing memory settings
            
        Returns:
            BaseMemory: The configured memory implementation
            
        Raises:
            MemoryError: If the memory type is not supported or initialization fails
        """
        try:
            memory_type = config.get("type", "simple")
            
            if memory_type not in cls._MEMORY_TYPES:
                supported_types = list(cls._MEMORY_TYPES.keys())
                raise MemoryError(
                    f"Unsupported memory type: {memory_type}. "
                    f"Supported types: {supported_types}"
                )
            
            memory_class = cls._MEMORY_TYPES[memory_type]
            memory_instance = memory_class(config)
            
            logger.info(f"Created memory implementation: {memory_type}")
            return memory_instance
            
        except Exception as e:
            error_msg = f"Failed to create memory implementation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MemoryError(error_msg) from e
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported memory types.
        
        Returns:
            List of supported memory type names
        """
        return list(cls._MEMORY_TYPES.keys())
    
    @classmethod
    def register_memory_type(cls, name: str, memory_class: type) -> None:
        """Register a new memory type.
        
        Args:
            name: Name of the memory type
            memory_class: Memory class to register
        """
        if not issubclass(memory_class, BaseMemory):
            raise ValueError(f"Memory class must inherit from BaseMemory: {memory_class}")
        
        cls._MEMORY_TYPES[name] = memory_class
        logger.info(f"Registered new memory type: {name}") 