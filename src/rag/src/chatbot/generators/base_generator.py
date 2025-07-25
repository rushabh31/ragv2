import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.src.core.interfaces.base import Generator, Document
from src.rag.src.core.exceptions.exceptions import GenerationError
from src.rag.src.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BaseGenerator(Generator):
    """Base class for response generators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the generator with configuration.
        
        Args:
            config: Configuration dictionary for the generator
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2048)
    
    @abstractmethod
    async def _generate_response(self, 
                               query: str, 
                               documents: List[Document], 
                               conversation_history: Optional[List[Dict[str, str]]] = None, 
                               config: Dict[str, Any] = None) -> str:
        """Implement this method in subclasses for specific generation strategies.
        
        Args:
            query: User query string
            documents: List of relevant documents
            conversation_history: Optional conversation history
            config: Configuration parameters for generation
            
        Returns:
            Generated response
        """
        pass
    
    async def generate(self, 
                     query: str, 
                     documents: List[Document], 
                     conversation_history: Optional[List[Dict[str, str]]] = None, 
                     config: Dict[str, Any] = None) -> str:
        """Generate a response based on the query and relevant documents.
        
        Args:
            query: User query string
            documents: List of relevant documents
            conversation_history: Optional conversation history
            config: Configuration parameters for generation
            
        Returns:
            Generated response
            
        Raises:
            GenerationError: If response generation fails
        """
        try:
            # Skip empty query
            if not query.strip():
                logger.warning("Empty query provided for response generation")
                return "I don't understand your question. Could you please rephrase it?"
            
            # Merge provided config with default config
            merged_config = {**self.config, **(config or {})}
            
            # Generate response
            logger.debug(f"Generating response for query: {query}")
            response = await self._generate_response(query, documents, conversation_history, merged_config)
            
            return response
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise GenerationError(error_msg) from e
