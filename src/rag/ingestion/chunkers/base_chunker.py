import logging
from abc import abstractmethod
from typing import Dict, Any, List

from src.rag.core.interfaces.base import Chunker, Document, Chunk
from src.rag.core.exceptions.exceptions import ChunkingError

logger = logging.getLogger(__name__)

class BaseChunker(Chunker):
    """Base class for document chunking strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary for the chunker
        """
        self.config = config or {}
    
    @abstractmethod
    async def _chunk_document(self, document: Document, config: Dict[str, Any]) -> List[Chunk]:
        """Implement this method in subclasses for specific chunking strategies.
        
        Args:
            document: Document to chunk
            config: Configuration parameters for chunking
            
        Returns:
            List of document chunks
        """
        pass
    
    async def chunk(self, documents: List[Document], config: Dict[str, Any]) -> List[Chunk]:
        """Split documents into chunks according to the strategy.
        
        Args:
            documents: List of documents to chunk
            config: Configuration parameters for chunking
            
        Returns:
            List of document chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        try:
            # Merge provided config with default config
            merged_config = {**self.config, **(config or {})}
            
            # Process each document
            all_chunks = []
            for document in documents:
                logger.debug(f"Chunking document with ID: {document.metadata.get('document_id', 'unknown')}")
                chunks = await self._chunk_document(document, merged_config)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            return all_chunks
            
        except Exception as e:
            error_msg = f"Chunking failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ChunkingError(error_msg) from e
