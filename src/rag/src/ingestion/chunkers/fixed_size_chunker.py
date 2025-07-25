import logging
from typing import Dict, Any, List
import uuid

from src.rag.src.ingestion.chunkers.base_chunker import BaseChunker
from src.rag.src.core.interfaces.base import Document, Chunk
from src.rag.src.shared.models.schema import ChunkMetadata

logger = logging.getLogger(__name__)

class FixedSizeChunker(BaseChunker):
    """Chunker that splits documents into fixed-size chunks with optional overlap."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - chunk_size: Target size of each chunk in characters
                - chunk_overlap: Overlap between chunks in characters
        """
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
    
    async def _chunk_document(self, document: Document, config: Dict[str, Any]) -> List[Chunk]:
        """Split a document into fixed-size chunks.
        
        Args:
            document: Document to chunk
            config: Configuration parameters for chunking
            
        Returns:
            List of document chunks
        """
        # Get configuration from merged config
        chunk_size = config.get("chunk_size", self.chunk_size)
        chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        
        # Validate configuration
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        
        # Get document content and metadata
        content = document.content
        metadata = document.metadata
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        page_number = metadata.get("page_number", 1)
        
        # If content length is less than chunk size, return a single chunk
        if len(content) <= chunk_size:
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[page_number],
                chunk_index=0,
                extraction_method=metadata.get("extraction_method", "unknown"),
                chunk_type="text"
            ).dict()
            
            return [Chunk(
                content=content,
                metadata={**metadata, **chunk_metadata}
            )]
        
        # Split content into overlapping chunks
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(content):
            # Calculate end position for this chunk
            end = min(start + chunk_size, len(content))
            
            # Extract chunk content
            chunk_content = content[start:end]
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[page_number],
                chunk_index=chunk_index,
                extraction_method=metadata.get("extraction_method", "unknown"),
                chunk_type="text"
            ).dict()
            
            # Create chunk object
            chunk = Chunk(
                content=chunk_content,
                metadata={**metadata, **chunk_metadata}
            )
            chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - chunk_overlap if end < len(content) else len(content)
            chunk_index += 1
        
        logger.debug(f"Created {len(chunks)} fixed-size chunks from document {document_id}")
        return chunks
