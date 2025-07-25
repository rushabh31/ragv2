import logging
import re
import uuid
from typing import Dict, Any, List

from src.rag.src.ingestion.chunkers.base_chunker import BaseChunker
from src.rag.src.core.interfaces.base import Document, Chunk
from src.rag.src.shared.models.schema import ChunkMetadata

logger = logging.getLogger(__name__)

class PageBasedChunker(BaseChunker):
    """Chunker that splits documents by page number found in markdown content."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary with optional keys:
                - min_chunk_size: Minimum size of a chunk in characters (default: 100)
                - use_headers_fallback: Whether to use headers as fallback if no page numbers found (default: True)
        """
        super().__init__(config)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.use_headers_fallback = self.config.get("use_headers_fallback", True)
        
        # Regex patterns for page detection
        self.page_patterns = [
            r'(?:Page|PAGE)\s+(\d+)',  # "Page 1", "PAGE 1"
            r'\n\s*[-–—]\s*(\d+)\s*[-–—]',  # "- 1 -"
            r'\n\s*(\d+)\s*\n'  # Standalone page numbers like "\n 1 \n"
        ]
        
        # Regex pattern for headers
        self.header_pattern = r'\n#+\s+([^\n]+)'
    
    async def _chunk_document(self, document: Document, config: Dict[str, Any]) -> List[Chunk]:
        """Split a document based on page numbers in markdown.
        
        Args:
            document: Document to chunk
            config: Configuration parameters for chunking
            
        Returns:
            List of document chunks
        """
        # Get configuration from merged config
        min_chunk_size = config.get("min_chunk_size", self.min_chunk_size)
        use_headers_fallback = config.get("use_headers_fallback", self.use_headers_fallback)
        
        # Get document content and metadata
        content = document.content
        metadata = document.metadata or {}
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        
        # If document already has a page_number, use it directly to create a single chunk
        if "page_number" in metadata:
            page_number = metadata.get("page_number")
            logger.debug(f"Using document's page_number {page_number} for document {document_id}")
            
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[page_number],
                chunk_index=0,
                extraction_method=metadata.get("extraction_method", "page_based"),
                chunk_type="page"
            ).dict()
            
            # Add fields for pgvector compatibility
            chunk_metadata.update({
                'page_number': page_number,  # Single page number for pgvector
                'file_name': metadata.get('file_name', metadata.get('filename', 'unknown')),
                'document_id': document_id
            })
            
            return [Chunk(
                content=content,
                metadata={**metadata, **chunk_metadata}
            )]
            
        # Look for page boundaries in content
        page_boundaries = []
        for pattern in self.page_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                try:
                    page_num = int(match.group(1))
                    page_boundaries.append((match.start(), page_num))
                except (IndexError, ValueError):
                    continue
        
        # If no page boundaries found and use_headers_fallback is True, try headers
        if not page_boundaries and use_headers_fallback:
            logger.debug(f"No page numbers found in document {document_id}, using headers as fallback")
            header_matches = list(re.finditer(self.header_pattern, content))
            for i, match in enumerate(header_matches):
                # Use 1-based indexing for header-based chunks
                page_boundaries.append((match.start(), i + 1))
        
        # Sort boundaries by position
        page_boundaries.sort()
        
        # If still no boundaries, return the whole document as a single chunk
        if not page_boundaries:
            logger.debug(f"No page numbers or headers found in document {document_id}, using single chunk")
            
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[1],  # Default to page 1
                chunk_index=0,
                extraction_method=metadata.get("extraction_method", "page_based"),
                chunk_type="document"
            ).dict()
            
            # Add fields for pgvector compatibility
            chunk_metadata.update({
                'page_number': 1,  # Single page number for pgvector
                'file_name': metadata.get('file_name', metadata.get('filename', 'unknown')),
                'document_id': document_id
            })
            
            return [Chunk(
                content=content,
                metadata={**metadata, **chunk_metadata}
            )]
        
        # Create chunks based on page boundaries
        chunks = []
        for i, (start_pos, page_num) in enumerate(page_boundaries):
            # Determine end position
            if i < len(page_boundaries) - 1:
                end_pos = page_boundaries[i + 1][0]
            else:
                end_pos = len(content)
                
            # Get chunk content
            chunk_content = content[start_pos:end_pos].strip()
            
            # Skip chunks that are too small
            if len(chunk_content) < min_chunk_size:
                continue
                
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[page_num],
                chunk_index=i,
                extraction_method=metadata.get("extraction_method", "page_based"),
                chunk_type="page"
            ).dict()
            
            # Add fields for pgvector compatibility
            chunk_metadata.update({
                'page_number': page_num,  # Single page number for pgvector
                'file_name': metadata.get('file_name', metadata.get('filename', 'unknown')),
                'document_id': document_id,
                'chunk_index': i
            })
            
            # Create chunk
            chunk = Chunk(
                content=chunk_content,
                metadata={**metadata, **chunk_metadata}
            )
            chunks.append(chunk)
        
        # If no chunks were created, return whole document as a single chunk
        if not chunks:
            logger.debug(f"No chunks created for document {document_id}, using single chunk")
            
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[1],  # Default to page 1
                chunk_index=0,
                extraction_method=metadata.get("extraction_method", "page_based"),
                chunk_type="document"
            ).dict()
            
            # Add fields for pgvector compatibility
            chunk_metadata.update({
                'page_number': 1,  # Single page number for pgvector
                'file_name': metadata.get('file_name', metadata.get('filename', 'unknown')),
                'document_id': document_id
            })
            
            return [Chunk(
                content=content,
                metadata={**metadata, **chunk_metadata}
            )]
        
        logger.debug(f"Created {len(chunks)} page-based chunks from document {document_id}")
        return chunks
