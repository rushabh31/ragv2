import logging
import re
from typing import Dict, Any, List
import uuid
from src.rag.ingestion.chunkers.base_chunker import BaseChunker
from src.rag.core.interfaces.base import Document, Chunk
from src.rag.shared.models.schema import ChunkMetadata
from src.rag.shared.utils.vertex_ai import VertexGenAI

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    """Chunker that splits documents at semantic boundaries."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - max_chunk_size: Maximum size of each chunk in characters
                - min_chunk_size: Minimum size of each chunk in characters
                - use_llm_boundary: Whether to use LLM to determine optimal chunk boundaries
        """
        super().__init__(config)
        self.max_chunk_size = self.config.get("max_chunk_size", 1500)
        self.min_chunk_size = self.config.get("min_chunk_size", 300)
        self.use_llm_boundary = self.config.get("use_llm_boundary", False)
        self._vertex_ai = None

    async def _init_vertex_ai(self):
        """Initialize VertexAI client lazily."""
        if self._vertex_ai is None and self.use_llm_boundary:
            self._vertex_ai = VertexGenAI()
    
    async def _chunk_document(self, document: Document, config: Dict[str, Any]) -> List[Chunk]:
        """Split a document into semantic chunks based on content structure.
        
        Args:
            document: Document to chunk
            config: Configuration parameters for chunking
            
        Returns:
            List of document chunks
        """
        # Get configuration from merged config
        max_chunk_size = config.get("max_chunk_size", self.max_chunk_size)
        min_chunk_size = config.get("min_chunk_size", self.min_chunk_size)
        use_llm_boundary = config.get("use_llm_boundary", self.use_llm_boundary)
        
        # Validate configuration
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if min_chunk_size <= 0 or min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size must be positive and less than max_chunk_size")
        
        # Get document content and metadata
        content = document.content
        metadata = document.metadata
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        page_number = metadata.get("page_number", 1)
        
        # If content is very short, return as a single chunk
        if len(content) <= max_chunk_size:
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
        
        # Split content at semantic boundaries
        if use_llm_boundary:
            await self._init_vertex_ai()
            chunks = await self._chunk_with_llm(content, document_id, page_number, metadata, max_chunk_size)
        else:
            chunks = self._chunk_with_heuristics(content, document_id, page_number, metadata, max_chunk_size, min_chunk_size)
        
        logger.debug(f"Created {len(chunks)} semantic chunks from document {document_id}")
        return chunks
    
    async def _chunk_with_llm(self, content: str, document_id: str, page_number: int, metadata: Dict[str, Any], max_chunk_size: int) -> List[Chunk]:
        """Use LLM to determine optimal chunk boundaries.
        
        Args:
            content: Document content to chunk
            document_id: Document identifier
            page_number: Page number of the document
            metadata: Document metadata
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        
        # First split content into rough sections to fit within LLM context
        rough_sections = self._split_content_by_size(content, max_chunk_size * 2)
        
        for section_index, section in enumerate(rough_sections):
            # If section is small enough, use it directly
            if len(section) <= max_chunk_size:
                chunk_metadata = ChunkMetadata(
                    document_id=document_id,
                    page_numbers=[page_number],
                    chunk_index=len(chunks),
                    extraction_method=metadata.get("extraction_method", "unknown"),
                    chunk_type="text"
                ).dict()
                
                chunks.append(Chunk(
                    content=section,
                    metadata={**metadata, **chunk_metadata}
                ))
                continue
            
            # Ask LLM to identify logical splitting points
            prompt = f"""
            You are an AI trained to identify logical breaking points in text content.
            I will provide you a section of content that needs to be split into smaller semantic chunks.
            Each chunk should be a coherent unit that preserves meaning and context.
            The maximum chunk size is {max_chunk_size} characters.
            
            Please identify the optimal splitting points in this content. 
            Return only line numbers where splits should occur, one number per line.
            
            Content:
            ```
            {section}
            ```
            """
            
            try:
                response = await self._vertex_ai.generate_content(
                    prompt=prompt,
                    generation_config={"temperature": 0.0}
                )
                
                # Parse line numbers from response
                split_points = []
                for line in response.text.strip().split('\n'):
                    try:
                        point = int(line.strip())
                        split_points.append(point)
                    except (ValueError, TypeError):
                        continue
                
                # Split the section at the identified points
                section_lines = section.split('\n')
                start_line = 0
                
                for end_line in sorted(split_points):
                    if end_line > start_line and end_line < len(section_lines):
                        chunk_content = '\n'.join(section_lines[start_line:end_line])
                        
                        chunk_metadata = ChunkMetadata(
                            document_id=document_id,
                            page_numbers=[page_number],
                            chunk_index=len(chunks),
                            extraction_method=metadata.get("extraction_method", "unknown"),
                            chunk_type="text"
                        ).dict()
                        
                        chunks.append(Chunk(
                            content=chunk_content,
                            metadata={**metadata, **chunk_metadata}
                        ))
                        
                        start_line = end_line
                
                # Add the final chunk
                if start_line < len(section_lines):
                    chunk_content = '\n'.join(section_lines[start_line:])
                    
                    chunk_metadata = ChunkMetadata(
                        document_id=document_id,
                        page_numbers=[page_number],
                        chunk_index=len(chunks),
                        extraction_method=metadata.get("extraction_method", "unknown"),
                        chunk_type="text"
                    ).dict()
                    
                    chunks.append(Chunk(
                        content=chunk_content,
                        metadata={**metadata, **chunk_metadata}
                    ))
            
            except Exception as e:
                logger.error(f"LLM chunking failed: {str(e)}")
                # Fall back to heuristic chunking
                section_chunks = self._chunk_with_heuristics(section, document_id, page_number, metadata, max_chunk_size, self.min_chunk_size, base_index=len(chunks))
                chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_with_heuristics(self, content: str, document_id: str, page_number: int, metadata: Dict[str, Any], 
                              max_chunk_size: int, min_chunk_size: int, base_index: int = 0) -> List[Chunk]:
        """Split content into chunks using heuristic rules.
        
        Args:
            content: Document content to chunk
            document_id: Document identifier
            page_number: Page number of the document
            metadata: Document metadata
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
            base_index: Starting index for chunks
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        
        # Define semantic boundary patterns in order of precedence
        patterns = [
            r'\n#{1,6}\s',              # Headings (markdown)
            r'\n\n',                    # Paragraph breaks
            r'\.\s+(?=[A-Z])',          # End of sentences
            r';\s+(?=[A-Z])',           # Semicolons followed by capital letter
            r',\s+(?=and |or |but )',   # Commas before conjunctions
        ]
        
        # Split content if longer than max size
        if len(content) <= max_chunk_size:
            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                page_numbers=[page_number],
                chunk_index=base_index,
                extraction_method=metadata.get("extraction_method", "unknown"),
                chunk_type="text"
            ).dict()
            
            return [Chunk(
                content=content,
                metadata={**metadata, **chunk_metadata}
            )]
        
        # Start chunking
        start = 0
        chunk_index = base_index
        
        while start < len(content):
            # Try to find the best split point within max_chunk_size
            end = start + max_chunk_size if start + max_chunk_size < len(content) else len(content)
            best_split = end
            
            # Try each pattern to find a good split point
            for pattern in patterns:
                matches = list(re.finditer(pattern, content[start:end]))
                if matches:
                    # Take the last match as split point
                    last_match = matches[-1]
                    candidate_split = start + last_match.start() + 1  # +1 to include the newline or period
                    
                    # Check if it's not too close to start to ensure minimum chunk size
                    if candidate_split - start >= min_chunk_size:
                        best_split = candidate_split
                        break
            
            # Extract chunk content
            chunk_content = content[start:best_split]
            
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
            
            # Move to next chunk
            start = best_split
            chunk_index += 1
            
            # If at the end and there's a small remaining piece, merge with previous
            if len(content) - start <= min_chunk_size and chunks:
                last_chunk = chunks[-1]
                last_chunk.content += content[start:]
                break
        
        return chunks
    
    def _split_content_by_size(self, content: str, size: int) -> List[str]:
        """Split content into roughly equal sections by size.
        
        Args:
            content: Content to split
            size: Target size for each section
            
        Returns:
            List of content sections
        """
        sections = []
        start = 0
        
        while start < len(content):
            end = min(start + size, len(content))
            
            # Try to find a good boundary if not at the end
            if end < len(content):
                # Look for a paragraph break
                paragraph_break = content.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + size // 2:
                    end = paragraph_break + 2  # Include the double newline
                else:
                    # Look for a newline
                    newline = content.rfind('\n', start, end)
                    if newline != -1 and newline > start + size // 2:
                        end = newline + 1  # Include the newline
            
            sections.append(content[start:end])
            start = end
        
        return sections
