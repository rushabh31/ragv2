"""
Simple text parser for plain text files.
"""

import logging
from typing import Dict, Any, List
import uuid
from pathlib import Path

from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.core.interfaces.base import Document

logger = logging.getLogger(__name__)


class SimpleTextParser(BaseDocumentParser):
    """Parser for simple text files (.txt, .md, etc.)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the simple text parser.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.supported_extensions = {'.txt', '.md', '.text'}
        logger.info("Simple text parser initialized")
    
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a text file and return documents.
        
        Args:
            file_path: Path to the text file
            config: Configuration parameters for parsing
            
        Returns:
            List of Document objects
        """
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty or whitespace-only file: {file_path}")
                return []
            
            # Extract metadata from config
            metadata = config.get('metadata', {})
            
            # Add file-specific metadata
            file_path_obj = Path(file_path)
            metadata.update({
                'file_name': file_path_obj.name,
                'file_extension': file_path_obj.suffix,
                'file_size': file_path_obj.stat().st_size,
                'parser_type': 'simple_text'
            })
            
            # Generate document ID if not provided
            document_id = metadata.get('document_id', str(uuid.uuid4()))
            metadata['document_id'] = document_id
            
            # Create document
            document = Document(
                content=content,
                metadata=metadata
            )
            
            logger.info(f"Successfully parsed text file: {file_path} ({len(content)} characters)")
            return [document]
            
        except Exception as e:
            error_msg = f"Failed to parse text file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the parser can handle this file type
        """
        file_path_obj = Path(file_path)
        return file_path_obj.suffix.lower() in self.supported_extensions
