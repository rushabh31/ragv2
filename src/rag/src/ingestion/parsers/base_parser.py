import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.rag.src.core.interfaces.base import DocumentParser, Document
from src.rag.src.core.exceptions.exceptions import DocumentProcessingError
from src.rag.src.shared.models.schema import DocumentType

logger = logging.getLogger(__name__)

class BaseDocumentParser(DocumentParser):
    """Base class for document parsers with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with configuration.
        
        Args:
            config: Configuration dictionary for the parser
        """
        self.config = config or {}
    
    @abstractmethod
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Implement this method in subclasses to parse specific file types.
        
        Args:
            file_path: Path to the document file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects
        """
        pass
    
    async def parse(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a document file into a list of Document objects.
        
        Args:
            file_path: Path to the document file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects
            
        Raises:
            DocumentProcessingError: If document parsing fails
        """
        try:
            # Ensure file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            # Merge provided config with default config
            merged_config = {**self.config, **(config or {})}
            
            logger.info(f"Parsing document: {file_path}")
            return await self._parse_file(file_path, merged_config)
            
        except Exception as e:
            error_msg = f"Failed to parse document {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    @staticmethod
    def get_document_type(file_path: str) -> DocumentType:
        """Determine document type from file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentType enum value
        """
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext == '.docx':
            return DocumentType.DOCX
        elif ext == '.xlsx':
            return DocumentType.XLSX
        elif ext == '.txt':
            return DocumentType.TXT
        elif ext in ['.html', '.htm']:
            return DocumentType.HTML
        elif ext in ['.md', '.markdown']:
            return DocumentType.MD
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return DocumentType.IMAGE
        else:
            return DocumentType.UNKNOWN
