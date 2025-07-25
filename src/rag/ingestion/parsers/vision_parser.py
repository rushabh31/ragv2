import base64
import logging
import os
import uuid
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from datetime import datetime
import json

from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.core.interfaces.base import Document
from src.rag.core.exceptions.exceptions import DocumentProcessingError
from src.rag.shared.models.schema import DocumentType, DocumentMetadata, PageMetadata
from src.models.vision import VisionModelFactory
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class VisionParser(BaseDocumentParser):
    """PDF parser using Vision models for high-quality extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with configuration.
        
        Args:
            config: Configuration dictionary for the parser
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "gemini-1.5-pro-002")
        self.max_pages = self.config.get("max_pages", 100)
        self.vision_model = None
        
        # Load vision configuration from system config
        config_manager = ConfigManager()
        system_config = config_manager.get_config("vision")
        if system_config:
            self.vision_provider = system_config.get("provider", "vertex_ai")
            self.vision_config = system_config.get("config", {})
        else:
            # Default to vertex_ai if no vision config found
            self.vision_provider = "vertex_ai"
            self.vision_config = {}
    
    async def _init_vision_model(self):
        """Initialize the Vision model lazily using factory."""
        if self.vision_model is None:
            try:
                # Create vision model using factory based on configuration
                self.vision_model = VisionModelFactory.create_model(
                    provider=self.vision_provider,
                    model_name=self.model_name,
                    **self.vision_config
                )
                
                # Validate authentication
                is_valid = await self.vision_model.validate_authentication()
                if not is_valid:
                    raise ValueError(f"{self.vision_provider} vision model authentication failed")
                    
                logger.info(f"Successfully initialized {self.vision_provider} vision model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize vision model: {str(e)}")
                raise
    
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a PDF document into a list of Document objects using vision model.
        
        Args:
            file_path: Path to the PDF file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects (one document with all pages combined)
            
        Raises:
            DocumentProcessingError: If PDF parsing fails
        """
        try:
            await self._init_vision_model()
            
            # Open PDF document
            pdf_document = fitz.open(file_path)
            
            # Check if document exceeds max page limit
            page_count = len(pdf_document)
            if page_count > self.max_pages:
                logger.warning(f"PDF has {page_count} pages, exceeding max limit of {self.max_pages}")
            
            # Extract document metadata
            pdf_metadata = pdf_document.metadata
            doc_id = str(uuid.uuid4())
            
            # Create document-level metadata
            document_metadata = DocumentMetadata(
                document_id=doc_id,
                source=file_path,
                document_type=DocumentType.PDF,
                title=pdf_metadata.get("title"),
                author=pdf_metadata.get("author"),
                creation_date=self._parse_pdf_date(pdf_metadata.get("creationDate")),
                page_count=page_count,
                ingestion_time=datetime.now(),
                file_size=self._get_file_size(file_path)
            )
            
            # Process pages and combine into one document
            all_text = []
            
            for page_num, page in enumerate(pdf_document):
                if page_num >= self.max_pages:
                    break
                
                # Get page as image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                # Use vision model to extract text as markdown
                try:
                    markdown_content = await self._extract_text_with_vision(base64_image)
                    all_text.append(f"--- Page {page_num+1} ---\n\n{markdown_content}")
                    logger.info(f"Successfully processed page {page_num+1} of {file_path}")
                except Exception as e:
                    logger.error(f"Vision extraction failed for page {page_num+1}: {str(e)}")
                    # Fallback to regular text extraction
                    fallback_text = page.get_text()
                    all_text.append(f"--- Page {page_num+1} (fallback extraction) ---\n\n{fallback_text}")
            
            # Combine all text
            combined_text = "\n\n".join(all_text)
            
            # Create a single document with all pages
            doc = Document(
                content=combined_text,
                metadata={
                    **document_metadata.dict(),
                    "extraction_method": "vision_parser"
                }
            )
            
            return [doc]
            
        except Exception as e:
            error_msg = f"Vision parser failed for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    async def _extract_text_with_vision(self, base64_image: str) -> str:
        """Extract text from an image using vision model.
        
        Args:
            base64_image: Base64 encoded image
            
        Returns:
            Extracted text in markdown format
        """
        prompt = """
        Please extract all text content from this document page into markdown format.
        Preserve the structure, tables, and formatting as accurately as possible.
        For tables, use proper markdown table syntax.
        For lists, use proper markdown list syntax.
        For headings, use proper markdown heading syntax.
        Ignore any watermarks or page numbers.
        """
        
        # Use the new vision model's parse_text_from_image function
        response = await self.vision_model.parse_text_from_image(
            base64_encoded=base64_image,
            prompt=prompt
        )
        
        return response
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        try:
            return int(os.path.getsize(file_path))
        except Exception:
            return 0
    
    def _parse_pdf_date(self, date_string: Optional[str]) -> Optional[datetime]:
        """Parse PDF date string into datetime object.
        
        Args:
            date_string: PDF date string
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_string:
            return None
        
        try:
            # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
            # where O is + or -
            if date_string.startswith('D:'):
                date_string = date_string[2:]
            
            # Basic parsing for YYYYMMDD format
            year = int(date_string[0:4])
            month = int(date_string[4:6])
            day = int(date_string[6:8])
            
            # Parse time if available
            hour, minute, second = 0, 0, 0
            if len(date_string) >= 14:
                hour = int(date_string[8:10])
                minute = int(date_string[10:12])
                second = int(date_string[12:14])
            
            return datetime(year, month, day, hour, minute, second)
        except Exception:
            logger.warning(f"Failed to parse PDF date: {date_string}")
            return None
