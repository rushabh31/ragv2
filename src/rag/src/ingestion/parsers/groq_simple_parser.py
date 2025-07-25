import base64
import logging
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import groq
import json
from pathlib import Path
import fitz  # PyMuPDF

from src.rag.src.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.src.core.interfaces.base import Document
from src.rag.src.core.exceptions.exceptions import DocumentProcessingError
from src.rag.src.shared.models.schema import DocumentType, DocumentMetadata, PageMetadata
from src.rag.src.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GroqSimpleParser(BaseDocumentParser):
    """Simplified PDF parser using Groq Vision models for text extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with configuration.
        
        Args:
            config: Configuration dictionary for the parser
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "llama-3.1-70b-vision")
        self.max_pages = self.config.get("max_pages", 100)
        self.groq_client = None
        self.api_key = self.config.get("api_key") or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it in config or GROQ_API_KEY environment variable.")
    
    def _init_groq_client(self):
        """Initialize the Groq client lazily."""
        if self.groq_client is None:
            self.groq_client = groq.Client(api_key=self.api_key)
    
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a PDF document into a list of Document objects using Groq vision model.
        
        Args:
            file_path: Path to the PDF file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects (one per page)
            
        Raises:
            DocumentProcessingError: If PDF parsing fails
        """
        try:
            self._init_groq_client()
            
            # Use PyMuPDF to convert PDF to images
            pdf_document = fitz.open(file_path)
            
            # Create document metadata
            file_name = Path(file_path).name
            doc_id = str(uuid.uuid4())
            
            # Create document-level metadata
            document_metadata = DocumentMetadata(
                document_id=doc_id,
                source=file_path,
                document_type=DocumentType.PDF,
                title=file_name,
                creation_date=None,
                page_count=len(pdf_document),
                ingestion_time=datetime.now(),
                file_size=self._get_file_size(file_path)
            )
            
            # Process each page
            all_text = []
            for page_num, page in enumerate(pdf_document):
                if page_num >= self.max_pages:
                    break
                    
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                # Extract text from image
                try:
                    page_text = await self._extract_text_with_groq_vision(base64_image)
                    all_text.append(f"--- Page {page_num+1} ---\n\n{page_text}")
                    logger.info(f"Successfully processed page {page_num+1} of {file_path}")
                except Exception as e:
                    logger.error(f"Failed to process page {page_num+1}: {str(e)}")
                    # Continue with next page
            
            # Combine all text
            combined_text = "\n\n".join(all_text)
            
            # Create document object
            doc = Document(
                content=combined_text,
                metadata={
                    **document_metadata.dict(),
                    "extraction_method": "groq_simple_parser"
                }
            )
            
            return [doc]
            
        except Exception as e:
            error_msg = f"Groq Simple parser failed for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    async def _extract_text_with_groq_vision(self, base64_image: str) -> str:
        """Extract text from an image using Groq vision model.
        
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
        
        try:
            # Prepare the message with image content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call the Groq API
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "text"},
            )
            
            # Extract and return the markdown content
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq Vision API call failed: {str(e)}")
            raise DocumentProcessingError(f"Groq Vision API call failed: {str(e)}")
    
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
