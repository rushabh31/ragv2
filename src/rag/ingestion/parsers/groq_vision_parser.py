"""
Groq Vision Parser for Document Processing.

This parser uses Groq's vision models to extract text and structure from documents,
particularly PDFs and images, using the universal authentication system.
"""

import logging
import base64
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.models.vision.vision_factory import VisionModelFactory
from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.shared.models.schema import FullDocument, DocumentType, DocumentMetadata
from src.rag.core.exceptions.exceptions import ParsingError
from src.rag.core.interfaces.base import Document

logger = logging.getLogger(__name__)


class GroqVisionParser(BaseDocumentParser):
    """Parser that uses Groq vision models for document text extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Groq vision parser.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = config.get("model_name", "llama-3.2-11b-vision-preview")
        self.prompt_template = config.get(
            "prompt_template", 
            "Extract and structure the text content from this document. Preserve formatting and structure."
        )
        self.max_pages = config.get("max_pages", 50)
        self._vision_model = None
        
        logger.info(f"Initialized GroqVisionParser with model: {self.model_name}")
    
    async def _get_vision_model(self):
        """Get or create the vision model instance."""
        if self._vision_model is None:
            try:
                self._vision_model = VisionModelFactory.create_model(
                    provider="groq",
                    model_name=self.model_name
                )
                logger.info(f"Created Groq vision model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to create Groq vision model: {str(e)}")
                raise ParsingError(f"Failed to initialize Groq vision model: {str(e)}")
        return self._vision_model
    
    def _convert_to_base64(self, file_path: str) -> str:
        """Convert file to base64 string.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Base64 encoded string
        """
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert file to base64: {str(e)}")
            raise ParsingError(f"Failed to read file {file_path}: {str(e)}")
    
    async def parse(self, file_path: str, metadata: Dict[str, Any] = None) -> List[FullDocument]:
        """Parse document using Groq vision model.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata dictionary
            
        Returns:
            List of FullDocument objects
        """
        try:
            logger.info(f"Starting Groq vision parsing for: {file_path}")
            
            # Get vision model
            vision_model = await self._get_vision_model()
            
            # Convert file to base64
            base64_content = self._convert_to_base64(file_path)
            
            # Extract text using vision model
            extracted_text = await vision_model.parse_text_from_image(
                base64_image=base64_content,
                prompt=self.prompt_template
            )
            
            if not extracted_text or not extracted_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                extracted_text = "No text content could be extracted from this document."
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                title=metadata.get("title", os.path.basename(file_path)),
                filename=os.path.basename(file_path),
                document_type=self.get_document_type(file_path).value,
                created_at=datetime.now(),
                page_count=1,  # Vision models typically process the whole document as one
                **metadata if metadata else {}
            )
            
            # Create full document
            document = FullDocument(
                content=extracted_text,
                metadata=doc_metadata.dict(),
                document_id=metadata.get("document_id") if metadata else None
            )
            
            logger.info(f"Successfully parsed document with {len(extracted_text)} characters")
            return [document]
            
        except Exception as e:
            logger.error(f"Groq vision parsing failed for {file_path}: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to parse document with Groq vision: {str(e)}")
    
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Implementation of abstract method from BaseDocumentParser.
        
        Args:
            file_path: Path to the document file
            config: Configuration parameters for the parser
            
        Returns:
            List of Document objects (not FullDocument)
        """
        try:
            # Use the existing parse method to get FullDocument objects
            metadata = config.get('metadata', {})
            full_documents = await self.parse(file_path, metadata)
            
            # Convert FullDocument objects to Document objects
            documents = []
            
            for full_doc in full_documents:
                document = Document(
                    content=full_doc.content,
                    metadata=full_doc.metadata,
                    document_id=full_doc.document_id
                )
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"_parse_file failed for {file_path}: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to parse document: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about this parser.
        
        Returns:
            Dictionary containing parser information
        """
        return {
            "name": "groq_vision_parser",
            "provider": "groq",
            "model": self.model_name,
            "supported_formats": self.get_supported_formats(),
            "max_pages": self.max_pages,
            "description": "Groq vision-based document parser for text extraction"
        }
