"""Groq Vision Parser for Document Processing.

This parser uses Groq's vision models to extract text and structure from documents,
including PDFs, images, Word documents (.doc/.docx), and Excel files (.xls/.xlsx).

For PDF documents, each page is converted to an image and processed separately.
For Word and Excel documents, they are first converted to PDFs using LibreOffice or Pandoc,
then each page is converted to an image and processed.

Dependencies:
- PyMuPDF (fitz): For PDF page extraction and conversion to images
- LibreOffice: For converting Word and Excel documents to PDFs (optional)
- Pandoc: Alternative for converting Word documents to PDFs (optional)

Note: Either LibreOffice or Pandoc must be installed for Word/Excel processing.
"""

import logging
import base64
import os
import fitz  # PyMuPDF
import tempfile
import subprocess
import shutil
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.models.vision.vision_factory import VisionModelFactory
from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.shared.models.schema import DocumentType, DocumentMetadata, FullDocument, PageMetadata, DocumentPage
from src.rag.core.exceptions.exceptions import ParsingError
from src.rag.core.interfaces.base import Document

logger = logging.getLogger(__name__)


class GroqVisionParser(BaseDocumentParser):
    """Parser that uses Groq vision models for document text extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Groq vision parser.
        
        Args:
            config: Configuration dictionary containing model settings
                - model_name: Optional model name (default: "llama-3.2-11b-vision-preview")
                - prompt_template: Optional prompt template (default: "Extract and structure the text content from this document. Preserve formatting and structure.")
                - max_pages: Maximum number of pages to process in PDF documents (default: 50)
        """
        super().__init__(config)
        self.model_name = config.get("model_name", "llama-3.2-11b-vision-preview")
        self.prompt_template = config.get(
            "prompt_template", 
            "Extract and structure the text content from this document. Preserve formatting and structure."
        )
        self.max_pages = config.get("max_pages", 50)
        self.max_concurrent_pages = config.get("max_concurrent_pages", 5)
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
            
            # Get document type
            doc_type = self.get_document_type(file_path)
            
            # Initialize page count and extracted text
            page_count = 1
            extracted_text = ""
            
            # Handle based on document type
            file_ext = Path(file_path).suffix.lower()
            
            if doc_type == DocumentType.PDF:
                # Process PDF using PyMuPDF
                extracted_text, page_count = await self._parse_pdf_document(file_path, vision_model)
                
            elif file_ext in [".doc", ".docx"]:
                # Process Word documents
                extracted_text, page_count = await self._parse_word_document(file_path, vision_model)
                
            elif file_ext in [".xls", ".xlsx"]:
                # Process Excel documents
                extracted_text, page_count = await self._parse_excel_document(file_path, vision_model)
                
            else:
                # For regular images, use the standard approach
                base64_content = self._convert_to_base64(file_path)
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
                source=metadata.get("source", file_path),  # Add source field as required by DocumentMetadata
                document_type=doc_type.value,
                page_count=page_count,
                **{k: v for k, v in (metadata or {}).items() if k not in ["title", "source", "document_type", "page_count"]}
            )
            
            # Create document_id
            document_id = metadata.get("document_id") if metadata and "document_id" in metadata else str(uuid.uuid4())
            
            # Create pages list for FullDocument
            pages = []
            
            # If this is a multi-page document (PDF, Word, Excel)
            if isinstance(extracted_text, str) and "--- Page " in extracted_text:
                # Split the text by page markers
                page_texts = []
                if doc_type == DocumentType.PDF or file_ext in [".doc", ".docx", ".xls", ".xlsx"]:
                    page_parts = extracted_text.split("--- Page ")
                    # First part is empty or header
                    for i, page_part in enumerate(page_parts):
                        if i == 0 and not page_part.strip():
                            continue
                        # Extract page number and content
                        if i == 0:
                            # Handle case where there's text before the first page marker
                            page_texts.append((1, page_part))
                        else:
                            # Format is "N ---\n\nContent"
                            page_num_str, *content_parts = page_part.split("---", 1)
                            try:
                                page_num = int(page_num_str.strip())
                                page_content = "---".join(content_parts).strip()
                                page_texts.append((page_num, page_content))
                            except ValueError:
                                # If page number parsing fails, just add as is
                                page_texts.append((i, page_part))
                else:
                    # For single page documents
                    page_texts = [(1, extracted_text)]
            else:
                # For single page documents or when there are no page markers
                page_texts = [(1, extracted_text)]
            
            # Combine all page texts into one string for compatibility with Document interface
            all_page_texts = []
            for page_num, page_content in page_texts:
                all_page_texts.append(f"--- Page {page_num} ---\n\n{page_content}")
            
            # Combine all text
            combined_text = "\n\n".join(all_page_texts)
            
            # Create a Document object with dictionary metadata (not Pydantic model)
            # This is compatible with how the service code expects to work with documents
            document = Document(
                content=combined_text,
                metadata={
                    **doc_metadata.dict(),  # Convert Pydantic model to dict
                    "extraction_method": "groq_vision",
                    "document_id": document_id,
                    "page_count": page_count
                }
            )
            
            logger.info(f"Successfully parsed document with {len(combined_text)} characters")
            return [document]
            
        except Exception as e:
            logger.error(f"Groq vision parsing failed for {file_path}: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to parse document with Groq vision: {str(e)}")
    
    async def _parse_word_document(self, file_path: str, vision_model) -> Tuple[str, int]:
        """Process a Word document by first converting it to PDF, then to images for vision processing.
        
        Args:
            file_path: Path to the Word document file
            vision_model: Initialized vision model
            
        Returns:
            Tuple of (extracted text, page count)
        """
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, "converted_document.pdf")
                
                # Convert Word document to PDF using LibreOffice if available
                if shutil.which("soffice"):
                    logger.info(f"Converting Word document to PDF using LibreOffice: {file_path}")
                    cmd = [
                        "soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", temp_dir, file_path
                    ]
                    process = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if process.returncode != 0:
                        logger.error(f"LibreOffice conversion failed: {process.stderr}")
                        raise ParsingError(f"Failed to convert Word document to PDF: {process.stderr}")
                    
                    # Get the output PDF path (LibreOffice maintains original filename with .pdf extension)
                    pdf_filename = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
                    temp_pdf_path = os.path.join(temp_dir, pdf_filename)
                else:
                    # Fallback: Try to use Pandoc if available
                    if shutil.which("pandoc"):
                        logger.info(f"Converting Word document to PDF using Pandoc: {file_path}")
                        cmd = [
                            "pandoc", file_path, "-o", temp_pdf_path
                        ]
                        process = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if process.returncode != 0:
                            logger.error(f"Pandoc conversion failed: {process.stderr}")
                            raise ParsingError(f"Failed to convert Word document to PDF: {process.stderr}")
                    else:
                        logger.error("Neither LibreOffice nor Pandoc found for Word document conversion")
                        raise ParsingError("Word document conversion requires LibreOffice or Pandoc")
                
                # Now process the converted PDF
                if os.path.exists(temp_pdf_path):
                    return await self._parse_pdf_document(temp_pdf_path, vision_model)
                else:
                    raise ParsingError("PDF conversion failed, no output file created")
        
        except Exception as e:
            logger.error(f"Word document processing failed: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to process Word document: {str(e)}")
    
    async def _parse_excel_document(self, file_path: str, vision_model) -> Tuple[str, int]:
        """Process an Excel document by first converting it to PDF, then to images for vision processing.
        
        Args:
            file_path: Path to the Excel file
            vision_model: Initialized vision model
            
        Returns:
            Tuple of (extracted text, page count)
        """
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, "converted_spreadsheet.pdf")
                
                # Convert Excel to PDF using LibreOffice if available
                if shutil.which("soffice"):
                    logger.info(f"Converting Excel document to PDF using LibreOffice: {file_path}")
                    cmd = [
                        "soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", temp_dir, file_path
                    ]
                    process = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if process.returncode != 0:
                        logger.error(f"LibreOffice conversion failed: {process.stderr}")
                        raise ParsingError(f"Failed to convert Excel document to PDF: {process.stderr}")
                    
                    # Get the output PDF path (LibreOffice maintains original filename with .pdf extension)
                    pdf_filename = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
                    temp_pdf_path = os.path.join(temp_dir, pdf_filename)
                else:
                    logger.error("LibreOffice not found for Excel document conversion")
                    raise ParsingError("Excel document conversion requires LibreOffice")
                
                # Now process the converted PDF
                if os.path.exists(temp_pdf_path):
                    return await self._parse_pdf_document(temp_pdf_path, vision_model)
                else:
                    raise ParsingError("PDF conversion failed, no output file created")
        
        except Exception as e:
            logger.error(f"Excel document processing failed: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to process Excel document: {str(e)}")
    
    async def _parse_pdf_document(self, file_path: str, vision_model) -> Tuple[str, int]:
        """Process a PDF document by converting each page to an image and extracting text.
        
        Args:
            file_path: Path to the PDF file
            vision_model: Initialized vision model
            
        Returns:
            Tuple of (extracted text, page count)
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            page_count = len(pdf_document)
            
            # Check if PDF exceeds maximum page count
            if page_count > self.max_pages:
                logger.warning(f"PDF has {page_count} pages, exceeding max limit of {self.max_pages}. Will only process first {self.max_pages} pages.")
                page_count_to_process = self.max_pages
            else:
                page_count_to_process = page_count
            
            # Process pages in parallel
            logger.info(f"Processing {page_count_to_process} pages in parallel with max concurrency: {self.max_concurrent_pages}")
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(self.max_concurrent_pages)
            
            # Process pages in batches to avoid memory issues
            all_text = [None] * page_count_to_process  # Pre-allocate list to maintain order
            
            # Create tasks for all pages
            tasks = []
            for page_num in range(page_count_to_process):
                task = self._process_single_pdf_page(pdf_document, page_num, vision_model, semaphore)
                tasks.append(task)
            
            # Execute all tasks in parallel and collect results
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and maintain page order
                for page_num, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Page {page_num+1} processing failed: {str(result)}")
                        all_text[page_num] = f"--- Page {page_num+1} ---\n\n[Error: Failed to extract text from this page]"
                    else:
                        all_text[page_num] = result
                        
            except Exception as e:
                logger.error(f"Parallel processing failed: {str(e)}")
                # Fallback to sequential processing
                logger.info("Falling back to sequential processing...")
                all_text = await self._process_pdf_pages_sequentially(pdf_document, page_count_to_process, vision_model)
            
            # Close the PDF document
            pdf_document.close()
            
            # Combine text from all pages
            combined_text = "\n\n".join(all_text)
            return combined_text, page_count
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
            raise ParsingError(f"Failed to process PDF: {str(e)}")
    
    async def _process_single_pdf_page(self, pdf_document, page_num: int, vision_model, semaphore: asyncio.Semaphore) -> str:
        """Process a single PDF page with concurrency control.
        
        Args:
            pdf_document: PyMuPDF document object
            page_num: Page number (0-indexed)
            vision_model: Initialized vision model
            semaphore: Semaphore to control concurrency
            
        Returns:
            Formatted text content for the page
        """
        async with semaphore:
            try:
                # Convert page to image in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    base64_image = await loop.run_in_executor(
                        executor, 
                        self._convert_pdf_page_to_base64, 
                        pdf_document, 
                        page_num
                    )
                
                # Use vision model to extract text
                page_text = await vision_model.parse_text_from_image(
                    base64_image=base64_image,
                    prompt=self.prompt_template
                )
                logger.info(f"Successfully extracted text from page {page_num + 1}")
                return f"--- Page {page_num+1} ---\n\n{page_text}"
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
                raise e
    
    def _convert_pdf_page_to_base64(self, pdf_document, page_num: int) -> str:
        """Convert a PDF page to base64 image (synchronous operation for thread pool).
        
        Args:
            pdf_document: PyMuPDF document object
            page_num: Page number (0-indexed)
            
        Returns:
            Base64 encoded image string
        """
        # Get the page
        page = pdf_document[page_num]
        
        # Convert page to pixmap (image)
        pix = page.get_pixmap()
        
        # Convert pixmap to PNG image data
        png_data = pix.tobytes("png")
        
        # Convert PNG data to base64
        return base64.b64encode(png_data).decode('utf-8')
    
    async def _process_pdf_pages_sequentially(self, pdf_document, page_count_to_process: int, vision_model) -> List[str]:
        """Fallback method to process PDF pages sequentially.
        
        Args:
            pdf_document: PyMuPDF document object
            page_count_to_process: Number of pages to process
            vision_model: Initialized vision model
            
        Returns:
            List of formatted text content for each page
        """
        all_text = []
        
        for page_num in range(page_count_to_process):
            try:
                # Get page as base64 image
                base64_image = self._convert_pdf_page_to_base64(pdf_document, page_num)
                
                # Use vision model to extract text
                page_text = await vision_model.parse_text_from_image(
                    base64_image=base64_image,
                    prompt=self.prompt_template
                )
                all_text.append(f"--- Page {page_num+1} ---\n\n{page_text}")
                logger.info(f"Successfully extracted text from page {page_num + 1} (sequential)")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
                all_text.append(f"--- Page {page_num+1} ---\n\n[Error: Failed to extract text from this page]")
        
        return all_text
    
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
        return [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".doc", ".docx", ".xls", ".xlsx"]
    
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
