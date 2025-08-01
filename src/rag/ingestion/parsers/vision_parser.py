import base64
import logging
import os
import uuid
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.core.interfaces.base import Document
from src.utils.env_manager import env
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
        
        # Load vision configuration from system config first
        config_manager = ConfigManager()
        system_config = config_manager.get_section("vision")
        if system_config:
            self.vision_provider = system_config.get("provider", "vertex_ai")
            self.vision_config = system_config.get("config", {})
            # Read model from system vision config
            self.model_name = self.vision_config.get("model", "gemini-1.5-pro-002")
        else:
            # Default to vertex_ai if no vision config found
            self.vision_provider = "vertex_ai"
            self.vision_config = {}
            self.model_name = "gemini-2.5-flash"  # Default model
        
        # Read parser-specific settings from self.config (passed from ingestion.parser.config)
        self.max_pages = self.config.get("max_pages", 100)
        self.max_concurrent_pages = self.config.get("max_concurrent_pages", 5)
        
        # Retry configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2.0)  # seconds
        self.retry_backoff_multiplier = self.config.get("retry_backoff_multiplier", 1.5)
        
        # SSL certificate configuration
        self.ssl_cert_path = env.get_string("SSL_CERT_FILE", "config/certs.pem")
        if self.ssl_cert_path:
            env.set("SSL_CERT_FILE", self.ssl_cert_path)
            logger.info(f"Using SSL certificate from: {self.ssl_cert_path}")
        else:
            logger.warning(f"SSL certificate not found at: {self.ssl_cert_path}")
        
        self.vision_model = None
    
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
            
            # Extract document metadata and filename
            pdf_metadata = pdf_document.metadata
            doc_id = str(uuid.uuid4())
            
            # Extract filename from file path
            filename = Path(file_path).name
            logger.info(f"Processing file: {filename}")
            
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
            
            # Process pages in parallel
            pages_to_process = min(page_count, self.max_pages)
            logger.info(f"Processing {pages_to_process} pages in parallel with max concurrency: {self.max_concurrent_pages}")
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(self.max_concurrent_pages)
            
            # Process pages in batches to avoid memory issues
            all_text = [None] * pages_to_process  # Pre-allocate list to maintain order
            
            # Create tasks for all pages
            tasks = []
            for page_num in range(pages_to_process):
                task = self._process_single_page(pdf_document, page_num, file_path, semaphore)
                tasks.append(task)
            
            # Execute all tasks in parallel and collect results
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and maintain page order
                page_data_list = []  # Store page data with images
                for page_num, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Page {page_num+1} processing failed: {str(result)}")
                        # Fallback to regular text extraction
                        try:
                            page = pdf_document[page_num]
                            fallback_text = page.get_text()
                            fallback_page_data = {
                                'text': f"--- Page {page_num+1} (fallback extraction) ---\n\n{fallback_text}",
                                'base64_image': None,
                                'image_width': None,
                                'image_height': None,
                                'page_number': page_num + 1
                            }
                            all_text[page_num] = fallback_page_data['text']
                            page_data_list.append(fallback_page_data)
                        except Exception as fallback_error:
                            logger.error(f"Fallback extraction also failed for page {page_num+1}: {str(fallback_error)}")
                            error_page_data = {
                                'text': f"--- Page {page_num+1} (extraction failed) ---\n\n[Content could not be extracted]",
                                'base64_image': None,
                                'image_width': None,
                                'image_height': None,
                                'page_number': page_num + 1
                            }
                            all_text[page_num] = error_page_data['text']
                            page_data_list.append(error_page_data)
                    else:
                        # Result is now a page data dictionary
                        if isinstance(result, dict) and 'text' in result:
                            all_text[page_num] = result['text']
                            page_data_list.append(result)
                        else:
                            # Handle legacy string format
                            all_text[page_num] = result
                            legacy_page_data = {
                                'text': result,
                                'base64_image': None,
                                'image_width': None,
                                'image_height': None,
                                'page_number': page_num + 1
                            }
                            page_data_list.append(legacy_page_data)
                        
            except Exception as e:
                logger.error(f"Parallel processing failed: {str(e)}")
                # Fallback to sequential processing
                logger.info("Falling back to sequential processing...")
                all_text = await self._process_pages_sequentially(pdf_document, pages_to_process, file_path)
            
            # Combine all text
            combined_text = "\n\n".join(all_text)
            
            # Create documents with page-level image data
            documents = []
            
            # Create a document for each page with its image data
            for page_data in page_data_list:
                page_metadata = {
                    **document_metadata.dict(),
                    "extraction_method": "vision_parser",
                    "file_name": filename,
                    "source_file": filename,
                    "page_number": page_data['page_number']
                }
                
                # Add image data to metadata if available
                if page_data['base64_image']:
                    page_metadata['base64_image'] = page_data['base64_image']
                    page_metadata['image_width'] = page_data['image_width']
                    page_metadata['image_height'] = page_data['image_height']
                
                page_doc = Document(
                    content=page_data['text'],
                    metadata=page_metadata
                )
                documents.append(page_doc)
            
            # Also create a combined document for backward compatibility
            combined_doc = Document(
                content=combined_text,
                metadata={
                    **document_metadata.dict(),
                    "extraction_method": "vision_parser",
                    "file_name": filename,
                    "source_file": filename,
                    "document_type": "combined_pages"
                }
            )
            documents.append(combined_doc)
            
            return documents
            
        except Exception as e:
            error_msg = f"Vision parser failed for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    async def _extract_text_with_vision(self, base64_image: str, page_num: int = 0) -> str:
        """Extract text from an image using vision model with retry logic.
        
        Args:
            base64_image: Base64 encoded image
            page_num: Page number for logging purposes
            
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
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (self.retry_backoff_multiplier ** (attempt - 1))
                    logger.info(f"Retrying vision extraction for page {page_num + 1}, attempt {attempt + 1}/{self.max_retries + 1} after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                # Use the new vision model's parse_text_from_image function with timeout
                response = await asyncio.wait_for(
                    self.vision_model.parse_text_from_image(
                        base64_encoded=base64_image,
                        prompt=prompt,
                        timeout=30  # 30 second timeout per page
                    ),
                    timeout=45  # Additional 45 second timeout at parser level
                )
                
                if attempt > 0:
                    logger.info(f"Vision extraction succeeded for page {page_num + 1} on attempt {attempt + 1}")
                
                return response
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Vision model call timed out for page {page_num + 1}, attempt {attempt + 1}/{self.max_retries + 1}")
                if attempt == self.max_retries:
                    logger.error(f"Vision extraction failed for page {page_num + 1} after {self.max_retries + 1} attempts due to timeout")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Vision model call failed for page {page_num + 1}, attempt {attempt + 1}/{self.max_retries + 1}: {str(e)}")
                if attempt == self.max_retries:
                    logger.error(f"Vision extraction failed for page {page_num + 1} after {self.max_retries + 1} attempts: {str(e)}")
        
        # If all retries failed, raise the last exception
        raise Exception(f"Vision extraction failed after {self.max_retries + 1} attempts: {str(last_exception)}")
    
    async def _process_single_page(self, pdf_document, page_num: int, file_path: str, semaphore: asyncio.Semaphore) -> str:
        """Process a single page with concurrency control and retry logic.
        
        Args:
            pdf_document: PyMuPDF document object
            page_num: Page number (0-indexed)
            file_path: Path to the PDF file (for logging)
            semaphore: Semaphore to control concurrency
            
        Returns:
            Formatted text content for the page
        """
        async with semaphore:
            try:
                # Convert page to image in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    base64_image, image_width, image_height = await loop.run_in_executor(
                        executor, 
                        self._convert_page_to_base64, 
                        pdf_document, 
                        page_num
                    )
                
                # Use vision model to extract text as markdown with retry logic
                markdown_content = await self._extract_text_with_vision(base64_image, page_num)
                logger.info(f"Successfully processed page {page_num+1} of {file_path}")
                
                # Return both text content and image data
                page_data = {
                    'text': f"--- Page {page_num+1} ---\n\n{markdown_content}",
                    'base64_image': base64_image,
                    'image_width': image_width,
                    'image_height': image_height,
                    'page_number': page_num + 1
                }
                return page_data
                
            except Exception as e:
                logger.error(f"Vision extraction failed for page {page_num+1} after all retries: {str(e)}")
                # Fallback to simple text extraction
                try:
                    page = pdf_document[page_num]
                    fallback_text = page.get_text()
                    logger.info(f"Using fallback text extraction for page {page_num+1}")
                    
                    # Return fallback page data with image if available
                    fallback_page_data = {
                        'text': f"--- Page {page_num+1} (fallback extraction) ---\n\n{fallback_text}",
                        'base64_image': base64_image if 'base64_image' in locals() else None,
                        'image_width': image_width if 'image_width' in locals() else None,
                        'image_height': image_height if 'image_height' in locals() else None,
                        'page_number': page_num + 1
                    }
                    return fallback_page_data
                except Exception as fallback_error:
                    logger.error(f"Fallback extraction also failed for page {page_num+1}: {str(fallback_error)}")
                    
                    # Return error page data
                    error_page_data = {
                        'text': f"--- Page {page_num+1} (extraction failed) ---\n\n[Content could not be extracted]",
                        'base64_image': None,
                        'image_width': None,
                        'image_height': None,
                        'page_number': page_num + 1
                    }
                    return error_page_data
    
    def _convert_page_to_base64(self, pdf_document, page_num: int) -> tuple:
        """Convert a PDF page to base64 image with dimensions (synchronous operation for thread pool).
        
        Args:
            pdf_document: PyMuPDF document object
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (base64_encoded_image, width, height)
        """
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        base64_image = base64.b64encode(img_data).decode('utf-8')
        return base64_image, pix.width, pix.height
    
    async def _process_pages_sequentially(self, pdf_document, pages_to_process: int, file_path: str) -> List[str]:
        """Fallback method to process pages sequentially with retry logic.
        
        Args:
            pdf_document: PyMuPDF document object
            pages_to_process: Number of pages to process
            file_path: Path to the PDF file (for logging)
            
        Returns:
            List of formatted text content for each page
        """
        all_text = []
        
        for page_num in range(pages_to_process):
            try:
                # Get page as image
                base64_image = self._convert_page_to_base64(pdf_document, page_num)
                
                # Use vision model to extract text as markdown with retry logic
                markdown_content = await self._extract_text_with_vision(base64_image, page_num)
                all_text.append(f"--- Page {page_num+1} ---\n\n{markdown_content}")
                logger.info(f"Successfully processed page {page_num+1} of {file_path} (sequential)")
                
            except Exception as e:
                logger.error(f"Vision extraction failed for page {page_num+1} after all retries: {str(e)}")
                # Fallback to regular text extraction
                try:
                    page = pdf_document[page_num]
                    fallback_text = page.get_text()
                    all_text.append(f"--- Page {page_num+1} (fallback extraction) ---\n\n{fallback_text}")
                    logger.info(f"Using fallback text extraction for page {page_num+1} (sequential)")
                except Exception as fallback_error:
                    logger.error(f"Fallback extraction also failed for page {page_num+1}: {str(fallback_error)}")
                    all_text.append(f"--- Page {page_num+1} (extraction failed) ---\n\n[Content could not be extracted]")
        
        return all_text
    
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
