import base64
import logging
import os
import uuid
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from datetime import datetime
import json
import aiohttp
import requests

from src.rag.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.core.interfaces.base import Document
from src.rag.core.exceptions.exceptions import DocumentProcessingError
from src.rag.shared.models.schema import DocumentType, DocumentMetadata, PageMetadata
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class OpenAIVisionParser(BaseDocumentParser):
    """PDF parser using OpenAI-compatible Vision models for high-quality extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with configuration.
        
        Args:
            config: Configuration dictionary for the parser
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "gpt-4o")
        self.max_pages = self.config.get("max_pages", 100)
        self.api_key = None
        self.api_base = self.config.get("api_base", "https://api.openai.com/v1")
        self.api_version = self.config.get("api_version", None)
        self.api_type = self.config.get("api_type", "openai")
        
        # Load configuration for OpenAI
        config_manager = ConfigManager()
        self.vertex_config = config_manager.get_config("vertex")
        if not self.vertex_config:
            raise ValueError("Vertex configuration not found in config. Required for token authentication.")
    
    async def _parse_file(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a PDF document into a list of Document objects using OpenAI-compatible vision model.
        
        Args:
            file_path: Path to the PDF file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects (one document with all pages combined)
            
        Raises:
            DocumentProcessingError: If PDF parsing fails
        """
        try:
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
                    markdown_content = await self._extract_text_with_vision_api(base64_image)
                    all_text.append(f"--- Page {page_num+1} ---\n\n{markdown_content}")
                    logger.info(f"Successfully processed page {page_num+1} of {file_path}")
                except Exception as e:
                    logger.error(f"Vision API extraction failed for page {page_num+1}: {str(e)}")
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
                    "extraction_method": "openai_vision_parser"
                }
            )
            
            return [doc]
            
        except Exception as e:
            error_msg = f"OpenAI Vision parser failed for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
    
    def get_coin_token(self):
        """Get authentication token for API access."""
        url = self.vertex_config.get("COIN_CONSUMER_ENDPOINT_URL") # URL from .env
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "scope": self.vertex_config.get("COIN_CONSUMER_SCOPE"), # Scope from .env
            "client_id": self.vertex_config.get("COIN_CONSUMER_CLIENT_ID"), # Client ID from .env
            "client_secret": self.vertex_config.get("COIN_CONSUMER_CLIENT_SECRET"), # Client Secret from .env
        }
        
        print("Requesting API token for OpenAI") # Debug step 
        
        response = requests.post(url, headers=headers, data=body, verify=False, timeout=10)
        
        if response.status_code == 200:
            print("Token received successfully for OpenAI") # Debug step 
            return response.json().get('access_token', None)
        else:
            print(f"Failed to get token! status code: {response.status_code}, Response: {response.text}")
            return None
    
    async def _extract_text_with_vision_api(self, base64_image: str) -> str:
        """Extract text from an image using OpenAI-compatible vision model.
        
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
            # Get token for authentication
            token = self.get_coin_token()
            if not token:
                raise ValueError("Failed to retrieve token for OpenAI client.")
            
            # Prepare the payload for API request
            payload = {
                "model": self.model_name,
                "messages": [
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
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
                "response_format": {"type": "text"}
            }
            
            # Determine the API endpoint
            endpoint = f"{self.api_base}/chat/completions"
            if self.api_version and self.api_type in ['azure', 'azure_ad']:
                # Format for Azure OpenAI
                endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            # Make the async API request
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise DocumentProcessingError(f"API returned error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    
                    # Extract content from response
                    try:
                        content = result["choices"][0]["message"]["content"]
                        return content
                    except (KeyError, IndexError) as e:
                        raise DocumentProcessingError(f"Invalid API response format: {str(e)}")
            
        except Exception as e:
            logger.error(f"Vision API call failed: {str(e)}")
            raise DocumentProcessingError(f"Vision API call failed: {str(e)}")
    
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
