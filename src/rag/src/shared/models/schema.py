from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class DocumentType(str, Enum):
    """Enum for document types."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    HTML = "html"
    MD = "md"
    IMAGE = "image"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    document_type: DocumentType
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    file_size: Optional[int] = None  # Size in bytes
    ingestion_time: datetime = Field(default_factory=datetime.now)
    ingested_by: Optional[str] = None  # soeid of the user who uploaded the document
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class PageMetadata(BaseModel):
    """Metadata for a page within a document."""
    page_number: int
    page_image_base64: Optional[str] = None  # Base64 encoded page image
    extraction_method: str
    page_size: Optional[Dict[str, int]] = None  # Width and height in pixels or points
    custom_page_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentPage(BaseModel):
    """A page within a document with content and metadata."""
    content: str
    metadata: PageMetadata
    document_id: str  # Reference to parent document


class FullDocument(BaseModel):
    """Complete document representation with metadata and pages."""
    metadata: DocumentMetadata
    pages: List[DocumentPage]


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_numbers: List[int]  # A chunk may span multiple pages
    chunk_index: int  # Position of the chunk within the document
    extraction_method: str
    chunk_type: str  # "text", "table", "image_caption", etc.
    custom_chunk_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of text from a document with metadata."""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class QueryResult(BaseModel):
    """Result for a query containing chunks and similarity scores."""
    query: str
    results: List[Dict[str, Any]]  # List of chunks with scores
    total_results: int
    retrieval_time: float  # Time in milliseconds
    reranked: bool = False


class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    """Chat session with user identifier and messages."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str  # soeid
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)


class SearchRequest(BaseModel):
    """Request for document search."""
    query: str
    top_k: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = True


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    query: str
    session_id: Optional[str] = None
    user_id: str  # soeid
    context: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    job_id: str
    status: str = "processing"
    document_id: Optional[str] = None
    message: str


class JobStatus(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(BaseModel):
    """Status of a processing job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None  # soeid
    error_message: Optional[str] = None
    progress: Optional[float] = None  # Progress as a percentage
