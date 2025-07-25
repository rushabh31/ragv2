from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid
from datetime import datetime


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


class JobStatus(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    document_type: Optional[DocumentType] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    extraction_options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    document_id: Optional[str] = None
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: JobStatus
    document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class DocumentMetadataResponse(BaseModel):
    """Response model for document metadata."""
    document_id: str
    document_type: DocumentType
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    ingestion_time: datetime
    ingested_by: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[DocumentMetadataResponse]
    total: int
    page: int
    page_size: int


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration update."""
    config_path: str
    value: Any


class ConfigUpdateResponse(BaseModel):
    """Response model for configuration update."""
    success: bool
    message: str
