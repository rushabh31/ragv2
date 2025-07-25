import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header, BackgroundTasks, Query, status, Form
from fastapi.responses import JSONResponse
import json

from src.rag.ingestion.api.models import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    JobStatusResponse,
    DocumentListResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse
)
from src.rag.ingestion.api.service import IngestionService
from src.rag.core.exceptions.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Service dependency
def get_ingestion_service():
    """Dependency to get ingestion service instance."""
    return IngestionService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),  # Changed to str with Form to accept JSON string
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Upload a document for processing."""
    try:
        # Save the uploaded file to disk immediately
        import os, shutil
        uploads_dir = os.path.join("data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        _, ext = os.path.splitext(file.filename)
        import uuid
        document_id = str(uuid.uuid4())
        file_path = os.path.join(uploads_dir, f"{document_id}{ext}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Parse options if provided
        metadata = {}
        if options:
            try:
                # Try to parse as JSON string
                options_dict = json.loads(options)
                if isinstance(options_dict, dict):
                    metadata = options_dict.get("metadata", {})
                    logger.info(f"Successfully parsed options: {options_dict}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse options as JSON: {options}")

        # Create job and start processing, passing the file path instead of UploadFile
        job = await service.upload_document(file_path, soeid, metadata)

        return DocumentUploadResponse(
            job_id=job.job_id,
            status=job.status,
            document_id=job.document_id,
            message="Document upload accepted for processing"
        )
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def check_job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service)
):
    """Check the status of a processing job.
    
    Args:
        job_id: Job identifier
        service: Ingestion service instance
    
    Returns:
        Job status details
    """
    job = await service.get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        document_id=job.document_id,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        message="Processing in progress" if job.status == "processing" else None,
        error=job.error_message
    )

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """List ingested documents.
    
    Args:
        page: Page number
        page_size: Number of documents per page
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        List of document metadata with pagination info
    """
    result = await service.list_documents(user_id=None, page=page, page_size=page_size)
    
    return DocumentListResponse(
        documents=result["documents"],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"]
    )

@router.get("/documents/user/{soeid}", response_model=DocumentListResponse)
async def get_user_documents(
    soeid: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    auth_soeid: str = Header(..., alias="soeid"),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Get documents for a specific user by their SOEID.
    
    Args:
        soeid: User ID to retrieve documents for
        page: Page number
        page_size: Number of documents per page
        auth_soeid: Authenticated user's ID (from header)
        service: Ingestion service instance
    
    Returns:
        List of document metadata with pagination info for the specified user
    """
    # Optional: Add authorization check if needed
    # if auth_soeid != soeid and not is_admin(auth_soeid):
    #     raise HTTPException(status_code=403, detail="Not authorized to access other users' documents")
    
    result = await service.list_documents(user_id=soeid, page=page, page_size=page_size)
    
    return DocumentListResponse(
        documents=result["documents"],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"]
    )
    
@router.get("/test-documents/{soeid}", response_model=Dict[str, Any])
async def test_document_metadata(
    soeid: str,
    auth_soeid: str = Header(..., alias="soeid"),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Test endpoint to view detailed metadata for debugging purposes.
    
    Args:
        soeid: User ID to retrieve documents for
        auth_soeid: Authenticated user ID from header
        service: Ingestion service instance
        
    Returns:
        Raw metadata information for debugging
    """
    try:
        if not service._vector_store:
            await service._init_components()
            
        # For debugging, dump all chunk metadata
        metadata = {
            "chunks": [],
            "vector_store_initialized": service._vector_store._initialized,
            "total_chunks": len(service._vector_store.chunks) if hasattr(service._vector_store, 'chunks') else 0,
            "soeid_requested": soeid,
            "auth_soeid": auth_soeid
        }
        
        if hasattr(service._vector_store, 'chunks'):
            for i, chunk in enumerate(service._vector_store.chunks):
                if i < 50:  # Limit to first 50 chunks to prevent huge response
                    metadata["chunks"].append({
                        "chunk_idx": i,
                        "content_preview": chunk.content[:100] + "...",
                        "metadata": chunk.metadata,
                        "has_embedding": chunk.embedding is not None,
                        "embedding_shape": len(chunk.embedding) if chunk.embedding is not None else None
                    })
        
        return metadata
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}", exc_info=True)
        return {"error": str(e), "error_type": type(e).__name__}

@router.post("/configure", response_model=ConfigUpdateResponse)
async def update_configuration(
    config_update: ConfigUpdateRequest,
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Update ingestion configuration.
    
    Args:
        config_update: Configuration update request
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        Result of configuration update
    """
    try:
        success = await service.update_config(config_update.config_path, config_update.value)
        
        if success:
            return ConfigUpdateResponse(
                success=True,
                message=f"Configuration {config_update.config_path} updated successfully"
            )
        else:
            return ConfigUpdateResponse(
                success=False,
                message=f"Failed to update configuration {config_update.config_path}"
            )
    except Exception as e:
        logger.error(f"Configuration update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration update failed: {str(e)}"
        )
