import logging
import os
import shutil
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header, BackgroundTasks, Query, status, Form
from fastapi.responses import JSONResponse
import json

from examples.rag.ingestion.api.models import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    JobStatusResponse,
    DocumentListResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    SessionInfoResponse,
    SessionListResponse,
    BulkUploadRequest,
    BulkUploadResponse,
    BatchStatusResponse,
    FileUploadResult
)
from examples.rag.ingestion.api.service import IngestionService
from src.rag.core.exceptions.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


def _parse_options_robust(options: str) -> Dict[str, Any]:
    """
    Robustly parse options parameter with multiple fallback strategies.
    
    Handles:
    1. Valid JSON strings: '{"metadata": {"key": "value"}}'
    2. Simple key-value strings: 'key=value,key2=value2'
    3. Plain strings: treats as a single metadata value
    4. Empty/None values: returns empty dict
    
    Args:
        options: The options string to parse
        
    Returns:
        Dict containing parsed metadata
    """
    if not options or not options.strip():
        return {}
    
    options = options.strip()
    
    # Strategy 1: Try to parse as JSON
    try:
        parsed = json.loads(options)
        if isinstance(parsed, dict):
            # If it's a dict, extract metadata or use the whole dict
            metadata = parsed.get("metadata", parsed)
            logger.info(f"Successfully parsed options as JSON: {parsed}")
            return metadata if isinstance(metadata, dict) else {"value": metadata}
        elif isinstance(parsed, (str, int, float, bool)):
            # If it's a primitive value, wrap it in metadata
            logger.info(f"Parsed JSON primitive value: {parsed}")
            return {"value": parsed}
        else:
            logger.warning(f"Unexpected JSON type {type(parsed)}: {parsed}")
            return {"raw_value": str(parsed)}
    except json.JSONDecodeError:
        logger.debug(f"Options not valid JSON, trying alternative parsing: {options}")
    
    # Strategy 2: Try to parse as key=value pairs (e.g., "key1=value1,key2=value2")
    if '=' in options and (',' in options or ';' in options):
        try:
            metadata = {}
            # Support both comma and semicolon separators
            separator = ',' if ',' in options else ';'
            pairs = options.split(separator)
            
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse value as JSON if it looks like JSON
                    if value.startswith(('{', '[', '"')) or value in ('true', 'false', 'null'):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string
                    # Try to parse as number
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                    
                    metadata[key] = value
            
            if metadata:
                logger.info(f"Successfully parsed options as key-value pairs: {metadata}")
                return metadata
        except Exception as e:
            logger.debug(f"Failed to parse as key-value pairs: {e}")
    
    # Strategy 3: Try to parse as single key=value pair
    if '=' in options and options.count('=') == 1:
        try:
            key, value = options.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to parse value as JSON
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Keep as string
            
            metadata = {key: value}
            logger.info(f"Successfully parsed options as single key-value: {metadata}")
            return metadata
        except Exception as e:
            logger.debug(f"Failed to parse as single key-value: {e}")
    
    # Strategy 4: Treat as plain string metadata
    logger.info(f"Treating options as plain string metadata: {options}")
    return {"description": options}


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
        
        # Preserve original filename information
        original_filename = file.filename or "unknown_file"
        _, ext = os.path.splitext(original_filename)
        import uuid
        document_id = str(uuid.uuid4())
        
        # Save with UUID but preserve original filename in metadata
        file_path = os.path.join(uploads_dir, f"{document_id}{ext}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Uploaded file saved: original='{original_filename}' -> temp='{file_path}'")

        # Parse options if provided - robust parsing with multiple fallback strategies
        metadata = {}
        if options:
            metadata = _parse_options_robust(options)
        
        # Add original filename to metadata to preserve it through the pipeline
        metadata["original_filename"] = original_filename
        metadata["upload_filename"] = original_filename
        logger.info(f"Added original filename to metadata: {original_filename}")

        # Create job and start processing, passing the file path and enhanced metadata
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

@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Get information about a specific ingestion session.
    
    Args:
        session_id: Session identifier
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        Session information including chunks, document details, and processing stats
    """
    try:
        session_info = await service.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return SessionInfoResponse(**session_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session information: {str(e)}"
        )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: Optional[str] = Query(None, description="Filter sessions by user ID"),
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """List ingestion sessions with pagination.
    
    Args:
        page: Page number
        page_size: Number of sessions per page
        user_id: Optional user ID to filter sessions
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        List of session information with pagination
    """
    try:
        # If no user_id filter provided, use the authenticated user's SOEID
        filter_user_id = user_id if user_id else soeid
        
        result = await service.list_sessions(user_id=filter_user_id, page=page, page_size=page_size)
        
        # Convert to response models
        sessions = [SessionInfoResponse(**session) for session in result["sessions"]]
        
        return SessionListResponse(
            sessions=sessions,
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"]
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )


@router.get("/sessions/all", response_model=SessionListResponse)
async def list_all_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """List all ingestion sessions (admin endpoint).
    
    Args:
        page: Page number
        page_size: Number of sessions per page
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        List of all session information with pagination
    """
    try:
        # Note: In production, you might want to add admin authorization here
        result = await service.list_sessions(user_id=None, page=page, page_size=page_size)
        
        # Convert to response models
        sessions = [SessionInfoResponse(**session) for session in result["sessions"]]
        
        return SessionListResponse(
            sessions=sessions,
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"]
        )
        
    except Exception as e:
        logger.error(f"Failed to list all sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve all sessions: {str(e)}"
        )


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


@router.post("/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    global_metadata: Optional[str] = Form(None, description="JSON string of metadata to apply to all files"),
    process_in_parallel: bool = Form(True, description="Whether to process files in parallel"),
    max_parallel_jobs: int = Form(3, ge=1, le=10, description="Maximum number of parallel processing jobs"),
    soeid: str = Header(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    """Upload multiple documents for processing.
    
    Args:
        background_tasks: FastAPI background tasks
        files: List of uploaded files
        global_metadata: JSON string of metadata to apply to all files
        process_in_parallel: Whether to process files in parallel
        max_parallel_jobs: Maximum number of parallel jobs
        soeid: User identifier (header)
        service: Ingestion service instance
    
    Returns:
        Bulk upload response with batch information
    """
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        if len(files) > 50:  # Reasonable limit for bulk upload
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many files. Maximum 50 files per batch."
            )
        
        # Parse global metadata if provided
        metadata = {}
        if global_metadata:
            try:
                metadata = json.loads(global_metadata)
                if not isinstance(metadata, dict):
                    metadata = {"description": global_metadata}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse global metadata JSON: {global_metadata}")
                metadata = {"description": global_metadata}
        
        # Save all uploaded files to disk
        uploads_dir = os.path.join("data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        saved_files = []
        file_results = []
        
        for file in files:
            try:
                # Preserve original filename information
                original_filename = file.filename or "unknown_file"
                _, ext = os.path.splitext(original_filename)
                document_id = str(uuid.uuid4())
                
                # Save with UUID but preserve original filename in metadata
                file_path = os.path.join(uploads_dir, f"{document_id}{ext}")
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                
                saved_files.append(file_path)
                logger.info(f"Saved file: {original_filename} -> {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save file {file.filename}: {str(e)}")
                file_results.append(FileUploadResult(
                    filename=file.filename or "unknown_file",
                    status="failed",
                    message="Failed to save file",
                    error=str(e)
                ))
        
        # Process saved files in bulk
        if saved_files:
            result = await service.bulk_upload_documents(
                files=saved_files,
                user_id=soeid,
                global_metadata=metadata,
                process_in_parallel=process_in_parallel,
                max_parallel_jobs=max_parallel_jobs
            )
            
            return BulkUploadResponse(
                batch_id=result["batch_id"],
                total_files=result["total_files"],
                successful_uploads=result["successful_uploads"],
                failed_uploads=result["failed_uploads"],
                results=[FileUploadResult(**r) for r in result["results"]] + file_results,
                message=result["message"]
            )
        else:
            # All files failed to save
            return BulkUploadResponse(
                total_files=len(files),
                successful_uploads=0,
                failed_uploads=len(files),
                results=file_results,
                message="All files failed to save"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk upload failed: {str(e)}"
        )


@router.get("/batch-status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    service: IngestionService = Depends(get_ingestion_service)
):
    """Get the status of a bulk processing batch.
    
    Args:
        batch_id: Batch identifier
        service: Ingestion service instance
    
    Returns:
        Batch status information
    """
    try:
        batch_status = await service.get_batch_status(batch_id)
        
        if not batch_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        return BatchStatusResponse(
            batch_id=batch_status["batch_id"],
            total_files=batch_status["total_files"],
            completed_jobs=batch_status["completed_jobs"],
            failed_jobs=batch_status["failed_jobs"],
            in_progress_jobs=batch_status["in_progress_jobs"],
            pending_jobs=batch_status["pending_jobs"],
            overall_status=batch_status["overall_status"],
            results=[FileUploadResult(**r) for r in batch_status["results"]],
            created_at=batch_status["created_at"],
            updated_at=batch_status["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve batch status: {str(e)}"
        )
