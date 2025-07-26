import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status, Form
import json

from .models import (
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SOEIDHistoryResponse,
    SessionHistoryWithSOEIDResponse,
    SessionListForSOEIDResponse,
    ThreadListResponse,
    MemoryStatsResponse,
    FeedbackRequest,
    FeedbackResponse
)
from .service import ChatbotService
from src.rag.core.exceptions.exceptions import GenerationError, MemoryError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/chat", tags=["chatbot"])

# Service dependency
def get_chatbot_service():
    """Dependency to get chatbot service instance."""
    return ChatbotService()

@router.post("/message", response_model=ChatResponse)
async def send_message(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_retrieval: bool = Form(True),
    use_history: bool = Form(True),
    use_chat_history: bool = Form(False),
    chat_history_days: int = Form(7),
    metadata_json: Optional[str] = Form(None),
    soeid: str = Header(...),
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Send a message and get a response."""
    try:
        # Parse metadata if provided
        metadata = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    metadata = {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata JSON: {metadata_json}")

        # Always inject SOEID into metadata
        metadata["soeid"] = soeid

        result = await service.process_chat(
            user_id=soeid,
            query=query,
            session_id=session_id,
            use_retrieval=use_retrieval,
            use_history=use_history,
            use_chat_history=use_chat_history,
            chat_history_days=chat_history_days,
            metadata=metadata
        )
        
        return ChatResponse(
            session_id=result["session_id"],
            soeid=soeid,
            response=result["response"],
            created_at=result["created_at"],
            query=result["query"],
            retrieved_documents=result["retrieved_documents"],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message processing failed: {str(e)}"
        )

# Also add a JSON endpoint for API clients that prefer JSON
@router.post("/message/json", response_model=ChatResponse)
async def send_message_json(
    request: ChatRequest,
    soeid: str = Header(...),
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Send a message and get a response (JSON endpoint)."""
    try:
        # Always inject SOEID into metadata
        metadata = dict(request.metadata) if request.metadata else {}
        metadata["soeid"] = soeid

        result = await service.process_chat(
            user_id=soeid,
            query=request.query,
            session_id=request.session_id,
            use_retrieval=request.use_retrieval,
            use_history=request.use_history,
            use_chat_history=request.use_chat_history,
            chat_history_days=request.chat_history_days,
            metadata=metadata
        )
        
        return ChatResponse(
            session_id=result["session_id"],
            soeid=soeid,
            response=result["response"],
            created_at=result["created_at"],
            query=result["query"],
            retrieved_documents=result["retrieved_documents"],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message processing failed: {str(e)}"
        )


@router.get("/history/{soeid}", response_model=SOEIDHistoryResponse)
async def get_all_history_by_soeid(
    soeid: str,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Return all chat history for a SOEID, grouped by session."""
    result = await service.get_all_history_by_soeid(soeid)
    return SOEIDHistoryResponse(**result)

@router.get("/history/session/{session_id}", response_model=SessionHistoryWithSOEIDResponse)
async def get_session_history_with_soeid(
    session_id: str,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Return all chat history for a session_id, including soeid."""
    result = await service.get_session_history_with_soeid(session_id)
    return SessionHistoryWithSOEIDResponse(**result)

@router.delete("/history/{soeid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history_by_soeid(
    soeid: str,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Delete all chat history for a SOEID."""
    success = await service.delete_history_by_soeid(soeid)
    if not success:
        raise HTTPException(status_code=404, detail=f"No history found for SOEID {soeid}")

@router.delete("/history/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history_by_session_id(
    session_id: str,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Delete all chat history for a session_id."""
    success = await service.delete_history_by_session_id(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"No history found for session {session_id}")

@router.get("/debug/all-messages")
async def debug_all_messages(service: ChatbotService = Depends(get_chatbot_service)):
    """Return all messages in all session namespaces for debugging SOEID storage."""
    memory = await service._get_memory_instance()
    all_messages = []
    # Try to get all session_ids from the memory store
    session_ids = set()
    if hasattr(memory._store, 'namespaces'):
        for ns in memory._store.namespaces():
            if isinstance(ns, tuple) and len(ns) == 2 and ns[1] == "conversation":
                session_ids.add(ns[0])
    else:
        session_ids = set(getattr(memory, '_sessions', {}).keys())
    for session_id in session_ids:
        session_namespace = (session_id, "conversation")
        try:
            msgs = memory._store.search(session_namespace, query="", limit=10000)
            for msg in msgs:
                all_messages.append({
                    "session_id": msg.value.get("session_id"),
                    "soeid": msg.value.get("soeid"),
                    "role": msg.value.get("role"),
                    "content": msg.value.get("content"),
                    "timestamp": msg.value.get("timestamp"),
                    "metadata": msg.value.get("metadata", {})
                })
        except Exception as e:
            continue
    return {"messages": all_messages, "total": len(all_messages)}


# New LangGraph Memory Endpoints

@router.get("/sessions/{soeid}", response_model=SessionListForSOEIDResponse)
async def get_sessions_for_soeid(
    soeid: str,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Get all sessions for a specific SOEID with metadata."""
    result = await service.get_sessions_for_soeid(soeid)
    return SessionListForSOEIDResponse(**result)


@router.get("/threads", response_model=ThreadListResponse)
async def get_all_threads(
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Get all threads (sessions) in the system."""
    result = await service.get_all_threads()
    return ThreadListResponse(**result)


@router.get("/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Get memory system statistics."""
    result = await service.get_memory_stats()
    return MemoryStatsResponse(**result)


@router.get("/sessions/{soeid}/history", response_model=SOEIDHistoryResponse)
async def get_soeid_history_by_sessions(
    soeid: str,
    limit: Optional[int] = Query(None, description="Maximum number of sessions to return"),
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Get chat history for a SOEID, organized by sessions (alternative endpoint)."""
    result = await service.get_all_history_by_soeid(soeid)
    
    # Apply limit if specified
    if limit and result.get("sessions"):
        result["sessions"] = result["sessions"][:limit]
        result["total_sessions"] = len(result["sessions"])
    
    return SOEIDHistoryResponse(**result)


@router.get("/threads/{thread_id}/history", response_model=SessionHistoryWithSOEIDResponse)
async def get_thread_history(
    thread_id: str,
    limit: Optional[int] = Query(None, description="Maximum number of messages to return"),
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Get chat history for a specific thread (session) ID."""
    result = await service.get_session_history_with_soeid(thread_id)
    
    # Apply limit if specified
    if limit and result.get("messages"):
        result["messages"] = result["messages"][-limit:]  # Get most recent messages
    
    return SessionHistoryWithSOEIDResponse(**result)
