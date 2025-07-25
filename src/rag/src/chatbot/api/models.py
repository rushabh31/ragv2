from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class ChatSession(BaseModel):
    """Model for chat sessions."""
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserHistoryMessage(BaseModel):
    """Model for user history messages with session information."""
    session_id: str
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request model for chat."""
    query: str
    session_id: Optional[str] = None
    use_retrieval: bool = True
    use_history: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """Model for retrieved documents."""
    content: str
    metadata: Dict[str, Any]
    score: float


class ChatResponse(BaseModel):
    """Response model for chat."""
    session_id: str
    soeid: str
    response: str
    created_at: datetime
    query: str
    retrieved_documents: Optional[List[RetrievedDocument]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    session_id: str
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserHistoryResponse(BaseModel):
    """Response model for user history by SOEID."""
    soeid: str
    messages: List[UserHistoryMessage]
    total_messages: int
    total_sessions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """Response model for session list."""
    sessions: List[ChatSession]
    total: int
    page: int
    page_size: int


class FeedbackRequest(BaseModel):
    """Request model for feedback on chat responses."""
    session_id: str
    message_id: str
    feedback_type: str
    score: Optional[float] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str


class SOEIDSessionHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class SOEIDHistoryResponse(BaseModel):
    soeid: str
    sessions: List[SOEIDSessionHistory]
    total_sessions: int
    total_messages: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionHistoryWithSOEIDResponse(BaseModel):
    session_id: str
    soeid: str
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)
