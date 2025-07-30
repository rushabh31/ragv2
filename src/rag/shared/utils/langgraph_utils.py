"""Utility functions and types for LangGraph integration."""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Union, Literal, cast
from enum import Enum
import json
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

# Type definitions for workflow state
class MessageRole(str, Enum):
    """Enum for message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Message model for conversation."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """Model for a retrieved document."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class RAGWorkflowState(TypedDict, total=False):
    """Type for RAG workflow state."""
    # Input fields
    query: str
    session_id: str
    user_id: str
    soeid: str  # Source of Entity ID for user identification
    messages: List[Message]
    
    # Chat history configuration
    use_chat_history: bool
    chat_history_days: int
    
    # Processing fields
    retrieved_documents: List[RetrievedDocument]
    reranked_documents: List[RetrievedDocument]
    generation_parameters: Dict[str, Any]
    
    # Output fields
    response: str
    error: Optional[str]
    metrics: Dict[str, Any]


class RAGDecision(str, Enum):
    """Decision points in the RAG workflow."""
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    REFINE = "refine"
    END = "end"
    ERROR = "error"


# Utility functions for workflow
def create_rag_workflow() -> StateGraph:
    """Create a RAG workflow graph with LangGraph.
    
    Returns:
        StateGraph: The workflow graph
    """
    # Define the workflow state
    workflow = StateGraph(RAGWorkflowState)
    
    # Add nodes for each step in the RAG process
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("update_memory", update_memory_node)
    workflow.add_node("decide_next_step", decide_next_step)
    
    # Define the workflow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "update_memory")
    workflow.add_edge("update_memory", "decide_next_step")
    
    # Decision branching
    workflow.add_conditional_edges(
        "decide_next_step",
        lambda state: decide_next_action(state),
        {
            RAGDecision.RETRIEVE: "retrieve",
            RAGDecision.RERANK: "rerank",
            RAGDecision.GENERATE: "generate",
            RAGDecision.END: END,
            RAGDecision.ERROR: END
        }
    )
    
    return workflow


# Node implementations
async def retrieve_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Node for retrieving documents based on query.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with retrieved documents
    """
    try:
        from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever
        from src.rag.shared.utils.config_manager import ConfigManager
        
        config = ConfigManager().get_section("chatbot.retrieval", {})
        retriever = VectorRetriever(config)
        
        # Start with empty metrics
        if "metrics" not in state:
            state["metrics"] = {}
        state["metrics"]["retrieval"] = {}
        
        # Retrieve relevant documents
        start_time = __import__("time").time()
        query = state.get("query", "")
        documents = await retriever.retrieve(query)
        
        # Convert to workflow document format
        retrieved_docs = []
        for doc in documents:
            retrieved_docs.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=doc.metadata.get("score", 0.0)
            ))
        
        # Update state with retrieved documents and metrics
        state["retrieved_documents"] = retrieved_docs
        state["metrics"]["retrieval"]["time_seconds"] = __import__("time").time() - start_time
        state["metrics"]["retrieval"]["document_count"] = len(retrieved_docs)
        
        return state
    except Exception as e:
        logger.error(f"Error in retrieve node: {str(e)}", exc_info=True)
        state["error"] = f"Retrieval error: {str(e)}"
        return state


async def rerank_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Node for reranking retrieved documents.
    
    Args:
        state: Current workflow state with retrieved documents
        
    Returns:
        Updated workflow state with reranked documents
    """
    try:
        from src.rag.chatbot.rerankers.cross_encoder_reranker import CrossEncoderReranker
        from src.rag.core.interfaces.base import Document
        from src.rag.shared.utils.config_manager import ConfigManager
        
        config = ConfigManager().get_section("chatbot.reranking", {})
        reranker = CrossEncoderReranker(config)
        
        # Start metrics tracking
        if "metrics" not in state:
            state["metrics"] = {}
        state["metrics"]["reranking"] = {}
        start_time = __import__("time").time()
        
        # Skip if no documents were retrieved
        if not state.get("retrieved_documents", []):
            state["reranked_documents"] = []
            state["metrics"]["reranking"]["skipped"] = True
            return state
        
        # Convert to Document format for reranker
        documents = []
        for doc in state.get("retrieved_documents", []):
            documents.append(Document(
                content=doc.content,
                metadata=doc.metadata
            ))
        
        # Rerank documents
        query = state.get("query", "")
        reranked_docs = await reranker.rerank(query, documents)
        
        # Convert back to workflow document format
        reranked = []
        for doc in reranked_docs:
            reranked.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=doc.metadata.get("reranker_score", doc.metadata.get("score", 0.0))
            ))
        
        # Update state with reranked documents and metrics
        state["reranked_documents"] = reranked
        state["metrics"]["reranking"]["time_seconds"] = __import__("time").time() - start_time
        state["metrics"]["reranking"]["document_count"] = len(reranked)
        
        return state
    except Exception as e:
        logger.error(f"Error in rerank node: {str(e)}", exc_info=True)
        state["error"] = f"Reranking error: {str(e)}"
        return state


async def generate_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Node for generating response from context documents.
    
    Args:
        state: Current workflow state with reranked documents
        
    Returns:
        Updated workflow state with generated response
    """
    try:
        from src.rag.core.interfaces.base import Document
        from src.rag.shared.utils.config_manager import ConfigManager
        from src.models.generation.model_factory import GenerationModelFactory
        
        # Use factory to create generator - it will automatically read config from YAML
        generator = GenerationModelFactory.create_model()
        
        # Start metrics tracking
        if "metrics" not in state:
            state["metrics"] = {}
        state["metrics"]["generation"] = {}
        start_time = __import__("time").time()
        
        # Get conversation history from memory
        session_id = state.get("session_id", "default")
        query = state.get("query", "")
        soeid = state.get("soeid")  # Get SOEID from state if available
        use_chat_history = state.get("use_chat_history", False)
        chat_history_days = state.get("chat_history_days", 7)
        
        try:
            # Use the singleton memory instance to avoid creating multiple instances
            from examples.rag.chatbot.api.service import ChatbotService
            
            # Get the singleton memory instance
            service = ChatbotService()
            memory = await service._get_memory_instance()
            
            conversation_history = []
            
            # If chat history is enabled and we have a SOEID, get date-filtered history
            if use_chat_history and soeid:
                logger.info(f"Getting chat history for SOEID {soeid} within {chat_history_days} days")
                chat_history = await memory.get_chat_history_by_soeid_and_date(
                    soeid=soeid,
                    days=chat_history_days,
                    limit=20  # Limit to last 20 interactions across all sessions
                )
                
                # Convert chat history to conversation format
                for msg in chat_history:
                    conversation_history.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                        "session_id": msg.get("session_id", ""),
                        "timestamp": msg.get("timestamp", "")
                    })
                
                logger.info(f"Retrieved {len(conversation_history)} messages from chat history")
            
            # If no chat history or SOEID, fall back to session-based history
            if not conversation_history:
                if soeid:
                    session_history = await memory.get_user_relevant_history_by_soeid(
                        soeid=soeid,
                        query=query,
                        limit=10  # Limit to last 10 interactions
                    )
                else:
                    session_history = await memory.get_relevant_history(
                        session_id=session_id,
                        query=query,
                        limit=10  # Limit to last 10 interactions
                    )
                
                # Convert to the format expected by the generator
                for msg in session_history:
                    conversation_history.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
        except Exception as e:
            logger.warning(f"Failed to get conversation history from memory: {str(e)}")
            # Fall back to state messages
            messages = state.get("messages", [])
            conversation_history = []
            for msg in messages:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Convert to Document format for generator
        documents = []
        try:
            from src.rag.core.interfaces.base import Document
            for doc in state.get("reranked_documents", []) or state.get("retrieved_documents", []):
                documents.append(Document(
                    content=doc.content,
                    metadata=doc.metadata
                ))
        except ImportError:
            # Fall back to a simple dict structure if Document can't be imported
            for doc in state.get("reranked_documents", []) or state.get("retrieved_documents", []):
                documents.append({
                    "content": doc.content,
                    "metadata": doc.metadata
                })
        
        # Generate response
        query = state.get("query", "")
        gen_config = state.get("generation_parameters", {})
        
        response = await generator.generate(
            query=query,
            documents=documents,
            conversation_history=conversation_history,
            config=gen_config
        )
        
        # Update state with response and metrics
        state["response"] = response
        state["metrics"]["generation"]["time_seconds"] = __import__("time").time() - start_time
        
        # Add the new messages to the conversation
        state["messages"] = state.get("messages", []) + [
            Message(role=MessageRole.USER, content=query),
            Message(role=MessageRole.ASSISTANT, content=response)
        ]
        
        return state
    except Exception as e:
        logger.error(f"Error in generate node: {str(e)}", exc_info=True)
        state["error"] = f"Generation error: {str(e)}"
        state["response"] = "I apologize, but I encountered an error while generating a response. Please try again."
        return state


async def update_memory_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Node for updating conversation memory.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state
    """
    try:
        # Use the singleton memory instance to avoid recreating memory
        from examples.rag.chatbot.api.service import ChatbotService
        
        # Get the singleton memory instance
        service = ChatbotService()
        memory = await service._get_memory_instance()
        
        # Start metrics tracking
        if "metrics" not in state:
            state["metrics"] = {}
        state["metrics"]["memory"] = {}
        start_time = __import__("time").time()
        
        # Get session ID, query, and SOEID
        session_id = state.get("session_id", "default")
        query = state.get("query", "")
        response = state.get("response", "")
        soeid = state.get("soeid")
        
        # Add interaction to memory
        if query and response:
            metadata = {
                "workflow_run_id": state.get("workflow_run_id", ""),
                "retrieved_documents_count": len(state.get("retrieved_documents", [])),
                "reranked_documents_count": len(state.get("reranked_documents", []))
            }
            
            # Add SOEID to metadata if available
            if soeid:
                metadata["soeid"] = soeid
            
            success = await memory.add(
                session_id=session_id,
                query=query,
                response=response,
                metadata=metadata
            )
            
            if success:
                state["metrics"]["memory"]["time_seconds"] = __import__("time").time() - start_time
                state["metrics"]["memory"]["success"] = True
                logger.debug(f"Updated memory for session {session_id}")
            else:
                state["metrics"]["memory"]["success"] = False
                logger.warning(f"Failed to update memory for session {session_id}")
        
        return state
    except Exception as e:
        logger.error(f"Error in update_memory node: {str(e)}", exc_info=True)
        state["error"] = f"Memory update error: {str(e)}"
        return state


def decide_next_step(state: RAGWorkflowState) -> RAGWorkflowState:
    """Node for deciding the next step in the workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state
    """
    # This node doesn't modify state, it just helps with decision branching
    return state


def decide_next_action(state: RAGWorkflowState) -> RAGDecision:
    """Decide the next action in the workflow based on state.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next action to take
    """
    # Check for errors
    if "error" in state:
        return RAGDecision.ERROR
    
    # Check if we need to retrieve more documents
    if not state.get("retrieved_documents", []):
        # No documents were retrieved, check if we should retry
        return RAGDecision.END
    
    # By default, end the workflow
    return RAGDecision.END
