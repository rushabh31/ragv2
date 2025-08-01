"""
Advanced LangGraph Workflow Utilities for RAG Chatbot

This module provides enterprise-grade workflow management for the RAG chatbot
using LangGraph. It includes state management, node implementations, and
workflow orchestration with comprehensive error handling and monitoring.

Author: Expert Python Developer
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Union, Literal, cast
from enum import Enum
import json
import time
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import environment manager for configuration
from src.utils.env_manager import env

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


class WorkflowManager:
    """
    Advanced workflow manager for RAG operations.
    
    Provides enterprise-grade workflow orchestration with:
    - State management and validation
    - Error handling and recovery
    - Performance monitoring
    - Configurable retry logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.workflow = self._create_workflow()
        
        # Performance settings
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        logger.info("WorkflowManager initialized with advanced configuration")
    
    def _create_workflow(self) -> StateGraph:
        """Create a RAG workflow graph with LangGraph."""
        workflow = StateGraph(RAGWorkflowState)
        
        # Add nodes for each step in the RAG process
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("update_memory", self._update_memory_node)
        workflow.add_node("decide_next_step", self._decide_next_step)
        
        # Define the workflow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "update_memory")
        workflow.add_edge("update_memory", "decide_next_step")
        
        # Decision branching
        workflow.add_conditional_edges(
            "decide_next_step",
            lambda state: self._decide_next_action(state),
            {
                RAGDecision.RETRIEVE: "retrieve",
                RAGDecision.RERANK: "rerank",
                RAGDecision.GENERATE: "generate",
                RAGDecision.END: END,
                RAGDecision.ERROR: END
            }
        )
        
        return workflow
    
    async def _retrieve_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Node for retrieving documents based on query."""
        try:
            from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever
            from src.rag.shared.utils.config_manager import ConfigManager
            
            config = ConfigManager().get_section("chatbot.retrieval", {})
            retriever = VectorRetriever(config)
            
            # Initialize metrics
            if "metrics" not in state:
                state["metrics"] = {}
            state["metrics"]["retrieval"] = {}
            
            # Retrieve relevant documents
            start_time = time.time()
            query = state.get("query", "")
            documents = await retriever.retrieve(query)
            
            # Convert to workflow document format
            retrieved_docs = []
            for doc in documents:
                retrieved_docs.append(RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata or {},
                    score=getattr(doc, 'score', None)
                ))
            
            # Update state
            state["retrieved_documents"] = retrieved_docs
            state["metrics"]["retrieval"]["time_seconds"] = time.time() - start_time
            state["metrics"]["retrieval"]["document_count"] = len(retrieved_docs)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve node: {str(e)}", exc_info=True)
            state["error"] = f"Retrieval error: {str(e)}"
            return state
    
    async def _rerank_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Node for reranking retrieved documents."""
        try:
            from src.rag.chatbot.rerankers.reranker_factory import RerankerFactory
            from src.rag.shared.utils.config_manager import ConfigManager
            
            config = ConfigManager().get_section("chatbot.reranking", {})
            reranker = RerankerFactory.create_reranker(config.get("provider", "custom"))
            
            # Initialize metrics
            if "metrics" not in state:
                state["metrics"] = {}
            state["metrics"]["reranking"] = {}
            
            # Rerank documents
            start_time = time.time()
            query = state.get("query", "")
            retrieved_docs = state.get("retrieved_documents", [])
            
            # Convert to reranker format
            documents = []
            for doc in retrieved_docs:
                documents.append(type('Document', (), {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': doc.score
                })())
            
            reranked_docs = await reranker.rerank(query, documents)
            
            # Convert back to workflow format
            reranked_workflow_docs = []
            for doc in reranked_docs:
                reranked_workflow_docs.append(RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata or {},
                    score=getattr(doc, 'score', None)
                ))
            
            # Update state
            state["reranked_documents"] = reranked_workflow_docs
            state["metrics"]["reranking"]["time_seconds"] = time.time() - start_time
            state["metrics"]["reranking"]["document_count"] = len(reranked_workflow_docs)
            
            logger.info(f"Reranked to {len(reranked_workflow_docs)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Error in rerank node: {str(e)}", exc_info=True)
            # Continue with original documents if reranking fails
            state["reranked_documents"] = state.get("retrieved_documents", [])
            state["metrics"]["reranking"] = {"error": str(e)}
            return state
    
    async def _generate_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Node for generating response from context documents."""
        try:
            from src.rag.chatbot.generators.generator_factory import GeneratorFactory
            from src.rag.shared.utils.config_manager import ConfigManager
            
            config = ConfigManager().get_section("chatbot.generation", {})
            generator = GeneratorFactory.create_generator(config.get("provider", "vertex"))
            
            # Initialize metrics
            if "metrics" not in state:
                state["metrics"] = {}
            state["metrics"]["generation"] = {}
            
            # Prepare generation
            start_time = time.time()
            query = state.get("query", "")
            reranked_docs = state.get("reranked_documents", [])
            
            # Convert documents to generator format
            documents = []
            for doc in reranked_docs:
                documents.append(type('Document', (), {
                    'content': doc.content,
                    'metadata': doc.metadata
                })())
            
            # Get conversation history
            conversation_history = []
            messages = state.get("messages", [])
            for msg in messages[-10:]:  # Last 10 messages
                conversation_history.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Handle chat history if enabled
            if state.get("use_chat_history", False):
                try:
                    from examples.rag.chatbot.api.service import ChatbotService
                    service = ChatbotService()
                    memory = await service._get_memory_instance()
                    
                    soeid = state.get("soeid")
                    if soeid:
                        chat_history_days = state.get("chat_history_days", 7)
                        chat_history = await memory.get_chat_history_by_soeid_and_date(
                            soeid=soeid,
                            days=chat_history_days,
                            limit=20
                        )
                        
                        # Add chat history to conversation context
                        for entry in chat_history[-10:]:  # Last 10 from chat history
                            conversation_history.append({
                                "role": "user",
                                "content": entry.get("query", "")
                            })
                            conversation_history.append({
                                "role": "assistant", 
                                "content": entry.get("response", "")
                            })
                        
                        logger.info(f"Added {len(chat_history)} chat history entries")
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve chat history: {str(e)}")
            
            # Generate response
            gen_config = config.get("config", {})
            response = await generator.generate(
                query=query,
                documents=documents,
                conversation_history=conversation_history,
                config=gen_config
            )
            
            # Update state
            state["response"] = response
            state["metrics"]["generation"]["time_seconds"] = time.time() - start_time
            
            # Add new messages to conversation
            state["messages"] = state.get("messages", []) + [
                Message(role=MessageRole.USER, content=query),
                Message(role=MessageRole.ASSISTANT, content=response)
            ]
            
            logger.info("Generated response successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in generate node: {str(e)}", exc_info=True)
            state["error"] = f"Generation error: {str(e)}"
            state["response"] = "I apologize, but I encountered an error while generating a response. Please try again."
            return state
    
    async def _update_memory_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Node for updating conversation memory."""
        try:
            from examples.rag.chatbot.api.service import ChatbotService
            
            # Get the singleton memory instance
            service = ChatbotService()
            memory = await service._get_memory_instance()
            
            # Initialize metrics
            if "metrics" not in state:
                state["metrics"] = {}
            state["metrics"]["memory"] = {}
            start_time = time.time()
            
            # Get session data
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
                
                if soeid:
                    metadata["soeid"] = soeid
                
                success = await memory.add(
                    session_id=session_id,
                    query=query,
                    response=response,
                    metadata=metadata
                )
                
                state["metrics"]["memory"]["time_seconds"] = time.time() - start_time
                state["metrics"]["memory"]["success"] = success
                
                if success:
                    logger.debug(f"Updated memory for session {session_id}")
                else:
                    logger.warning(f"Failed to update memory for session {session_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in update_memory node: {str(e)}", exc_info=True)
            state["error"] = f"Memory update error: {str(e)}"
            return state
    
    def _decide_next_step(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Node for deciding the next step in the workflow."""
        return state
    
    def _decide_next_action(self, state: RAGWorkflowState) -> RAGDecision:
        """Decide the next action in the workflow based on state."""
        # Check for errors
        if "error" in state:
            return RAGDecision.ERROR
        
        # Check if we need to retrieve more documents
        if not state.get("retrieved_documents", []):
            return RAGDecision.END
        
        # By default, end the workflow
        return RAGDecision.END


# Legacy functions for backward compatibility
def create_rag_workflow() -> StateGraph:
    """Create a RAG workflow graph with LangGraph (legacy function)."""
    manager = WorkflowManager()
    return manager.workflow


async def retrieve_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Legacy retrieve node function."""
    manager = WorkflowManager()
    return await manager._retrieve_node(state)


async def rerank_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Legacy rerank node function."""
    manager = WorkflowManager()
    return await manager._rerank_node(state)


async def generate_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Legacy generate node function."""
    manager = WorkflowManager()
    return await manager._generate_node(state)


async def update_memory_node(state: RAGWorkflowState) -> RAGWorkflowState:
    """Legacy update memory node function."""
    manager = WorkflowManager()
    return await manager._update_memory_node(state)


def decide_next_step(state: RAGWorkflowState) -> RAGWorkflowState:
    """Legacy decide next step function."""
    manager = WorkflowManager()
    return manager._decide_next_step(state)


def decide_next_action(state: RAGWorkflowState) -> RAGDecision:
    """Legacy decide next action function."""
    manager = WorkflowManager()
    return manager._decide_next_action(state)
