import logging
from typing import Dict, List, Any, Optional, TypedDict, Union
import asyncio
import time
import uuid
from datetime import datetime

from langgraph.graph import StateGraph
from src.rag.src.shared.utils.langgraph_utils import (
    RAGWorkflowState,
    Message,
    MessageRole,
    RetrievedDocument,
    create_rag_workflow
)
from src.rag.src.shared.utils.config_manager import ConfigManager
from src.rag.src.shared.cache.cache_manager import CacheManager
from src.rag.src.chatbot.memory.memory_factory import MemoryFactory

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manager for RAG workflows using LangGraph."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the workflow manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config_manager = ConfigManager()
        self.config = config or {}
        
        # Initialize cache
        self.cache_manager = CacheManager(self.config.get("cache", {}))
        
        # Initialize memory using singleton
        from src.rag.src.chatbot.memory.memory_singleton import memory_singleton
        self._memory = None  # Will be initialized lazily
        
        # Create workflow graph
        self._workflow = create_rag_workflow()
        
        # Compile workflow for execution
        self.workflow_app = self._workflow.compile()
        
        # Track active workflow runs
        self._active_runs: Dict[str, Dict[str, Any]] = {}
    
    async def _get_conversation_history(self, session_id: str) -> List[Message]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        # Get memory instance
        if self._memory is None:
            from src.rag.src.chatbot.memory.memory_singleton import memory_singleton
            self._memory = await memory_singleton.get_memory()
        
        # Get history from memory
        history = await self._memory.get_history(session_id)
        
        # Convert to Message objects
        messages = []
        for msg in history:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            messages.append(Message(
                role=role,
                content=msg["content"],
                metadata=msg.get("metadata", {})
            ))
        
        return messages
    
    async def process_query(self, 
                          user_id: str,
                          query: str,
                          session_id: Optional[str] = None,
                          workflow_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user query through the RAG workflow.
        
        Args:
            user_id: User identifier
            query: User query
            session_id: Optional session identifier
            workflow_params: Optional parameters for the workflow
            
        Returns:
            Dictionary with workflow results
        """
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        
        # Create a workflow run ID
        run_id = str(uuid.uuid4())
        
        try:
            # Record start time for metrics
            start_time = time.time()
            
            # Get conversation history
            messages = await self._get_conversation_history(session_id)
            
            # Initialize workflow state
            initial_state: RAGWorkflowState = {
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
                "messages": messages,
                "metrics": {"start_time": start_time}
            }
            
            # Add workflow parameters if provided
            if workflow_params:
                for key, value in workflow_params.items():
                    initial_state[key] = value
            
            # Store active run
            self._active_runs[run_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "query": query,
                "start_time": start_time,
                "status": "running"
            }
            
            # Execute workflow
            logger.info(f"Starting workflow run {run_id} for session {session_id}")
            result = await self.workflow_app.ainvoke(initial_state)
            
            # Record end time and duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Update run status
            self._active_runs[run_id]["status"] = "completed"
            self._active_runs[run_id]["end_time"] = end_time
            self._active_runs[run_id]["duration"] = duration
            
            # Process result
            response = result.get("response", "I'm sorry, I couldn't generate a response.")
            
            # Store in memory if response was successfully generated
            if "error" not in result:
                # Ensure memory is initialized
                if self._memory is None:
                    from src.rag.src.chatbot.memory.memory_singleton import memory_singleton
                    self._memory = await memory_singleton.get_memory()
                
                await self._memory.add(
                    session_id=session_id,
                    query=query,
                    response=response,
                    metadata={
                        "workflow_run_id": run_id,
                        "processing_time": duration
                    }
                )
            
            # Prepare response
            final_result = {
                "session_id": session_id,
                "response": response,
                "query": query,
                "created_at": datetime.now().isoformat(),
                "workflow_run_id": run_id,
                "retrieved_documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for doc in result.get("retrieved_documents", []) or result.get("reranked_documents", [])
                ],
                "metrics": result.get("metrics", {}),
                "error": result.get("error")
            }
            
            logger.info(f"Workflow run {run_id} completed in {duration:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            
            # Update run status
            if run_id in self._active_runs:
                self._active_runs[run_id]["status"] = "failed"
                self._active_runs[run_id]["error"] = str(e)
            
            # Return error response
            return {
                "session_id": session_id,
                "response": "I apologize, but I encountered an error processing your request.",
                "query": query,
                "created_at": datetime.now().isoformat(),
                "workflow_run_id": run_id,
                "error": str(e)
            }
    
    async def reset_memory_reference(self):
        """Reset the local memory reference (useful when singleton is reset)."""
        self._memory = None
    
    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get the status of a workflow run.
        
        Args:
            run_id: Workflow run identifier
            
        Returns:
            Dictionary with run status
        """
        if run_id in self._active_runs:
            return self._active_runs[run_id]
        return {"error": "Run not found"}
    
    def cleanup_old_runs(self, max_age_hours: int = 24) -> int:
        """Clean up old workflow runs to free memory.
        
        Args:
            max_age_hours: Maximum age of runs to keep in hours
            
        Returns:
            Number of runs cleaned up
        """
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        runs_to_remove = []
        
        for run_id, run_info in self._active_runs.items():
            start_time = run_info.get("start_time", 0)
            if now - start_time > max_age_seconds:
                runs_to_remove.append(run_id)
        
        # Remove old runs
        for run_id in runs_to_remove:
            del self._active_runs[run_id]
        
        return len(runs_to_remove)
