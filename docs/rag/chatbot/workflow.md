# Workflow Management

## 🎯 Overview

Workflow management orchestrates the entire RAG (Retrieval-Augmented Generation) process using LangGraph, a powerful state machine framework. It coordinates document retrieval, reranking, response generation, and memory management in a structured, observable, and fault-tolerant manner.

## 🔄 How Workflows Work

### **The RAG Workflow Process**
1. **Query Processing**: Parse and validate user input
2. **Document Retrieval**: Search vector store for relevant documents
3. **Document Reranking**: Improve relevance of retrieved documents
4. **Response Generation**: Generate AI response using context and history
5. **Memory Storage**: Store conversation for future reference
6. **Response Delivery**: Return structured response to user

```python
# Workflow state progression
User Query → Retrieval → Reranking → Generation → Memory → Response

# With conditional logic
if use_retrieval:
    documents = retrieve_documents(query)
    if documents:
        documents = rerank_documents(documents, query)
        response = generate_with_context(query, documents, history)
    else:
        response = generate_without_context(query, history)
else:
    response = generate_without_context(query, history)
```

## 🏗️ Workflow Architecture

### **LangGraph State Machine**
The workflow is implemented as a LangGraph state machine with typed state management:

```python
from typing import TypedDict, List, Optional, Dict, Any
from langgraph import StateGraph

class RAGWorkflowState(TypedDict):
    # Input
    query: str
    session_id: str
    user_id: Optional[str]
    soeid: Optional[str]
    use_retrieval: bool
    use_history: bool
    use_chat_history: bool
    chat_history_days: int
    metadata: Dict[str, Any]
    
    # Processing state
    documents: List[Dict[str, Any]]
    reranked_documents: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    chat_history: List[Dict[str, Any]]
    
    # Output
    response: str
    response_metadata: Dict[str, Any]
    error: Optional[str]
```

## 📋 Workflow Configuration

### **Basic Configuration**
```yaml
chatbot:
  workflow:
    type: "langgraph_rag"              # Workflow type
    
    # Node configuration
    nodes:
      retrieval:
        enabled: true                   # Enable retrieval node
        timeout: 30                     # Node timeout (seconds)
        retry_count: 3                  # Retry attempts
        
      reranking:
        enabled: true                   # Enable reranking node
        timeout: 15                     # Node timeout
        retry_count: 2                  # Retry attempts
        
      generation:
        enabled: true                   # Enable generation node
        timeout: 60                     # Node timeout
        retry_count: 3                  # Retry attempts
        
      memory:
        enabled: true                   # Enable memory node
        timeout: 10                     # Node timeout
        retry_count: 2                  # Retry attempts
    
    # Execution settings
    execution:
      max_execution_time: 120           # Total workflow timeout
      parallel_execution: false         # Sequential execution
      error_handling: "graceful"        # Error handling mode
      
    # Monitoring
    monitoring:
      enabled: true                     # Enable monitoring
      log_state_transitions: true       # Log state changes
      track_performance: true           # Track node performance
```

### **Advanced Configuration**
```yaml
chatbot:
  workflow:
    type: "langgraph_rag"
    
    # Conditional execution
    conditions:
      skip_retrieval_if_no_query: true  # Skip retrieval for empty queries
      skip_reranking_if_few_docs: true  # Skip reranking if < 3 documents
      use_cache_for_similar_queries: true # Cache similar queries
      
    # Performance optimization
    optimization:
      document_limit: 10                # Max documents to process
      context_window_size: 4000         # Max context tokens
      parallel_document_processing: true # Process documents in parallel
      cache_embeddings: true            # Cache query embeddings
      
    # Error handling
    error_handling:
      mode: "graceful"                  # graceful, strict, or fail_fast
      fallback_response: "I apologize, but I encountered an error processing your request."
      log_errors: true                  # Log all errors
      
    # Circuit breaker
    circuit_breaker:
      enabled: true                     # Enable circuit breaker
      failure_threshold: 5              # Failures before opening
      recovery_timeout: 300             # Recovery timeout (seconds)
```

## 🛠️ Workflow Implementation

### **Creating the Workflow**
```python
from langgraph import StateGraph, END
from src.rag.chatbot.workflow.rag_workflow import RAGWorkflow

class RAGWorkflow:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = self._create_retriever(config)
        self.reranker = self._create_reranker(config)
        self.generator = self._create_generator(config)
        self.memory = self._create_memory(config)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create state graph
        workflow = StateGraph(RAGWorkflowState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("memory", self.memory_node)
        
        # Define edges and conditions
        workflow.set_entry_point("retrieve")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "retrieve",
            self.should_rerank,
            {
                "rerank": "rerank",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "memory")
        workflow.add_edge("memory", END)
        
        return workflow.compile()
    
    async def execute(self, initial_state: RAGWorkflowState) -> RAGWorkflowState:
        """Execute the workflow with the given initial state."""
        
        try:
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Return error state
            return {
                **initial_state,
                "error": str(e),
                "response": self.config.get("fallback_response", "An error occurred.")
            }
```

### **Workflow Nodes Implementation**
```python
async def retrieve_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
    """Retrieve relevant documents from vector store."""
    
    if not state.get("use_retrieval", True):
        logger.info("Retrieval disabled, skipping document retrieval")
        return {**state, "documents": []}
    
    try:
        # Retrieve documents
        documents = await self.retriever.retrieve(
            query=state["query"],
            limit=self.config.get("document_limit", 10),
            filters=state.get("metadata", {})
        )
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        return {
            **state,
            "documents": documents,
            "response_metadata": {
                **state.get("response_metadata", {}),
                "retrieved_documents": len(documents)
            }
        }
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        
        if self.config.get("error_handling", {}).get("mode") == "strict":
            raise e
        
        # Graceful fallback
        return {**state, "documents": [], "error": f"Retrieval failed: {str(e)}"}

async def generate_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
    """Generate response using retrieved context and conversation history."""
    
    try:
        # Get conversation history
        conversation_history = []
        if state.get("use_history", True):
            conversation_history = await self.memory.get_history(
                session_id=state["session_id"],
                limit=10
            )
        
        # Get cross-session chat history
        chat_history = []
        if state.get("use_chat_history", False) and state.get("soeid"):
            chat_history = await self.memory.get_chat_history_by_soeid_and_date(
                soeid=state["soeid"],
                days=state.get("chat_history_days", 7),
                limit=20
            )
        
        # Generate response
        response = await self.generator.generate(
            query=state["query"],
            documents=state.get("reranked_documents", state.get("documents", [])),
            conversation_history=conversation_history,
            chat_history=chat_history,
            metadata=state.get("metadata", {})
        )
        
        logger.info("Response generated successfully")
        
        return {
            **state,
            "conversation_history": conversation_history,
            "chat_history": chat_history,
            "response": response,
            "response_metadata": {
                **state.get("response_metadata", {}),
                "generation_successful": True,
                "used_chat_history": len(chat_history) > 0,
                "used_conversation_history": len(conversation_history) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        
        fallback_response = self.config.get("fallback_response", 
                                          "I apologize, but I encountered an error generating a response.")
        
        return {
            **state,
            "response": fallback_response,
            "error": f"Generation failed: {str(e)}"
        }
```

## 📊 Workflow Monitoring

### **Performance Tracking**
```python
class WorkflowMonitor:
    def __init__(self):
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0,
            "node_performance": {}
        }
    
    async def track_execution(self, workflow_func, *args, **kwargs):
        """Track workflow execution performance."""
        
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting workflow execution {execution_id}")
            
            result = await workflow_func(*args, **kwargs)
            
            # Track success
            execution_time = time.time() - start_time
            self._update_success_metrics(execution_time)
            
            logger.info(f"Workflow execution {execution_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Track failure
            execution_time = time.time() - start_time
            self._update_failure_metrics(execution_time, str(e))
            
            logger.error(f"Workflow execution {execution_id} failed after {execution_time:.2f}s: {e}")
            raise
    
    def get_performance_report(self):
        """Get comprehensive performance report."""
        
        total_executions = self.metrics["total_executions"]
        success_rate = (self.metrics["successful_executions"] / max(total_executions, 1)) * 100
        
        return {
            "total_executions": total_executions,
            "successful_executions": self.metrics["successful_executions"],
            "failed_executions": self.metrics["failed_executions"],
            "success_rate_percent": round(success_rate, 2),
            "average_execution_time_seconds": round(self.metrics["avg_execution_time"], 2),
            "node_performance": {
                node: {
                    "executions": stats["executions"],
                    "success_rate_percent": round((stats["successes"] / max(stats["executions"], 1)) * 100, 2),
                    "average_time_seconds": round(stats["avg_time"], 2)
                }
                for node, stats in self.metrics["node_performance"].items()
            }
        }
```

## 🎯 Workflow Customization

### **Custom Workflow Nodes**
```python
class CustomWorkflowNode:
    def __init__(self, config):
        self.config = config
    
    async def custom_preprocessing_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Custom preprocessing before retrieval."""
        
        query = state["query"]
        
        # Custom query preprocessing
        processed_query = self._preprocess_query(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Add custom metadata
        custom_metadata = {
            "original_query": query,
            "processed_query": processed_query,
            "extracted_entities": entities,
            "preprocessing_timestamp": datetime.now().isoformat()
        }
        
        return {
            **state,
            "query": processed_query,
            "metadata": {
                **state.get("metadata", {}),
                **custom_metadata
            }
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Custom query preprocessing logic."""
        
        # Remove special characters
        import re
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Expand abbreviations
        abbreviations = {
            "ML": "machine learning",
            "AI": "artificial intelligence",
            "NLP": "natural language processing"
        }
        
        for abbr, expansion in abbreviations.items():
            query = query.replace(abbr, expansion)
        
        return query
```

## 🚨 Common Issues and Solutions

### **Workflow Timeout Issues**
```python
# Issue: Workflow taking too long
# Solution: Implement proper timeouts

class TimeoutWorkflow(RAGWorkflow):
    async def execute_with_timeout(self, initial_state: RAGWorkflowState, timeout: int = 120):
        """Execute workflow with timeout protection."""
        
        try:
            result = await asyncio.wait_for(
                self.workflow.ainvoke(initial_state),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Workflow execution timed out after {timeout} seconds")
            
            return {
                **initial_state,
                "error": f"Workflow timed out after {timeout} seconds",
                "response": "I apologize, but your request is taking longer than expected."
            }
```

### **State Management Issues**
```python
# Issue: State corruption or missing fields
# Solution: Implement state validation

def validate_state(state: RAGWorkflowState, required_fields: List[str]) -> bool:
    """Validate workflow state has required fields."""
    
    missing_fields = []
    for field in required_fields:
        if field not in state or state[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"Missing required state fields: {missing_fields}")
        return False
    
    return True

# Use in workflow nodes
async def retrieve_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
    if not validate_state(state, ["query"]):
        raise ValueError("Invalid state: missing required fields")
    
    # Continue with retrieval...
```

## 🎯 Best Practices

### **Error Handling**
```python
class WorkflowErrorHandler:
    def __init__(self, config):
        self.config = config
    
    async def handle_node_error(self, node_name: str, error: Exception, state: RAGWorkflowState):
        """Handle errors in workflow nodes."""
        
        error_config = self.config.get("error_handling", {})
        mode = error_config.get("mode", "graceful")
        
        if mode == "strict":
            # Fail fast - propagate error immediately
            raise error
        
        elif mode == "graceful":
            # Graceful degradation
            return await self._graceful_error_recovery(node_name, error, state)
    
    async def _graceful_error_recovery(self, node_name: str, error: Exception, state: RAGWorkflowState):
        """Implement graceful error recovery strategies."""
        
        if node_name == "retrieve":
            # If retrieval fails, continue without documents
            logger.warning(f"Retrieval failed, continuing without documents: {error}")
            return {**state, "documents": [], "error": f"Retrieval failed: {str(error)}"}
        
        elif node_name == "rerank":
            # If reranking fails, use original documents
            logger.warning(f"Reranking failed, using original documents: {error}")
            return {**state, "reranked_documents": state.get("documents", [])}
        
        elif node_name == "generate":
            # If generation fails, use fallback response
            fallback_response = self.config.get("fallback_response", 
                                              "I apologize, but I encountered an error.")
            logger.error(f"Generation failed, using fallback response: {error}")
            return {**state, "response": fallback_response, "error": f"Generation failed: {str(error)}"}
        
        else:
            # Unknown node, propagate error
            raise error
```

### **Performance Optimization**
```python
def optimize_workflow_performance(config):
    """Optimize workflow configuration for performance."""
    
    # Adjust timeouts based on expected load
    if config.get("expected_load") == "high":
        config["nodes"]["retrieval"]["timeout"] = 15  # Reduce timeout
        config["nodes"]["generation"]["timeout"] = 30  # Reduce timeout
        config["optimization"]["document_limit"] = 5   # Fewer documents
    
    # Enable caching for repeated queries
    config["optimization"]["cache_embeddings"] = True
    config["optimization"]["cache_responses"] = True
    
    # Parallel processing for better throughput
    config["optimization"]["parallel_document_processing"] = True
    
    return config
```

## 📚 Related Documentation

- **[Document Retrievers](./retrievers.md)** - Configure document retrieval
- **[Result Rerankers](./rerankers.md)** - Set up document reranking
- **[Response Generators](./generators.md)** - Configure response generation
- **[Memory Systems](./memory.md)** - Set up conversation memory
- **[Chatbot API](./api.md)** - Use workflow through API

## 🚀 Quick Examples

### **Basic Workflow Usage**
```python
# Initialize workflow
workflow = RAGWorkflow(config)

# Execute workflow
initial_state = {
    "query": "What is machine learning?",
    "session_id": "session_123",
    "use_retrieval": True,
    "use_history": True,
    "metadata": {}
}

result = await workflow.execute(initial_state)
print(f"Response: {result['response']}")
```

### **API Usage with Workflow**
```bash
# Chat with workflow processing
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain neural networks",
    "use_retrieval": true,
    "use_history": true,
    "use_chat_history": true,
    "chat_history_days": 7
  }'
```

---

**Next Steps**: 
- [Use the Chatbot API](./api.md)
- [Configure Memory Systems](./memory.md)
- [Set up Response Generators](./generators.md)
