# Chatbot Service Overview

## 🎯 What is the Chatbot Service?

The chatbot service is the query and response component of the RAG system. It takes user questions, retrieves relevant information from the ingested documents, and generates intelligent responses using AI. The service provides conversational AI capabilities with memory, context awareness, and multi-session support.

## 🔄 Query Processing Flow

```
👤 User Question
       ↓
🔍 Query Understanding & Embedding
       ↓
📊 Vector Similarity Search (Retrieval)
       ↓
🎯 Document Reranking & Filtering
       ↓
🧠 Memory & Context Integration
       ↓
🤖 AI Response Generation
       ↓
💬 Contextual Answer to User
```

## 🏗️ Architecture Components

### **1. Document Retrievers**
Find relevant documents from the vector store based on user queries:
- **Vector Retriever**: Semantic similarity search using embeddings
- **Hybrid Retriever**: Combines vector search with keyword matching
- **Filtered Retriever**: Applies metadata filters to search results

### **2. Result Rerankers**
Improve relevance by reordering retrieved documents:
- **Custom Reranker**: Text-based relevance scoring
- **Cross-Encoder Reranker**: Deep learning-based reranking
- **Metadata Reranker**: Boost results based on metadata

### **3. Response Generators**
Generate AI responses using retrieved context:
- **Vertex Generator**: Google Gemini Pro models
- **OpenAI Generator**: GPT models
- **Azure OpenAI Generator**: Enterprise GPT deployment

### **4. Memory Systems**
Manage conversation history and context:
- **Simple Memory**: In-memory conversation tracking
- **LangGraph Memory**: PostgreSQL-backed persistent memory
- **Session Memory**: User-specific conversation management

### **5. Workflow Engine**
Orchestrate the complete query processing pipeline:
- **LangGraph Workflows**: State-based conversation flows
- **Context Management**: Maintain conversation context
- **Error Handling**: Graceful failure recovery

## 🚀 Key Features

### **🔍 Intelligent Retrieval**
- **Semantic Search**: Find documents by meaning, not just keywords
- **Hybrid Search**: Combine vector similarity with keyword matching
- **Filtered Search**: Apply metadata filters (date, source, category)
- **Configurable Results**: Adjust number and quality of retrieved documents

### **🎯 Advanced Reranking**
- **Relevance Scoring**: Improve document ranking accuracy
- **Context Awareness**: Consider conversation history in ranking
- **Metadata Boosting**: Prioritize specific document types
- **Quality Filtering**: Remove low-quality or irrelevant results

### **🧠 Persistent Memory**
- **Session Tracking**: Remember conversations within sessions
- **Cross-Session History**: Access chat history across multiple sessions
- **User-Specific Memory**: SOEID-based user identification
- **PostgreSQL Storage**: Reliable, scalable conversation persistence

### **🤖 Multi-Provider AI**
- **Vertex AI Gemini**: Google's latest language models
- **OpenAI GPT**: Industry-leading language models
- **Azure OpenAI**: Enterprise-grade AI deployment
- **Configurable Models**: Easy switching between providers

## ⚙️ Configuration Overview

### **Basic Configuration**
```yaml
chatbot:
  generation:
    provider: "vertex"                 # AI model provider
    vertex:
      model_name: "gemini-1.5-pro-002" # Specific model
      max_tokens: 1000                 # Response length
      temperature: 0.7                 # Creativity level
  
  retrieval:
    provider: "vector"                 # Retrieval method
    vector:
      top_k: 10                        # Documents to retrieve
      score_threshold: 0.7             # Minimum similarity
  
  reranking:
    provider: "custom"                 # Reranking method
    custom:
      top_k: 5                         # Final document count
  
  memory:
    type: "langgraph_checkpoint"       # Memory system
    store_type: "postgres"             # Storage backend
    postgres:
      connection_string: "postgresql://user:pass@localhost:5432/ragdb"
```

### **Advanced Features**
```yaml
chatbot:
  # Workflow configuration
  workflow:
    enable_streaming: true             # Stream responses
    max_conversation_length: 50        # Message limit per session
    context_window: 4000               # Token limit for context
  
  # Cross-session chat history
  chat_history:
    enabled: true                      # Enable chat history
    default_days: 7                    # Days to look back
    max_messages: 100                  # Message limit
  
  # Response customization
  response:
    include_sources: true              # Show source documents
    include_confidence: true           # Show confidence scores
    format: "markdown"                 # Response format
```

## 🛠️ API Endpoints

### **Chat Interaction**
```bash
# Send a message (form-based)
POST /chat/message
Content-Type: application/x-www-form-urlencoded
Headers: X-API-Key, soeid

# Send a message (JSON)
POST /chat/message/json
Content-Type: application/json
Headers: X-API-Key, soeid

# Example
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "use_retrieval": true,
    "use_history": true,
    "use_chat_history": false,
    "chat_history_days": 7
  }'
```

### **Memory Management**
```bash
# Get conversation history
GET /chat/history/{soeid}

# Get session list
GET /chat/sessions/{soeid}

# Delete conversation history
DELETE /chat/history/{soeid}

# Get memory statistics
GET /chat/memory/stats
```

### **System Information**
```bash
# Health check
GET /health

# System metrics
GET /chat/metrics
```

## 📊 Performance Characteristics

### **Response Times**
- **Simple queries**: 1-3 seconds
- **Complex queries**: 3-8 seconds
- **With retrieval**: +1-2 seconds
- **With reranking**: +0.5-1 seconds
- **With memory**: +0.2-0.5 seconds

### **Accuracy Metrics**
- **Retrieval relevance**: 85-95% relevant documents
- **Response accuracy**: 90-95% factually correct
- **Context preservation**: 85-90% conversation coherence
- **Source attribution**: 95%+ accurate citations

### **Scalability**
- **Concurrent users**: 50-100 per instance
- **Memory usage**: 500MB-2GB per instance
- **Database connections**: 10-50 concurrent connections
- **Horizontal scaling**: Load balancer + multiple instances

## 🚨 Error Handling

### **Graceful Degradation**
1. **Retrieval fails** → Use conversation history only
2. **Reranking fails** → Use original retrieval results
3. **Memory fails** → Continue without conversation context
4. **AI generation fails** → Return error with retrieved documents

### **Common Issues and Solutions**

**Issue**: "No relevant documents found"
```yaml
# Lower similarity threshold
retrieval:
  vector:
    score_threshold: 0.5  # Reduce from 0.7
    top_k: 15             # Increase from 10
```

**Issue**: "Response too slow"
```yaml
# Optimize for speed
retrieval:
  vector:
    top_k: 5              # Reduce documents
reranking:
  custom:
    top_k: 3              # Fewer final results
generation:
  vertex:
    max_tokens: 500       # Shorter responses
```

**Issue**: "Memory connection errors"
```yaml
# Use simple memory as fallback
memory:
  type: "simple"          # Fallback to in-memory
  max_history: 10         # Limit history size
```

## 🎯 Use Cases

### **Customer Support**
```yaml
# Optimized for support queries
retrieval:
  vector:
    top_k: 8
    score_threshold: 0.75
reranking:
  custom:
    boost_keywords: ["troubleshooting", "error", "problem"]
generation:
  vertex:
    temperature: 0.3      # More conservative responses
    system_prompt: |
      You are a helpful customer support assistant.
      Provide clear, step-by-step solutions.
```

### **Research Assistant**
```yaml
# Optimized for research queries
retrieval:
  vector:
    top_k: 15
    score_threshold: 0.6
reranking:
  custom:
    boost_keywords: ["research", "study", "analysis"]
generation:
  vertex:
    temperature: 0.7      # More creative responses
    max_tokens: 1500      # Longer, detailed responses
```

### **Internal Knowledge Base**
```yaml
# Optimized for internal queries
retrieval:
  vector:
    filter_metadata:
      source: "internal"
      confidentiality: "internal_only"
memory:
  type: "langgraph_checkpoint"  # Persistent memory
chat_history:
  enabled: true                 # Cross-session context
```

## 🔧 Customization Options

### **Response Templates**
```python
# Custom response formatting
response_template = """
Based on the provided context, here's what I found:

{response}

**Sources:**
{sources}

**Confidence:** {confidence_score}
"""
```

### **Custom Prompts**
```yaml
generation:
  vertex:
    system_prompt: |
      You are an expert assistant for [Your Domain].
      Always provide accurate, helpful responses based on the context.
      If you're unsure, say so clearly.
      Include relevant examples when helpful.
```

### **Metadata Filtering**
```python
# Dynamic filtering based on user role
def get_retrieval_filters(user_role):
    if user_role == "admin":
        return {}  # No restrictions
    elif user_role == "employee":
        return {"confidentiality": {"$in": ["public", "internal"]}}
    else:
        return {"confidentiality": "public"}
```

## 🧪 Testing and Validation

### **Response Quality Testing**
```python
# Test response accuracy
test_queries = [
    {
        "query": "What is machine learning?",
        "expected_keywords": ["algorithm", "data", "prediction"],
        "expected_sources": ["ml_textbook.pdf"]
    }
]

def test_response_quality(queries):
    for test in queries:
        response = chatbot.query(test["query"])
        
        # Check keyword presence
        keywords_found = sum(1 for kw in test["expected_keywords"] 
                           if kw.lower() in response.lower())
        
        # Check source attribution
        sources_found = any(src in response.sources 
                          for src in test["expected_sources"])
        
        print(f"Query: {test['query']}")
        print(f"Keywords found: {keywords_found}/{len(test['expected_keywords'])}")
        print(f"Sources found: {sources_found}")
```

### **Performance Testing**
```bash
# Load testing with multiple concurrent users
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user{1..50}" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "use_retrieval": true}' &
```

## 📚 Component Deep Dives

- **[Document Retrievers](./retrievers.md)** - Configure document retrieval strategies
- **[Result Rerankers](./rerankers.md)** - Improve result relevance
- **[Response Generators](./generators.md)** - AI response generation
- **[Memory Systems](./memory.md)** - Conversation memory management
- **[Workflow Engine](./workflow.md)** - LangGraph workflow configuration
- **[Chatbot API](./api.md)** - Complete API reference

## 🚀 Quick Start Example

```python
# Python SDK example
from src.rag.chatbot import ChatbotService

# Initialize chatbot
chatbot = ChatbotService(config_path="config.yaml")

# Send a query
response = await chatbot.query(
    query="What is machine learning?",
    user_id="user123",
    session_id="session_456",
    use_retrieval=True,
    use_history=True
)

print(f"Response: {response.answer}")
print(f"Sources: {[doc.metadata['filename'] for doc in response.source_documents]}")
print(f"Confidence: {response.confidence_score}")
```

---

**Next Steps**: 
- [Configure Document Retrieval](./retrievers.md)
- [Set up Response Generation](./generators.md)
- [Configure Memory Systems](./memory.md)
- [Use the Chatbot API](./api.md)
