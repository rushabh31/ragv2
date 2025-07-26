# Chatbot API

## 🎯 Overview

The Chatbot API provides REST endpoints for interacting with the RAG (Retrieval-Augmented Generation) system. It offers comprehensive chat functionality including document retrieval, conversation memory, cross-session chat history, and session management. The API is built with FastAPI and provides both form-based and JSON endpoints for maximum flexibility.

## 🚀 Quick Start

### **Start the Chatbot Server**
```bash
# Navigate to chatbot directory
cd examples/rag/chatbot

# Start the server
python api/main.py

# Server starts on http://localhost:8001
```

### **Basic Chat Request**
```bash
# Simple chat request
curl -X POST "http://localhost:8001/chat/message" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -F "query=What is machine learning?" \
  -F "use_retrieval=true"
```

## 🔑 Authentication

### **API Key Authentication**
All endpoints require an API key in the request header:

```bash
# Required header
-H "X-API-Key: your-api-key"
```

### **User Identification (SOEID)**
For user-specific features like chat history, include the SOEID header:

```bash
# Optional but recommended header
-H "soeid: john.doe"
```

### **Configuration**
```yaml
# In config.yaml
api:
  authentication:
    api_keys:
      - "test-api-key"
      - "production-api-key"
    require_soeid: false          # Make SOEID optional/required
    rate_limiting:
      enabled: true
      requests_per_minute: 60
```

## 📋 Core Chat Endpoints

### **1. Form-Based Chat Message**
Send a chat message using form data (multipart/form-data).

**Endpoint:** `POST /chat/message`

**Headers:**
```
X-API-Key: your-api-key
soeid: user-identifier (optional)
Content-Type: application/x-www-form-urlencoded
```

**Form Parameters:**
```
query: string (required)              # User's question/message
session_id: string (optional)        # Session identifier
use_retrieval: boolean (default: true) # Enable document retrieval
use_history: boolean (default: true)  # Use session conversation history
use_chat_history: boolean (default: false) # Use cross-session chat history
chat_history_days: integer (default: 7) # Days to look back for chat history
metadata: JSON string (optional)     # Additional metadata
```

**Example:**
```bash
curl -X POST "http://localhost:8001/chat/message" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -F "query=Explain neural networks" \
  -F "session_id=session_123" \
  -F "use_retrieval=true" \
  -F "use_history=true" \
  -F "use_chat_history=true" \
  -F "chat_history_days=14" \
  -F 'metadata={"priority": "high", "department": "AI"}'
```

### **2. JSON Chat Message**
Send a chat message using JSON payload.

**Endpoint:** `POST /chat/message/json`

**Headers:**
```
X-API-Key: your-api-key
soeid: user-identifier (optional)
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "session_id": "session_123",
  "use_retrieval": true,
  "use_history": true,
  "use_chat_history": false,
  "chat_history_days": 7,
  "metadata": {
    "priority": "high",
    "department": "AI",
    "source": "web_app"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain neural networks",
    "use_retrieval": true,
    "use_history": true,
    "use_chat_history": true,
    "chat_history_days": 14
  }'
```

### **Response Format**
Both endpoints return the same response format:

```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "session_id": "session_123",
  "metadata": {
    "retrieved_documents": 5,
    "reranked_documents": 3,
    "generation_successful": true,
    "used_chat_history": true,
    "used_conversation_history": true,
    "processing_time_seconds": 2.34,
    "model_used": "gemini-1.5-pro-002",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "sources": [
    {
      "title": "Introduction to Machine Learning",
      "content": "Machine learning algorithms enable computers...",
      "similarity_score": 0.95,
      "source": "ml_textbook.pdf",
      "page": 15
    }
  ]
}
```

## 📚 History and Memory Endpoints

### **1. Get Chat History by SOEID**
Retrieve all chat history for a specific user across all sessions.

**Endpoint:** `GET /chat/history/{soeid}`

**Parameters:**
- `soeid` (path): User identifier
- `limit` (query, optional): Maximum number of messages (default: 50)
- `days` (query, optional): Days to look back (default: 30)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/john.doe?limit=20&days=7" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "total_messages": 45,
  "sessions": [
    {
      "session_id": "session_123",
      "message_count": 12,
      "last_activity": "2024-01-15T10:30:00Z",
      "messages": [
        {
          "query": "What is machine learning?",
          "response": "Machine learning is a subset of AI...",
          "timestamp": "2024-01-15T10:25:00Z",
          "metadata": {
            "retrieved_documents": 5,
            "model_used": "gemini-1.5-pro-002"
          }
        }
      ]
    }
  ]
}
```

### **2. Get Session History**
Retrieve conversation history for a specific session.

**Endpoint:** `GET /chat/history/session/{session_id}`

**Parameters:**
- `session_id` (path): Session identifier
- `limit` (query, optional): Maximum number of messages (default: 50)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/session/session_123?limit=10" \
  -H "X-API-Key: test-api-key"
```

### **3. Get Sessions for SOEID**
Get metadata about all sessions for a specific user.

**Endpoint:** `GET /chat/sessions/{soeid}`

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/sessions/john.doe" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "total_sessions": 8,
  "sessions": [
    {
      "session_id": "session_123",
      "message_count": 12,
      "created_at": "2024-01-15T09:00:00Z",
      "last_activity": "2024-01-15T10:30:00Z",
      "metadata": {
        "device": "web",
        "location": "office"
      }
    }
  ]
}
```

### **4. Get All Threads**
List all conversation threads (sessions) in the system.

**Endpoint:** `GET /chat/threads`

**Parameters:**
- `limit` (query, optional): Maximum number of threads (default: 100)
- `offset` (query, optional): Pagination offset (default: 0)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/threads?limit=50&offset=0" \
  -H "X-API-Key: test-api-key"
```

## 🗑️ Management Endpoints

### **1. Delete User History**
Delete all chat history for a specific user.

**Endpoint:** `DELETE /chat/history/{soeid}`

**Example:**
```bash
curl -X DELETE "http://localhost:8001/chat/history/john.doe" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "message": "Successfully deleted chat history for user john.doe",
  "deleted_sessions": 8,
  "deleted_messages": 156
}
```

### **2. Delete Session History**
Delete conversation history for a specific session.

**Endpoint:** `DELETE /chat/history/session/{session_id}`

**Example:**
```bash
curl -X DELETE "http://localhost:8001/chat/history/session/session_123" \
  -H "X-API-Key: test-api-key"
```

### **3. Get Memory Statistics**
Get comprehensive statistics about the memory system.

**Endpoint:** `GET /chat/memory/stats`

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/memory/stats" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "total_sessions": 1250,
  "total_messages": 15680,
  "unique_users": 89,
  "active_sessions_24h": 45,
  "average_messages_per_session": 12.5,
  "oldest_session": "2024-01-01T00:00:00Z",
  "newest_session": "2024-01-15T10:30:00Z",
  "memory_usage": {
    "database_size_mb": 245.7,
    "index_size_mb": 12.3,
    "total_size_mb": 258.0
  }
}
```

## 🔧 System Information Endpoints

### **1. Health Check**
Check if the chatbot service is running and healthy.

**Endpoint:** `GET /health`

**Example:**
```bash
curl -X GET "http://localhost:8001/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "retriever": "healthy",
    "reranker": "healthy",
    "generator": "healthy",
    "memory": "healthy",
    "vector_store": "healthy"
  },
  "uptime_seconds": 3600
}
```

### **2. Service Information**
Get detailed information about the chatbot service configuration.

**Endpoint:** `GET /info`

**Example:**
```bash
curl -X GET "http://localhost:8001/info" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "service": "RAG Chatbot API",
  "version": "1.0.0",
  "configuration": {
    "retriever": {
      "type": "vector_retriever",
      "vector_store": "faiss",
      "embedding_model": "text-embedding-004"
    },
    "generator": {
      "provider": "vertex",
      "model": "gemini-1.5-pro-002"
    },
    "memory": {
      "type": "langgraph_checkpoint",
      "store_type": "postgres"
    }
  },
  "features": {
    "document_retrieval": true,
    "conversation_memory": true,
    "cross_session_history": true,
    "reranking": true,
    "streaming": false
  }
}
```

## 🐛 Debug Endpoints

### **1. Get All Messages (Debug)**
Get all messages in the system for debugging purposes.

**Endpoint:** `GET /chat/debug/all-messages`

**Parameters:**
- `limit` (query, optional): Maximum number of messages (default: 100)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/debug/all-messages?limit=20" \
  -H "X-API-Key: test-api-key"
```

**Note:** This endpoint should only be used for debugging and may be disabled in production.

## 📊 Error Handling

### **Error Response Format**
All endpoints return errors in a consistent format:

```json
{
  "error": "Invalid API key",
  "error_code": "AUTHENTICATION_FAILED",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### **Common Error Codes**

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid or missing API key |
| `INVALID_REQUEST` | 400 | Malformed request or missing required fields |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests within time window |
| `SESSION_NOT_FOUND` | 404 | Specified session does not exist |
| `USER_NOT_FOUND` | 404 | Specified user has no chat history |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### **Error Examples**

**Authentication Error:**
```bash
curl -X POST "http://localhost:8001/chat/message" \
  -F "query=Hello"
# Response: 401 - Missing API key
```

**Invalid Request:**
```bash
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{}'
# Response: 400 - Missing required field: query
```

**Rate Limit Exceeded:**
```bash
# After making too many requests
# Response: 429 - Rate limit exceeded. Try again in 60 seconds.
```

## 🔧 Advanced Usage

### **Using Chat History Across Sessions**
```bash
# First conversation
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -d '{"query": "What is machine learning?", "session_id": "session_1"}'

# Later conversation in different session
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -d '{
    "query": "Can you give me examples of what we discussed?",
    "session_id": "session_2",
    "use_chat_history": true,
    "chat_history_days": 7
  }'
```

### **Metadata Usage**
```bash
# Include custom metadata
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -d '{
    "query": "Explain neural networks",
    "metadata": {
      "department": "Engineering",
      "priority": "high",
      "source": "slack_bot",
      "user_role": "developer",
      "project_id": "proj_123"
    }
  }'
```

### **Session Management**
```bash
# Create a new session with specific ID
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -d '{
    "query": "Start new conversation about AI",
    "session_id": "ai_discussion_2024_01_15"
  }'

# Continue the same session
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -d '{
    "query": "Tell me more about neural networks",
    "session_id": "ai_discussion_2024_01_15"
  }'
```

## 🐍 Python SDK Usage

### **Basic Python Client**
```python
import requests
import json

class ChatbotClient:
    def __init__(self, base_url="http://localhost:8001", api_key="test-api-key"):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
    
    def chat(self, query, soeid=None, session_id=None, use_retrieval=True, 
             use_history=True, use_chat_history=False, chat_history_days=7, 
             metadata=None):
        """Send a chat message."""
        
        headers = {}
        if soeid:
            headers["soeid"] = soeid
        
        payload = {
            "query": query,
            "use_retrieval": use_retrieval,
            "use_history": use_history,
            "use_chat_history": use_chat_history,
            "chat_history_days": chat_history_days
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        if metadata:
            payload["metadata"] = metadata
        
        response = self.session.post(
            f"{self.base_url}/chat/message/json",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_history(self, soeid, limit=50, days=30):
        """Get chat history for a user."""
        
        response = self.session.get(
            f"{self.base_url}/chat/history/{soeid}",
            params={"limit": limit, "days": days}
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_memory_stats(self):
        """Get memory system statistics."""
        
        response = self.session.get(f"{self.base_url}/chat/memory/stats")
        response.raise_for_status()
        return response.json()

# Usage example
client = ChatbotClient(api_key="your-api-key")

# Send a message
response = client.chat(
    query="What is machine learning?",
    soeid="john.doe",
    use_retrieval=True,
    use_chat_history=True,
    metadata={"department": "AI"}
)

print(f"Response: {response['response']}")
print(f"Sources: {len(response['sources'])} documents")

# Get user's chat history
history = client.get_history("john.doe", limit=10)
print(f"User has {history['total_messages']} total messages")
```

### **Async Python Client**
```python
import aiohttp
import asyncio

class AsyncChatbotClient:
    def __init__(self, base_url="http://localhost:8001", api_key="test-api-key"):
        self.base_url = base_url
        self.api_key = api_key
    
    async def chat(self, query, soeid=None, **kwargs):
        """Send a chat message asynchronously."""
        
        headers = {"X-API-Key": self.api_key}
        if soeid:
            headers["soeid"] = soeid
        
        payload = {"query": query, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/message/json",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

# Usage example
async def main():
    client = AsyncChatbotClient(api_key="your-api-key")
    
    # Send multiple messages concurrently
    tasks = [
        client.chat("What is AI?", soeid="user1"),
        client.chat("Explain neural networks", soeid="user2"),
        client.chat("What is deep learning?", soeid="user3")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses, 1):
        print(f"Response {i}: {response['response'][:100]}...")

# Run async example
asyncio.run(main())
```

## 📚 Related Documentation

- **[Chatbot Service Overview](./README.md)** - Complete chatbot architecture
- **[Memory Systems](./memory.md)** - Configure conversation memory
- **[Workflow Management](./workflow.md)** - Understand the processing workflow
- **[Response Generators](./generators.md)** - Configure AI response generation
- **[Document Retrievers](./retrievers.md)** - Set up document retrieval

## 🚀 Quick Reference

### **Essential Endpoints**
```bash
# Send a message
POST /chat/message/json

# Get user history
GET /chat/history/{soeid}

# Get memory stats
GET /chat/memory/stats

# Health check
GET /health
```

### **Required Headers**
```bash
X-API-Key: your-api-key    # Always required
soeid: user-identifier     # Required for user-specific features
```

### **Key Parameters**
```bash
use_retrieval: true        # Enable document search
use_history: true          # Use session conversation history
use_chat_history: true     # Use cross-session chat history
chat_history_days: 7       # Days to look back for chat history
```

---

**Next Steps**: 
- [Set up Memory Systems](./memory.md)
- [Configure Response Generators](./generators.md)
- [Understand the Workflow](./workflow.md)
