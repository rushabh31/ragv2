# Memory API

## 🎯 Overview

The Memory API provides comprehensive endpoints for managing conversation history, chat history, and memory system operations. It enables programmatic access to all memory functions including session management, user history retrieval, cross-session chat history, and system maintenance operations.

## 🚀 Quick Start

### **Base URL**
```
http://localhost:8001  # Default chatbot server
```

### **Authentication**
All memory endpoints require API key authentication:
```bash
-H "X-API-Key: your-api-key"
```

### **Basic Memory Operation**
```bash
# Get user's chat history
curl -X GET "http://localhost:8001/chat/history/john.doe" \
  -H "X-API-Key: test-api-key"
```

## 📋 Core Memory Endpoints

### **1. Get Chat History by SOEID**
Retrieve all chat history for a specific user across all sessions.

**Endpoint:** `GET /chat/history/{soeid}`

**Parameters:**
- `soeid` (path): User identifier (SOEID)
- `limit` (query, optional): Maximum number of messages (default: 50, max: 200)
- `days` (query, optional): Days to look back (default: 30, max: 365)
- `include_metadata` (query, optional): Include message metadata (default: true)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/john.doe?limit=20&days=7&include_metadata=true" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "total_messages": 45,
  "date_range": {
    "from": "2024-01-08T10:30:00Z",
    "to": "2024-01-15T10:30:00Z",
    "days": 7
  },
  "sessions": [
    {
      "session_id": "session_123",
      "message_count": 12,
      "last_activity": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-15T09:00:00Z",
      "messages": [
        {
          "query": "What is machine learning?",
          "response": "Machine learning is a subset of artificial intelligence...",
          "timestamp": "2024-01-15T10:25:00Z",
          "metadata": {
            "model_used": "gemini-1.5-pro-002",
            "retrieved_documents": 5,
            "processing_time": 2.34,
            "source_documents": ["ml_textbook.pdf"]
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
- `include_metadata` (query, optional): Include message metadata (default: true)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/session/session_123?limit=10" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "session_id": "session_123",
  "message_count": 12,
  "created_at": "2024-01-15T09:00:00Z",
  "last_activity": "2024-01-15T10:30:00Z",
  "messages": [
    {
      "query": "What is machine learning?",
      "response": "Machine learning is a subset of AI...",
      "timestamp": "2024-01-15T10:25:00Z",
      "metadata": {
        "model_used": "gemini-1.5-pro-002",
        "retrieved_documents": 5
      }
    }
  ]
}
```

### **3. Get Sessions for SOEID**
Get metadata about all sessions for a specific user.

**Endpoint:** `GET /chat/sessions/{soeid}`

**Parameters:**
- `soeid` (path): User identifier
- `limit` (query, optional): Maximum number of sessions (default: 50)
- `include_empty` (query, optional): Include sessions with no messages (default: false)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/sessions/john.doe?limit=20" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "total_sessions": 8,
  "active_sessions": 3,
  "sessions": [
    {
      "session_id": "session_123",
      "message_count": 12,
      "created_at": "2024-01-15T09:00:00Z",
      "last_activity": "2024-01-15T10:30:00Z",
      "duration_minutes": 90,
      "metadata": {
        "device": "web",
        "location": "office",
        "topics": ["machine learning", "AI"]
      }
    }
  ]
}
```

### **4. Get Alternative History Endpoint**
Alternative endpoint for getting session history with additional filtering.

**Endpoint:** `GET /chat/sessions/{soeid}/history`

**Parameters:**
- `soeid` (path): User identifier
- `limit` (query, optional): Maximum number of messages (default: 50)
- `session_filter` (query, optional): Filter by specific sessions (comma-separated)
- `topic_filter` (query, optional): Filter by topic keywords

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/sessions/john.doe/history?limit=30&topic_filter=machine learning" \
  -H "X-API-Key: test-api-key"
```

## 🗑️ Memory Management Endpoints

### **1. Delete User History**
Delete all chat history for a specific user.

**Endpoint:** `DELETE /chat/history/{soeid}`

**Parameters:**
- `soeid` (path): User identifier
- `confirm` (query, required): Must be "true" to confirm deletion

**Example:**
```bash
curl -X DELETE "http://localhost:8001/chat/history/john.doe?confirm=true" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "message": "Successfully deleted chat history for user john.doe",
  "deleted_sessions": 8,
  "deleted_messages": 156,
  "deletion_timestamp": "2024-01-15T10:30:00Z"
}
```

### **2. Delete Session History**
Delete conversation history for a specific session.

**Endpoint:** `DELETE /chat/history/session/{session_id}`

**Parameters:**
- `session_id` (path): Session identifier
- `confirm` (query, required): Must be "true" to confirm deletion

**Example:**
```bash
curl -X DELETE "http://localhost:8001/chat/history/session/session_123?confirm=true" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "message": "Successfully deleted session session_123",
  "deleted_messages": 12,
  "session_id": "session_123",
  "deletion_timestamp": "2024-01-15T10:30:00Z"
}
```

### **3. Archive User History**
Archive user history instead of deleting (if archival is enabled).

**Endpoint:** `POST /chat/history/{soeid}/archive`

**Parameters:**
- `soeid` (path): User identifier
- `archive_reason` (body): Reason for archival

**Request Body:**
```json
{
  "archive_reason": "User requested data export",
  "include_metadata": true,
  "compression": "gzip"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/chat/history/john.doe/archive" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_reason": "User requested data export",
    "include_metadata": true
  }'
```

## 📊 Memory Statistics Endpoints

### **1. Get Memory Statistics**
Get comprehensive statistics about the memory system.

**Endpoint:** `GET /chat/memory/stats`

**Parameters:**
- `detailed` (query, optional): Include detailed breakdown (default: false)
- `time_range` (query, optional): Time range for statistics (1d, 7d, 30d, default: 30d)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/memory/stats?detailed=true&time_range=7d" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "overview": {
    "total_sessions": 1250,
    "total_messages": 15680,
    "unique_users": 89,
    "active_sessions_24h": 45,
    "average_messages_per_session": 12.5
  },
  "temporal": {
    "oldest_session": "2024-01-01T00:00:00Z",
    "newest_session": "2024-01-15T10:30:00Z",
    "messages_last_24h": 234,
    "messages_last_7d": 1456,
    "sessions_last_24h": 23,
    "sessions_last_7d": 156
  },
  "storage": {
    "database_size_mb": 245.7,
    "index_size_mb": 12.3,
    "total_size_mb": 258.0,
    "growth_rate_mb_per_day": 8.5
  },
  "performance": {
    "average_query_time_ms": 45.2,
    "cache_hit_rate": 0.85,
    "slow_queries_24h": 3
  },
  "user_activity": {
    "most_active_users": [
      {"soeid": "user1", "sessions": 15, "messages": 234},
      {"soeid": "user2", "sessions": 12, "messages": 189}
    ],
    "session_length_distribution": {
      "1-5_messages": 45,
      "6-10_messages": 32,
      "11-20_messages": 18,
      "20+_messages": 5
    }
  }
}
```

### **2. Get User Activity Statistics**
Get detailed activity statistics for a specific user.

**Endpoint:** `GET /chat/memory/stats/user/{soeid}`

**Parameters:**
- `soeid` (path): User identifier
- `days` (query, optional): Days to analyze (default: 30)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/memory/stats/user/john.doe?days=14" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "analysis_period": {
    "days": 14,
    "from": "2024-01-01T10:30:00Z",
    "to": "2024-01-15T10:30:00Z"
  },
  "activity": {
    "total_sessions": 8,
    "total_messages": 156,
    "average_messages_per_session": 19.5,
    "most_active_day": "2024-01-10",
    "messages_by_day": {
      "2024-01-15": 23,
      "2024-01-14": 18,
      "2024-01-13": 0
    }
  },
  "topics": {
    "most_discussed": ["machine learning", "AI", "data science"],
    "topic_distribution": {
      "machine_learning": 45,
      "artificial_intelligence": 32,
      "data_science": 28
    }
  },
  "patterns": {
    "preferred_session_length": "medium",
    "most_active_hours": [9, 10, 14, 15],
    "conversation_style": "technical"
  }
}
```

## 🔧 System Management Endpoints

### **1. Get All Threads**
List all conversation threads (sessions) in the system.

**Endpoint:** `GET /chat/threads`

**Parameters:**
- `limit` (query, optional): Maximum number of threads (default: 100, max: 500)
- `offset` (query, optional): Pagination offset (default: 0)
- `sort_by` (query, optional): Sort field (created_at, last_activity, message_count)
- `sort_order` (query, optional): Sort order (asc, desc, default: desc)
- `filter_active` (query, optional): Filter active threads only (default: false)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/threads?limit=50&offset=0&sort_by=last_activity&filter_active=true" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "total_threads": 1250,
  "returned_count": 50,
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_next": true,
    "has_previous": false,
    "total_pages": 25
  },
  "threads": [
    {
      "thread_id": "session_123",
      "soeid": "john.doe",
      "message_count": 12,
      "created_at": "2024-01-15T09:00:00Z",
      "last_activity": "2024-01-15T10:30:00Z",
      "is_active": true,
      "metadata": {
        "device": "web",
        "topics": ["machine learning"]
      }
    }
  ]
}
```

### **2. Get Thread History**
Get conversation history for a specific thread.

**Endpoint:** `GET /chat/threads/{thread_id}/history`

**Parameters:**
- `thread_id` (path): Thread identifier
- `limit` (query, optional): Maximum number of messages (default: 50)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/threads/session_123/history?limit=20" \
  -H "X-API-Key: test-api-key"
```

### **3. Memory Health Check**
Check the health status of the memory system.

**Endpoint:** `GET /chat/memory/health`

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/memory/health" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "connection_pool": "normal",
      "query_performance": "good",
      "last_backup": "2024-01-15T02:00:00Z"
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.85,
      "memory_usage": "normal"
    },
    "storage": {
      "status": "healthy",
      "disk_usage": "75%",
      "growth_rate": "normal"
    }
  },
  "alerts": [],
  "recommendations": [
    "Consider increasing cache size for better performance"
  ]
}
```

## 🐛 Debug Endpoints

### **1. Get All Messages (Debug)**
Get all messages in the system for debugging purposes.

**Endpoint:** `GET /chat/debug/all-messages`

**Parameters:**
- `limit` (query, optional): Maximum number of messages (default: 100, max: 1000)
- `include_system` (query, optional): Include system messages (default: false)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/debug/all-messages?limit=20&include_system=false" \
  -H "X-API-Key: test-api-key"
```

**Note:** This endpoint should only be used for debugging and may be disabled in production.

### **2. Memory Debug Info**
Get detailed debug information about memory system state.

**Endpoint:** `GET /chat/debug/memory-info`

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/debug/memory-info" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "memory_type": "langgraph_checkpoint",
  "store_type": "postgres",
  "configuration": {
    "pool_size": 20,
    "max_overflow": 30,
    "cache_enabled": true
  },
  "runtime_stats": {
    "active_connections": 8,
    "pool_utilization": 0.4,
    "cache_size": "125MB",
    "uptime_seconds": 86400
  },
  "recent_activity": {
    "queries_last_hour": 234,
    "errors_last_hour": 2,
    "slow_queries_last_hour": 1
  }
}
```

## 📊 Advanced Query Endpoints

### **1. Search Chat History**
Search chat history using text queries or filters.

**Endpoint:** `POST /chat/history/search`

**Request Body:**
```json
{
  "query": "machine learning algorithms",
  "soeid": "john.doe",
  "filters": {
    "date_from": "2024-01-01T00:00:00Z",
    "date_to": "2024-01-15T23:59:59Z",
    "min_relevance": 0.3,
    "include_responses": true
  },
  "limit": 20
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/chat/history/search" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "soeid": "john.doe",
    "filters": {
      "date_from": "2024-01-01T00:00:00Z",
      "min_relevance": 0.3
    },
    "limit": 10
  }'
```

### **2. Get Topic-Based History**
Get chat history filtered by specific topics.

**Endpoint:** `GET /chat/history/{soeid}/topics/{topic}`

**Parameters:**
- `soeid` (path): User identifier
- `topic` (path): Topic name or keyword
- `days` (query, optional): Days to look back (default: 30)
- `limit` (query, optional): Maximum number of messages (default: 50)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/john.doe/topics/machine_learning?days=14&limit=20" \
  -H "X-API-Key: test-api-key"
```

### **3. Get Conversation Threads**
Get reconstructed conversation threads for a user.

**Endpoint:** `GET /chat/history/{soeid}/threads`

**Parameters:**
- `soeid` (path): User identifier
- `days` (query, optional): Days to analyze (default: 7)
- `min_messages` (query, optional): Minimum messages per thread (default: 2)

**Example:**
```bash
curl -X GET "http://localhost:8001/chat/history/john.doe/threads?days=14&min_messages=3" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "soeid": "john.doe",
  "analysis_period": {
    "days": 14,
    "from": "2024-01-01T10:30:00Z",
    "to": "2024-01-15T10:30:00Z"
  },
  "threads": [
    {
      "session_id": "session_123",
      "message_count": 12,
      "start_time": "2024-01-15T09:00:00Z",
      "end_time": "2024-01-15T10:30:00Z",
      "duration_minutes": 90,
      "topics": ["machine learning", "neural networks"],
      "summary": "Discussion about machine learning fundamentals and neural network architectures"
    }
  ]
}
```

## 🔒 Authentication and Authorization

### **API Key Management**
```yaml
# Configuration for API keys
api:
  authentication:
    api_keys:
      - key: "prod-api-key-123"
        name: "Production App"
        permissions: ["read", "write", "delete"]
        rate_limit: 1000
      - key: "readonly-key-456"
        name: "Analytics Dashboard"
        permissions: ["read"]
        rate_limit: 500
      - key: "admin-key-789"
        name: "Admin Tools"
        permissions: ["read", "write", "delete", "admin"]
        rate_limit: 2000
```

### **Permission Levels**
- **read**: Access to GET endpoints
- **write**: Access to POST/PUT endpoints
- **delete**: Access to DELETE endpoints
- **admin**: Access to debug and system management endpoints

## 📈 Rate Limiting

### **Rate Limit Headers**
All responses include rate limiting information:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
X-RateLimit-Window: 3600
```

### **Rate Limit Configuration**
```yaml
api:
  rate_limiting:
    enabled: true
    default_limit: 1000          # Requests per hour
    burst_limit: 100             # Burst requests
    memory_endpoints:
      get_history: 100           # Specific limit for history endpoints
      delete_history: 10         # Lower limit for destructive operations
      search_history: 50         # Limit for search operations
```

## 🚨 Error Handling

### **Error Response Format**
```json
{
  "error": "User not found",
  "error_code": "USER_NOT_FOUND",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789",
  "details": {
    "soeid": "nonexistent.user",
    "searched_days": 30
  }
}
```

### **Common Error Codes**

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid or missing API key |
| `INSUFFICIENT_PERMISSIONS` | 403 | API key lacks required permissions |
| `USER_NOT_FOUND` | 404 | User has no chat history |
| `SESSION_NOT_FOUND` | 404 | Session does not exist |
| `INVALID_DATE_RANGE` | 400 | Invalid date parameters |
| `LIMIT_EXCEEDED` | 400 | Requested limit exceeds maximum |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `MEMORY_SYSTEM_ERROR` | 500 | Internal memory system error |
| `DATABASE_ERROR` | 503 | Database connection issues |

## 🐍 Python SDK

### **Memory API Client**
```python
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MemoryAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001", api_key: str = "test-api-key"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
    
    def get_chat_history(self, soeid: str, days: int = 30, limit: int = 50, 
                        include_metadata: bool = True) -> Dict:
        """Get chat history for a user."""
        
        response = self.session.get(
            f"{self.base_url}/chat/history/{soeid}",
            params={
                "days": days,
                "limit": limit,
                "include_metadata": include_metadata
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_session_history(self, session_id: str, limit: int = 50) -> Dict:
        """Get history for a specific session."""
        
        response = self.session.get(
            f"{self.base_url}/chat/history/session/{session_id}",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def get_user_sessions(self, soeid: str, limit: int = 50) -> Dict:
        """Get all sessions for a user."""
        
        response = self.session.get(
            f"{self.base_url}/chat/sessions/{soeid}",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_user_history(self, soeid: str) -> Dict:
        """Delete all history for a user."""
        
        response = self.session.delete(
            f"{self.base_url}/chat/history/{soeid}",
            params={"confirm": "true"}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str) -> Dict:
        """Delete a specific session."""
        
        response = self.session.delete(
            f"{self.base_url}/chat/history/session/{session_id}",
            params={"confirm": "true"}
        )
        response.raise_for_status()
        return response.json()
    
    def get_memory_stats(self, detailed: bool = False, time_range: str = "30d") -> Dict:
        """Get memory system statistics."""
        
        response = self.session.get(
            f"{self.base_url}/chat/memory/stats",
            params={
                "detailed": detailed,
                "time_range": time_range
            }
        )
        response.raise_for_status()
        return response.json()
    
    def search_chat_history(self, query: str, soeid: str, 
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None,
                           limit: int = 20) -> Dict:
        """Search chat history with text query."""
        
        search_payload = {
            "query": query,
            "soeid": soeid,
            "limit": limit,
            "filters": {}
        }
        
        if date_from:
            search_payload["filters"]["date_from"] = date_from.isoformat()
        if date_to:
            search_payload["filters"]["date_to"] = date_to.isoformat()
        
        response = self.session.post(
            f"{self.base_url}/chat/history/search",
            json=search_payload
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = MemoryAPIClient(api_key="your-api-key")

# Get user's recent chat history
history = client.get_chat_history("john.doe", days=7, limit=20)
print(f"Found {history['total_messages']} messages")

# Search for specific topics
search_results = client.search_chat_history(
    query="machine learning",
    soeid="john.doe",
    date_from=datetime.now() - timedelta(days=14)
)

# Get memory statistics
stats = client.get_memory_stats(detailed=True)
print(f"Total users: {stats['overview']['unique_users']}")
```

## 📚 Related Documentation

- **[Memory Systems Overview](./README.md)** - Complete memory architecture
- **[Chat History](./chat-history.md)** - Chat history implementation
- **[PostgreSQL Setup](./postgresql.md)** - Database configuration
- **[Chatbot API](../rag/chatbot/api.md)** - Main chatbot API

## 🚀 Quick Reference

### **Essential Endpoints**
```bash
# Get user history
GET /chat/history/{soeid}

# Get session history  
GET /chat/history/session/{session_id}

# Get memory stats
GET /chat/memory/stats

# Delete user history
DELETE /chat/history/{soeid}?confirm=true
```

### **Required Headers**
```bash
X-API-Key: your-api-key    # Always required
```

### **Common Parameters**
```bash
limit: 50                  # Limit results
days: 30                   # Days to look back
include_metadata: true     # Include message metadata
```

---

**Next Steps**: 
- [Set up Memory Systems](./README.md)
- [Configure PostgreSQL](./postgresql.md)
- [Use Chat History](./chat-history.md)
