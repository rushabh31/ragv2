# LangGraph Memory Implementation

This document describes the comprehensive LangGraph memory implementation for the RAG chatbot system, including both in-memory and PostgreSQL storage options.

## Overview

The system now supports LangGraph checkpointers for conversation history storage and LangGraph stores for long-term memory. This provides enterprise-grade memory management with support for:

- **Thread-scoped memory**: Conversation history within sessions using LangGraph checkpointers
- **Cross-thread memory**: Long-term memory that persists across sessions using LangGraph stores
- **SOEID support**: User identification and history retrieval across all sessions
- **PostgreSQL integration**: Production-ready persistent storage with pgvector support
- **In-memory fallback**: Development and testing support

## Memory Types

### 1. LangGraph Checkpoint Memory (`langgraph_checkpoint`)

**Class**: `LangGraphCheckpointMemory`
**File**: `src/rag/chatbot/memory/langgraph_checkpoint_memory.py`

This is the recommended implementation that uses:
- **LangGraph checkpointers** for conversation history (thread-scoped)
- **LangGraph stores** for long-term memory (cross-thread)
- **Universal authentication** integration
- **SOEID-based user tracking**

#### Storage Options

##### In-Memory Storage
```yaml
memory:
  type: langgraph_checkpoint
  store_type: in_memory
```

##### PostgreSQL Storage
```yaml
memory:
  type: langgraph_checkpoint
  store_type: postgres
  postgres:
    connection_string: "postgresql://username:password@localhost:5432/langgraph_db"
```

### 2. Legacy LangGraph Memory (`langgraph`)

**Class**: `LangGraphMemory`
**File**: `src/rag/chatbot/memory/langgraph_memory.py`

Legacy implementation using only LangGraph stores (maintained for backward compatibility).

## Configuration

### Environment Variables

For PostgreSQL support, ensure these are set:

```bash
# Database connection
DATABASE_URL=postgresql://username:password@localhost:5432/langgraph_db

# Optional: Custom embedding service (if not using simple hash embeddings)
EMBEDDING_SERVICE_URL=https://your-embedding-service.com
```

### Database Setup for PostgreSQL

1. **Create Database**:
```sql
CREATE DATABASE langgraph_db;
```

2. **Enable pgvector Extension** (for vector storage):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. **Create User** (if needed):
```sql
CREATE USER langgraph_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE langgraph_db TO langgraph_user;
```

### Configuration File

Update your `config.yaml`:

```yaml
chatbot:
  memory:
    # Memory type: "simple", "mem0", "langgraph", or "langgraph_checkpoint"
    type: langgraph_checkpoint
    
    # Storage type: "in_memory" or "postgres"
    store_type: postgres  # Use "in_memory" for development
    
    # LangGraph memory settings
    embedding_dimensions: 768
    max_history: 20
    
    # PostgreSQL settings (only used if store_type is "postgres")
    postgres:
      connection_string: "postgresql://username:password@localhost:5432/langgraph_db"
      # For pgvector support, ensure the database has the pgvector extension
      # CREATE EXTENSION IF NOT EXISTS vector;
    
    # Legacy settings for backward compatibility
    max_messages: 20
    message_expiry_minutes: 60
    max_sessions: 1000
```

## API Endpoints

### Core Chat Endpoints

#### Send Message
```http
POST /chat/message
Content-Type: application/x-www-form-urlencoded
soeid: user123

query=Hello&session_id=optional&use_retrieval=true&use_history=true
```

#### Send Message (JSON)
```http
POST /chat/message/json
Content-Type: application/json
soeid: user123

{
  "query": "Hello",
  "session_id": "optional",
  "use_retrieval": true,
  "use_history": true,
  "metadata": {}
}
```

### History Retrieval Endpoints

#### Get All History by SOEID
```http
GET /chat/history/{soeid}
```
Returns all chat history for a SOEID, grouped by session.

#### Get Session History with SOEID
```http
GET /chat/history/session/{session_id}
```
Returns all chat history for a specific session, including SOEID in every message.

#### Get Sessions for SOEID
```http
GET /chat/sessions/{soeid}
```
Returns metadata for all sessions associated with a SOEID.

#### Get SOEID History by Sessions (Alternative)
```http
GET /chat/sessions/{soeid}/history?limit=10
```
Alternative endpoint to get chat history for a SOEID, organized by sessions.

### Thread Management Endpoints

#### Get All Threads
```http
GET /chat/threads
```
Returns all threads (sessions) in the system with metadata.

#### Get Thread History
```http
GET /chat/threads/{thread_id}/history?limit=20
```
Returns chat history for a specific thread (session) ID.

### Memory Management Endpoints

#### Get Memory Statistics
```http
GET /chat/memory/stats
```
Returns comprehensive memory system statistics:
```json
{
  "memory_type": "LangGraphCheckpointMemory",
  "store_type": "postgres",
  "total_sessions": 150,
  "total_messages": 1200,
  "unique_soeids": 45,
  "oldest_session": "2024-01-15T10:30:00Z",
  "newest_session": "2024-01-25T16:45:00Z",
  "metadata": {
    "retrieved_at": "2024-01-25T19:34:24Z"
  }
}
```

### Deletion Endpoints

#### Delete History by SOEID
```http
DELETE /chat/history/{soeid}
```
Deletes all chat history for a SOEID across all sessions.

#### Delete History by Session ID
```http
DELETE /chat/history/session/{session_id}
```
Deletes all chat history for a specific session.

### Debug Endpoints

#### Debug All Messages
```http
GET /chat/debug/all-messages
```
Returns all messages in all session namespaces for debugging SOEID storage.

## Usage Examples

### Python Client Example

```python
import httpx
import asyncio

async def chat_example():
    async with httpx.AsyncClient() as client:
        # Send a message
        response = await client.post(
            "http://localhost:8001/chat/message/json",
            headers={"soeid": "user123"},
            json={
                "query": "What is machine learning?",
                "use_retrieval": True,
                "use_history": True
            }
        )
        chat_result = response.json()
        session_id = chat_result["session_id"]
        
        # Get session history
        history_response = await client.get(
            f"http://localhost:8001/chat/history/session/{session_id}"
        )
        history = history_response.json()
        
        # Get all sessions for user
        sessions_response = await client.get(
            "http://localhost:8001/chat/sessions/user123"
        )
        sessions = sessions_response.json()
        
        # Get memory stats
        stats_response = await client.get(
            "http://localhost:8001/chat/memory/stats"
        )
        stats = stats_response.json()
        
        print(f"Chat response: {chat_result['response']}")
        print(f"Session has {len(history['messages'])} messages")
        print(f"User has {sessions['total_sessions']} sessions")
        print(f"System has {stats['total_messages']} total messages")

# Run the example
asyncio.run(chat_example())
```

### cURL Examples

```bash
# Send a message
curl -X POST "http://localhost:8001/chat/message" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "soeid: user123" \
  -d "query=What is AI?&use_retrieval=true"

# Get user's chat history
curl -X GET "http://localhost:8001/chat/history/user123"

# Get sessions for user
curl -X GET "http://localhost:8001/chat/sessions/user123"

# Get all threads
curl -X GET "http://localhost:8001/chat/threads"

# Get memory statistics
curl -X GET "http://localhost:8001/chat/memory/stats"

# Get thread history with limit
curl -X GET "http://localhost:8001/chat/threads/session-123/history?limit=10"

# Delete user history
curl -X DELETE "http://localhost:8001/chat/history/user123"
```

## Key Features

### 1. SOEID-Based User Tracking
- Every message is tagged with the user's SOEID
- Cross-session history retrieval by SOEID
- User-specific session management

### 2. Thread Management
- Each conversation is a separate thread/session
- Thread-scoped memory using LangGraph checkpointers
- Thread listing and metadata retrieval

### 3. Persistent Storage
- **In-Memory**: Fast, suitable for development and testing
- **PostgreSQL**: Production-ready with pgvector support for semantic search

### 4. Memory Statistics
- Real-time statistics on memory usage
- Session and message counts
- Unique user tracking
- Temporal analysis (oldest/newest sessions)

### 5. Flexible History Retrieval
- By SOEID (all sessions for a user)
- By session ID (specific conversation)
- By thread ID (LangGraph thread)
- With configurable limits and pagination

## Architecture Benefits

### 1. LangGraph Integration
- Native LangGraph checkpointer support
- Thread-scoped conversation memory
- Cross-thread long-term memory
- Standard LangGraph patterns and best practices

### 2. Scalability
- PostgreSQL backend for production workloads
- Efficient indexing and querying
- pgvector support for semantic search
- Connection pooling and optimization

### 3. Developer Experience
- Multiple endpoint options for different use cases
- Comprehensive error handling and logging
- Debug endpoints for troubleshooting
- Clear API documentation and examples

### 4. Enterprise Features
- SOEID-based user identification
- Audit trails and message tracking
- Memory statistics and monitoring
- Secure deletion and data management

## Migration from Simple Memory

To migrate from simple memory to LangGraph checkpoint memory:

1. **Update Configuration**:
```yaml
memory:
  type: langgraph_checkpoint  # Changed from "simple"
  store_type: postgres        # or "in_memory" for testing
```

2. **Set up PostgreSQL** (if using postgres store_type):
```sql
CREATE DATABASE langgraph_db;
CREATE EXTENSION IF NOT EXISTS vector;
```

3. **Update Connection String**:
```yaml
postgres:
  connection_string: "postgresql://username:password@localhost:5432/langgraph_db"
```

4. **Test Migration**:
```bash
# Test the new endpoints
curl -X GET "http://localhost:8001/chat/memory/stats"
```

The system will automatically handle the migration and provide backward compatibility for existing endpoints.

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Issues**:
   - Verify database exists and is accessible
   - Check connection string format
   - Ensure user has proper permissions

2. **Missing pgvector Extension**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Memory Type Not Found**:
   - Verify `langgraph_checkpoint` is in the memory factory
   - Check import statements in memory_factory.py

4. **Thread Listing Not Working**:
   - Ensure using `langgraph_checkpoint` memory type
   - Check if `_list_thread_ids` method is available

### Debug Endpoints

Use these endpoints to troubleshoot:

```bash
# Check memory statistics
curl -X GET "http://localhost:8001/chat/memory/stats"

# Debug all messages
curl -X GET "http://localhost:8001/chat/debug/all-messages"

# List all threads
curl -X GET "http://localhost:8001/chat/threads"
```

## Performance Considerations

### PostgreSQL Optimization

1. **Indexing**:
```sql
-- Create indexes for common queries
CREATE INDEX idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX idx_checkpoints_timestamp ON checkpoints(ts);
```

2. **Connection Pooling**:
```python
# Use connection pooling in production
connection_string = "postgresql://user:pass@host:5432/db?pool_size=20&max_overflow=30"
```

3. **Vacuum and Analyze**:
```sql
-- Regular maintenance
VACUUM ANALYZE checkpoints;
VACUUM ANALYZE checkpoint_writes;
```

### Memory Usage

- **In-Memory**: Suitable for development, limited by RAM
- **PostgreSQL**: Suitable for production, limited by disk space
- **Message Limits**: Configure appropriate limits for your use case

## Security Considerations

1. **Database Security**:
   - Use strong passwords
   - Limit database user permissions
   - Enable SSL connections in production

2. **API Security**:
   - SOEID validation and sanitization
   - Rate limiting on endpoints
   - Input validation and sanitization

3. **Data Privacy**:
   - Consider encryption at rest
   - Implement data retention policies
   - Secure deletion of sensitive data

This comprehensive LangGraph memory implementation provides enterprise-grade conversation memory management with full SOEID support and flexible storage options.
