# LangGraph Memory Implementation

This document describes the LangGraph memory implementation for the RAG system, which provides both in-memory and PostgreSQL storage options for conversation history and long-term memory with SOEID (Source of Entity ID) support.

## Overview

The LangGraph memory implementation leverages LangGraph's memory stores to provide:
- **Short-term memory**: Conversation history within sessions
- **Long-term memory**: Persistent memory across sessions and users
- **Semantic search**: Find relevant conversation history using embeddings
- **Flexible storage**: Choose between in-memory or PostgreSQL storage
- **SOEID support**: User identification and cross-session memory retrieval

## Memory Types

### 1. In-Memory Store
- Fastest performance
- Data lost on restart
- Good for development and testing
- No external dependencies

### 2. PostgreSQL Store
- Persistent storage
- Scalable for production
- Requires PostgreSQL setup
- Supports complex queries and indexing

## SOEID (Source of Entity ID) Support

SOEID allows the system to maintain user-specific memory across different sessions and conversations. This enables:

- **Cross-session memory**: Retrieve conversation history for a specific user across multiple sessions
- **User-specific context**: Provide personalized responses based on user's conversation history
- **Long-term user memory**: Store user preferences and characteristics
- **Semantic user search**: Find relevant conversations for a specific user

### SOEID Usage

```python
# Add conversation with SOEID
await memory.add(
    session_id="session_123",
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    metadata={
        "soeid": "SOEID_ABC123",
        "user_id": "user_123"
    }
)

# Get user history by SOEID
user_history = await memory.get_user_history_by_soeid(
    soeid="SOEID_ABC123",
    limit=10
)

# Get relevant user history by SOEID
relevant_history = await memory.get_user_relevant_history_by_soeid(
    soeid="SOEID_ABC123",
    query="Tell me about neural networks",
    limit=5
)
```

## Configuration

### Basic Configuration

```yaml
chatbot:
  memory:
    # Memory type: "simple", "mem0", or "langgraph"
    type: langgraph
    # LangGraph memory settings
    store_type: in_memory  # "in_memory" or "postgres"
    embedding_dimensions: 384
    max_history: 20
    # PostgreSQL settings (only used if store_type is "postgres")
    postgres:
      connection_string: "${POSTGRES_CONNECTION_STRING}"
```

### Environment Variables

For PostgreSQL storage, set the connection string:
```bash
export POSTGRES_CONNECTION_STRING="postgresql://user:password@localhost:5432/rag_memory"
```

## Usage

### Using the Memory Factory

```python
from controlsgenai.funcs.rag.src.chatbot.memory.memory_factory import MemoryFactory

# Create in-memory memory
config = {
    "type": "langgraph",
    "store_type": "in_memory",
    "embedding_dimensions": 384,
    "max_history": 10
}
memory = MemoryFactory.create_memory(config)

# Add conversation with SOEID
await memory.add(
    session_id="user_123",
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    metadata={
        "soeid": "SOEID_ABC123",
        "topic": "AI"
    }
)

# Get conversation history
history = await memory.get_history("user_123", limit=5)

# Get user history by SOEID
user_history = await memory.get_user_history_by_soeid("SOEID_ABC123", limit=10)

# Get relevant history for a specific user
relevant_history = await memory.get_user_relevant_history_by_soeid(
    soeid="SOEID_ABC123",
    query="Tell me about neural networks",
    limit=3
)
```

### Long-Term Memory with SOEID

```python
# Add user-specific long-term memory
await memory.add_long_term_memory(
    namespace=("SOEID_ABC123", "user_profile"),
    key="preferences",
    data={
        "preferred_topics": ["AI", "machine learning"],
        "expertise_level": "intermediate",
        "communication_style": "technical"
    }
)

# Retrieve user-specific long-term memory
preferences = await memory.get_long_term_memory(
    namespace=("SOEID_ABC123", "user_profile"),
    key="preferences"
)

# Search user-specific memory
results = await memory.search_long_term_memory(
    namespace=("SOEID_ABC123", "user_profile"),
    query="technical communication",
    limit=5
)
```

## API Integration

### SOEID-based API Endpoint

The system provides two API endpoints to retrieve all chat history for a user by SOEID:

#### Endpoint 1: `GET /chat/user/history` (SOEID in Header)

**Headers:**
- `SOEID`: User's Source of Entity ID (required)

**Query Parameters:**
- `limit`: Maximum number of messages to retrieve (optional)

#### Endpoint 2: `GET /chat/user/history/{soeid}` (SOEID in Path)

**Path Parameters:**
- `soeid`: User's Source of Entity ID (required)

**Query Parameters:**
- `limit`: Maximum number of messages to retrieve (optional)

**Response Format:**
```json
{
  "soeid": "SOEID_ABC123",
  "messages": [
    {
      "session_id": "session_1",
      "role": "user",
      "content": "What is machine learning?",
      "timestamp": "2025-07-23T20:30:00.000000",
      "metadata": {
        "soeid": "SOEID_ABC123",
        "user_id": "user_123"
      }
    },
    {
      "session_id": "session_1",
      "role": "assistant",
      "content": "Machine learning is a subset of AI...",
      "timestamp": "2025-07-23T20:30:01.000000",
      "metadata": {
        "soeid": "SOEID_ABC123",
        "user_id": "user_123"
      }
    }
  ],
  "total_messages": 2,
  "total_sessions": 1,
  "metadata": {
    "session_ids": ["session_1"],
    "retrieved_at": "2025-07-23T20:30:05.000000"
  }
}
```

**Key Features:**
- ✅ Returns all messages across all sessions for the user
- ✅ Each message includes the session_id
- ✅ Includes total message and session counts
- ✅ Supports pagination with limit parameter
- ✅ Includes metadata with session IDs and retrieval timestamp
- ✅ Both endpoints return identical data

#### Usage Examples

**Python Requests (Header-based):**
```python
import requests

# Get user history by SOEID (header)
response = requests.get(
    "http://localhost:8001/chat/user/history",
    headers={"SOEID": "SOEID_ABC123"},
    params={"limit": 10}
)

data = response.json()
print(f"Total messages: {data['total_messages']}")
print(f"Total sessions: {data['total_sessions']}")

for msg in data['messages']:
    print(f"Session: {msg['session_id']}, Role: {msg['role']}, Content: {msg['content']}")
```

**Python Requests (Path-based):**
```python
import requests

# Get user history by SOEID (path)
response = requests.get(
    "http://localhost:8001/chat/user/history/SOEID_ABC123",
    params={"limit": 10}
)

data = response.json()
print(f"Total messages: {data['total_messages']}")
print(f"Total sessions: {data['total_sessions']}")

for msg in data['messages']:
    print(f"Session: {msg['session_id']}, Role: {msg['role']}, Content: {msg['content']}")
```

**cURL (Header-based):**
```bash
curl -X GET "http://localhost:8001/chat/user/history?limit=10" \
  -H "SOEID: SOEID_ABC123"
```

**cURL (Path-based):**
```bash
curl -X GET "http://localhost:8001/chat/user/history/SOEID_ABC123?limit=10"
```

**JavaScript/Fetch (Header-based):**
```javascript
const response = await fetch('http://localhost:8001/chat/user/history?limit=10', {
  headers: {
    'SOEID': 'SOEID_ABC123'
  }
});

const data = await response.json();
console.log(`Total messages: ${data.total_messages}`);
console.log(`Total sessions: ${data.total_sessions}`);
```

**JavaScript/Fetch (Path-based):**
```javascript
const response = await fetch('http://localhost:8001/chat/user/history/SOEID_ABC123?limit=10');

const data = await response.json();
console.log(`Total messages: ${data.total_messages}`);
console.log(`Total sessions: ${data.total_sessions}`);
```

### Chat Message Endpoint with SOEID

The existing chat endpoint also supports SOEID:

#### Endpoint: `POST /chat/message`

**Headers:**
- `SOEID`: User's Source of Entity ID (required)

**Form Data:**
- `query`: User's message
- `session_id`: Optional session ID
- `use_retrieval`: Whether to use retrieval (default: true)
- `use_history`: Whether to use conversation history (default: true)
- `metadata_json`: Optional JSON metadata

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8001/chat/message",
    headers={"SOEID": "SOEID_ABC123"},
    data={
        "query": "What is machine learning?",
        "session_id": "session_123",
        "use_retrieval": True,
        "use_history": True
    }
)

data = response.json()
print(f"Response: {data['response']}")
print(f"Session ID: {data['session_id']}")
```

## Integration with LangGraph Workflow

The memory system is integrated into the LangGraph workflow with SOEID support:

1. **Retrieve**: Get relevant documents
2. **Rerank**: Rerank documents for relevance
3. **Generate**: Generate response using context and user conversation history
4. **Update Memory**: Store the interaction in memory with SOEID
5. **Decide Next**: Determine next workflow step

### Workflow State with SOEID

The memory system uses the following state structure:
```python
RAGWorkflowState = {
    "query": str,
    "session_id": str,
    "soeid": str,  # Source of Entity ID for user identification
    "response": str,
    "retrieved_documents": List[RetrievedDocument],
    "reranked_documents": List[RetrievedDocument],
    "metrics": Dict[str, Any]
}
```

## Features

### 1. Session Management
- Each session maintains its own conversation history
- Sessions are isolated from each other
- Support for session-specific metadata

### 2. SOEID-based User Management
- Cross-session user identification
- User-specific conversation history
- Personalized context retrieval
- User-specific long-term memory

### 3. Semantic Search
- Uses embeddings to find relevant conversation history
- Supports similarity-based retrieval
- Configurable search limits
- User-specific semantic search

### 4. Long-Term Memory
- Persistent storage across sessions
- Namespace-based organization
- Support for complex data structures
- User-specific memory namespaces

### 5. Metadata Support
- Rich metadata for each interaction
- Custom metadata fields
- Metadata-based filtering
- SOEID integration

## Performance Considerations

### In-Memory Store
- **Pros**: Fastest performance, no network latency
- **Cons**: Data lost on restart, memory usage grows with data
- **Use case**: Development, testing, small-scale deployments

### PostgreSQL Store
- **Pros**: Persistent, scalable, ACID compliance
- **Cons**: Network latency, requires database setup
- **Use case**: Production deployments, large-scale systems

## Migration from Other Memory Types

### From Simple Memory
```python
# Old simple memory
from controlsgenai.funcs.rag.src.chatbot.memory.simple_memory import SimpleMemory
memory = SimpleMemory(config)

# New LangGraph memory with SOEID
from controlsgenai.funcs.rag.src.chatbot.memory.memory_factory import MemoryFactory
memory = MemoryFactory.create_memory({"type": "langgraph", "store_type": "in_memory"})

# Add with SOEID
await memory.add(
    session_id="user_123",
    query="Hello",
    response="Hi there!",
    metadata={"soeid": "SOEID_ABC123"}
)
```

### From Mem0 Memory
```python
# Old mem0 memory
from controlsgenai.funcs.rag.src.chatbot.memory.mem0_memory import Mem0Memory
memory = Mem0Memory(config)

# New LangGraph memory with SOEID
from controlsgenai.funcs.rag.src.chatbot.memory.memory_factory import MemoryFactory
memory = MemoryFactory.create_memory({"type": "langgraph", "store_type": "postgres"})
```

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Error**
   - Verify connection string format
   - Check database permissions
   - Ensure PostgreSQL is running

2. **Memory Not Persisting**
   - Check store_type configuration
   - Verify async/await usage
   - Check for exceptions in logs

3. **Poor Search Results**
   - Adjust embedding dimensions
   - Check embedding function quality
   - Verify search parameters

4. **SOEID Not Working**
   - Ensure SOEID is included in metadata
   - Check namespace structure
   - Verify user history retrieval

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("controlsgenai.funcs.rag.src.chatbot.memory").setLevel(logging.DEBUG)
```

## Examples

See `examples/langgraph_memory_example.py` for basic usage examples.
See `examples/soeid_memory_example.py` for SOEID-specific examples.

## API Reference

### LangGraphMemory Class

#### Methods

- `add(session_id, query, response, metadata)`: Add conversation interaction
- `get_history(session_id, limit)`: Get conversation history
- `get_relevant_history(session_id, query, limit)`: Get relevant history
- `get_user_history_by_soeid(soeid, limit)`: Get user history by SOEID
- `get_user_relevant_history_by_soeid(soeid, query, limit)`: Get relevant user history by SOEID
- `clear_session(session_id)`: Clear session history
- `clear_user_history(soeid)`: Clear user history by SOEID
- `add_long_term_memory(namespace, key, data)`: Add long-term memory
- `get_long_term_memory(namespace, key)`: Get long-term memory
- `search_long_term_memory(namespace, query, filter_dict, limit)`: Search long-term memory

#### Configuration Options

- `type`: Memory type ("langgraph")
- `store_type`: Storage type ("in_memory" or "postgres")
- `embedding_dimensions`: Embedding vector dimensions
- `max_history`: Maximum history items to retrieve
- `postgres.connection_string`: PostgreSQL connection string 