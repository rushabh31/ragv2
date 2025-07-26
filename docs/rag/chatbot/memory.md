# Memory Systems

## 🎯 Overview

Memory systems enable the chatbot to maintain conversation context across interactions. They store and retrieve conversation history, allowing the AI to understand context, reference previous discussions, and provide coherent, contextual responses. The system supports both session-based and persistent memory with user-specific tracking.

## 🧠 How Memory Works

### **The Process**
1. **Message Storage**: Store user queries and AI responses with metadata
2. **Context Retrieval**: Retrieve relevant conversation history for new queries
3. **Context Integration**: Include conversation context in AI prompt
4. **Session Management**: Track conversations across sessions and users
5. **Cleanup**: Manage memory lifecycle and cleanup old conversations

```python
# Example memory flow
Session 1:
User: "What is machine learning?"
AI: "Machine learning is a subset of AI that enables computers to learn..."
→ Stored in memory

Session 2 (same user):
User: "Can you give me examples of what we discussed?"
→ Memory retrieves previous conversation
→ AI: "In our previous discussion about machine learning, I explained..."
```

## 🔧 Available Memory Systems

### **1. LangGraph Checkpoint Memory (Recommended)**
PostgreSQL-backed persistent memory using LangGraph checkpointers.

**Features:**
- Persistent storage across application restarts
- User-specific conversation tracking (SOEID-based)
- Cross-session chat history access
- Thread-based session management
- Scalable PostgreSQL backend

**Best for:**
- Production deployments
- Multi-user environments
- Long-term conversation tracking
- Enterprise applications

### **2. Simple Memory**
In-memory conversation storage for development and testing.

**Features:**
- Fast in-memory storage
- Session-based conversation tracking
- No external dependencies
- Automatic cleanup

**Best for:**
- Development and testing
- Single-user applications
- Temporary conversations
- Quick prototyping

## 📋 LangGraph Memory Configuration

### **Basic Setup**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"           # or "in_memory"
    postgres:
      connection_string: "postgresql://user:pass@localhost:5432/ragdb"
      pool_size: 10                  # Connection pool size
      max_overflow: 20               # Additional connections
      pool_timeout: 30               # Connection timeout
```

### **Advanced Configuration**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    postgres:
      connection_string: "postgresql://user:pass@localhost:5432/ragdb"
      pool_size: 10
      max_overflow: 20
      pool_timeout: 30
      
      # Table configuration
      table_prefix: "rag_"           # Table name prefix
      schema: "public"               # Database schema
      
      # Performance settings
      batch_size: 100                # Batch operations
      connection_timeout: 60         # Connection timeout
      query_timeout: 30              # Query timeout
      
      # Cleanup settings
      cleanup_enabled: true          # Enable automatic cleanup
      cleanup_interval: 86400        # Cleanup interval (seconds)
      max_session_age: 2592000       # Max session age (30 days)
      max_sessions_per_user: 100     # Max sessions per user
    
    # Cross-session chat history
    chat_history:
      enabled: true                  # Enable chat history
      default_days: 7                # Default lookback days
      max_days: 365                  # Maximum lookback days
      max_messages: 100              # Maximum messages to retrieve
      
    # Session management
    session:
      auto_create: true              # Auto-create sessions
      session_timeout: 3600          # Session timeout (seconds)
      max_session_length: 50         # Max messages per session
```

### **In-Memory Configuration**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "in_memory"
    in_memory:
      max_sessions: 1000             # Maximum stored sessions
      cleanup_interval: 3600         # Cleanup interval (seconds)
      session_timeout: 7200          # Session timeout (seconds)
```

## 🛠️ Memory Implementation

### **Using LangGraph Memory**
```python
from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory

# Initialize memory system
memory = LangGraphCheckpointMemory({
    "store_type": "postgres",
    "postgres": {
        "connection_string": "postgresql://user:pass@localhost:5432/ragdb"
    },
    "chat_history": {
        "enabled": True,
        "default_days": 7
    }
})

# Add a message to memory
await memory.add(
    session_id="session_123",
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    metadata={
        "user_id": "user_456",
        "soeid": "john.doe",
        "timestamp": "2024-01-15T10:30:00Z",
        "source_documents": ["doc1.pdf", "doc2.pdf"]
    }
)

# Retrieve conversation history
history = await memory.get_history(
    session_id="session_123",
    limit=10
)

print(f"Retrieved {len(history)} messages")
for msg in history:
    print(f"User: {msg['query']}")
    print(f"AI: {msg['response']}")
    print(f"Time: {msg['timestamp']}")
```

### **Cross-Session Chat History**
```python
# Get chat history across all sessions for a user
chat_history = await memory.get_chat_history_by_soeid_and_date(
    soeid="john.doe",
    days=14,  # Look back 14 days
    limit=50  # Maximum 50 messages
)

print(f"Found {len(chat_history)} messages across sessions")
for msg in chat_history:
    print(f"Session: {msg['session_id']}")
    print(f"Query: {msg['query']}")
    print(f"Response: {msg['response'][:100]}...")
    print(f"Time: {msg['timestamp']}")
    print("---")

# Use in generation context
if chat_history:
    context_messages = [
        f"Previous: {msg['query']} → {msg['response'][:200]}..."
        for msg in chat_history[-5:]  # Last 5 messages
    ]
    conversation_context = "\n".join(context_messages)
```

### **Session Management**
```python
# Get all sessions for a user
sessions = await memory.get_sessions_for_soeid("john.doe")

print(f"User has {len(sessions)} sessions")
for session in sessions:
    print(f"Session: {session['session_id']}")
    print(f"Messages: {session['message_count']}")
    print(f"Last activity: {session['last_activity']}")
    print(f"Created: {session['created_at']}")

# Clear old sessions
deleted_count = await memory.clear_session("old_session_123")
print(f"Deleted {deleted_count} messages from session")

# Get memory statistics
stats = await memory.get_memory_stats()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Total messages: {stats['total_messages']}")
print(f"Unique users: {stats['unique_users']}")
print(f"Active sessions: {stats['active_sessions']}")
```

## 📊 Memory Performance and Scalability

### **PostgreSQL Performance**
```python
# Performance characteristics
Database Size     | Query Time | Memory Usage | Concurrent Users
------------------|------------|--------------|------------------
< 10K messages    | < 10ms     | 100MB        | 50+
10K - 100K msgs   | 10-50ms    | 200MB        | 100+
100K - 1M msgs    | 50-200ms   | 500MB        | 200+
> 1M messages     | 200-500ms  | 1GB+         | 500+
```

### **Optimization Strategies**
```sql
-- Database indexes for performance
CREATE INDEX idx_checkpoints_thread_id ON rag_checkpoints(thread_id);
CREATE INDEX idx_checkpoints_timestamp ON rag_checkpoints(created_at);
CREATE INDEX idx_checkpoint_writes_thread_id ON rag_checkpoint_writes(thread_id);

-- Cleanup old data
DELETE FROM rag_checkpoints 
WHERE created_at < NOW() - INTERVAL '30 days';

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM rag_checkpoints 
WHERE thread_id = 'session_123' 
ORDER BY created_at DESC LIMIT 10;
```

### **Memory Usage Optimization**
```python
# Efficient memory retrieval
async def get_optimized_history(memory, session_id, max_messages=10):
    """Get conversation history with optimization."""
    
    # Get recent messages only
    history = await memory.get_history(
        session_id=session_id,
        limit=max_messages
    )
    
    # Compress old messages
    if len(history) > max_messages:
        recent_messages = history[:max_messages//2]
        old_messages = history[max_messages//2:]
        
        # Summarize old messages
        old_summary = summarize_conversation(old_messages)
        
        return recent_messages + [{"summary": old_summary}]
    
    return history

def summarize_conversation(messages):
    """Create a summary of conversation messages."""
    topics = []
    for msg in messages:
        # Extract key topics from each message
        topics.extend(extract_topics(msg['query']))
    
    unique_topics = list(set(topics))
    return f"Previous discussion covered: {', '.join(unique_topics[:5])}"
```

## 🔍 Chat History Features

### **Cross-Session History Access**
```python
# API endpoint for chat history
@app.post("/chat/message/json")
async def chat_with_history(
    request: ChatRequest,
    soeid: str = Header(...),
    use_chat_history: bool = False,
    chat_history_days: int = 7
):
    if use_chat_history and soeid:
        # Get cross-session chat history
        chat_history = await memory.get_chat_history_by_soeid_and_date(
            soeid=soeid,
            days=chat_history_days,
            limit=20
        )
        
        # Include in generation context
        if chat_history:
            context_summary = create_chat_history_summary(chat_history)
            # Use in AI generation...
    
    # Continue with normal processing...
```

### **Temporal Filtering**
```python
# Get history for specific time periods
async def get_temporal_history(memory, soeid, time_filter):
    """Get history based on temporal filters."""
    
    if time_filter == "today":
        days = 1
    elif time_filter == "this_week":
        days = 7
    elif time_filter == "this_month":
        days = 30
    elif time_filter == "this_year":
        days = 365
    else:
        days = 7  # Default
    
    return await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid,
        days=days,
        limit=100
    )
```

### **Context-Aware Retrieval**
```python
def get_relevant_history(chat_history, current_query, max_context=5):
    """Get most relevant historical messages for current query."""
    
    if not chat_history:
        return []
    
    # Score messages by relevance to current query
    scored_messages = []
    query_words = set(current_query.lower().split())
    
    for msg in chat_history:
        # Calculate relevance score
        msg_words = set((msg['query'] + ' ' + msg['response']).lower().split())
        overlap = len(query_words & msg_words)
        relevance_score = overlap / len(query_words) if query_words else 0
        
        scored_messages.append((msg, relevance_score))
    
    # Sort by relevance and recency
    scored_messages.sort(key=lambda x: (x[1], x[0]['timestamp']), reverse=True)
    
    # Return top relevant messages
    return [msg for msg, score in scored_messages[:max_context] if score > 0.1]
```

## 🚨 Common Issues and Solutions

### **Database Connection Issues**
```python
# Issue: Connection pool exhaustion
# Solution: Optimize connection settings
memory_config = {
    "postgres": {
        "pool_size": 20,              # Increase pool size
        "max_overflow": 40,           # More overflow connections
        "pool_timeout": 60,           # Longer timeout
        "pool_recycle": 3600          # Recycle connections
    }
}

# Issue: Connection timeouts
# Solution: Add retry logic
class RobustMemory:
    async def add_with_retry(self, *args, **kwargs):
        for attempt in range(3):
            try:
                return await self.memory.add(*args, **kwargs)
            except ConnectionError as e:
                if attempt == 2:  # Last attempt
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### **Memory Growth Issues**
```yaml
# Issue: Memory usage growing too large
# Solution: Enable cleanup and set limits
memory:
  postgres:
    cleanup_enabled: true
    max_session_age: 1209600        # 14 days
    max_sessions_per_user: 50       # Limit sessions per user
    cleanup_interval: 43200         # Cleanup twice daily
```

### **Slow History Retrieval**
```python
# Issue: Slow chat history queries
# Solution: Add database indexes and optimize queries
async def optimized_get_history(memory, soeid, days=7):
    # Use indexed query with limits
    return await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid,
        days=days,
        limit=20,  # Limit results
        order_by="timestamp DESC"  # Use indexed order
    )
```

### **Context Window Overflow**
```python
# Issue: Chat history exceeds context window
# Solution: Smart context management
def manage_context_window(chat_history, current_query, max_tokens=4000):
    """Manage context to fit within token limits."""
    
    # Estimate tokens
    def estimate_tokens(text):
        return len(text.split()) * 1.3
    
    # Start with current query
    context_parts = [f"Current query: {current_query}"]
    used_tokens = estimate_tokens(current_query)
    
    # Add history messages starting from most recent
    for msg in reversed(chat_history):
        msg_text = f"Previous: {msg['query']} → {msg['response']}"
        msg_tokens = estimate_tokens(msg_text)
        
        if used_tokens + msg_tokens > max_tokens:
            # Summarize remaining messages
            remaining_msgs = chat_history[:chat_history.index(msg)]
            if remaining_msgs:
                summary = f"Earlier discussion: {len(remaining_msgs)} messages about {extract_topics(remaining_msgs)}"
                context_parts.append(summary)
            break
        
        context_parts.append(msg_text)
        used_tokens += msg_tokens
    
    return "\n".join(reversed(context_parts))
```

## 🎯 Best Practices

### **Memory Lifecycle Management**
```python
class MemoryManager:
    def __init__(self, memory):
        self.memory = memory
        self.cleanup_scheduler = None
    
    async def start_cleanup_scheduler(self):
        """Start automatic cleanup process."""
        while True:
            try:
                await self.cleanup_old_sessions()
                await asyncio.sleep(86400)  # Daily cleanup
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def cleanup_old_sessions(self):
        """Clean up old sessions and messages."""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Get old sessions
        old_sessions = await self.memory.get_sessions_before_date(cutoff_date)
        
        # Archive important sessions before deletion
        for session in old_sessions:
            if session.get('important', False):
                await self.archive_session(session['session_id'])
        
        # Delete old sessions
        deleted_count = 0
        for session in old_sessions:
            if not session.get('important', False):
                count = await self.memory.clear_session(session['session_id'])
                deleted_count += count
        
        logger.info(f"Cleaned up {deleted_count} old messages")
```

### **Privacy and Security**
```python
def sanitize_memory_data(message_data):
    """Remove sensitive information from memory storage."""
    
    # Remove PII from messages
    sanitized = message_data.copy()
    
    # Remove sensitive fields
    sensitive_fields = ['ssn', 'credit_card', 'password', 'api_key']
    for field in sensitive_fields:
        if field in sanitized.get('metadata', {}):
            del sanitized['metadata'][field]
    
    # Mask sensitive patterns in content
    sanitized['query'] = mask_sensitive_patterns(sanitized['query'])
    sanitized['response'] = mask_sensitive_patterns(sanitized['response'])
    
    return sanitized

def mask_sensitive_patterns(text):
    """Mask sensitive patterns in text."""
    import re
    
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    
    # Mask phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    
    # Mask credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 
                  '[CARD]', text)
    
    return text
```

### **Performance Monitoring**
```python
class MemoryMetrics:
    def __init__(self, memory):
        self.memory = memory
        self.metrics = {
            'total_queries': 0,
            'avg_query_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def track_query_performance(self, operation, *args, **kwargs):
        """Track memory operation performance."""
        start_time = time.time()
        
        try:
            result = await operation(*args, **kwargs)
            self.metrics['total_queries'] += 1
            
            # Update average query time
            query_time = time.time() - start_time
            self.metrics['avg_query_time'] = (
                (self.metrics['avg_query_time'] * (self.metrics['total_queries'] - 1) + query_time) 
                / self.metrics['total_queries']
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Memory operation failed: {e}")
            raise
    
    def get_performance_report(self):
        """Get performance metrics report."""
        return {
            'total_queries': self.metrics['total_queries'],
            'average_query_time_ms': self.metrics['avg_query_time'] * 1000,
            'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_queries'], 1),
            'queries_per_second': self.metrics['total_queries'] / max(time.time() - self.start_time, 1)
        }
```

## 🔧 Custom Memory Development

### **Creating a Custom Memory System**
```python
from src.rag.chatbot.memory.base_memory import BaseMemory

class CustomMemory(BaseMemory):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_storage = self._initialize_storage(config)
        self.custom_config = config.get("custom_config", {})
    
    def _initialize_storage(self, config):
        # Initialize custom storage backend
        pass
    
    async def add(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add message to custom memory storage."""
        
        message_data = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Custom storage logic
        await self.custom_storage.store_message(message_data)
    
    async def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history from custom storage."""
        
        # Custom retrieval logic
        messages = await self.custom_storage.get_messages(
            session_id=session_id,
            limit=limit,
            order="desc"
        )
        
        return messages
    
    async def clear_session(self, session_id: str) -> int:
        """Clear session from custom storage."""
        
        # Custom cleanup logic
        deleted_count = await self.custom_storage.delete_session(session_id)
        return deleted_count
```

### **Registering Custom Memory**
```python
# In memory factory
from src.rag.chatbot.memory.memory_factory import MemoryFactory

MemoryFactory.register_memory("custom", CustomMemory)

# Use in configuration
chatbot:
  memory:
    type: "custom"
    custom:
      storage_backend: "redis"
      custom_config:
        host: "localhost"
        port: 6379
```

## 📚 Related Documentation

- **[Response Generators](./generators.md)** - Use memory in response generation
- **[Workflow Management](./workflow.md)** - Integrate memory with workflows
- **[Chatbot API](./api.md)** - Memory management endpoints
- **[PostgreSQL Setup](../../memory/postgresql.md)** - Database configuration

## 🚀 Quick Examples

### **Basic Memory Usage**
```python
# Initialize and use memory
memory = LangGraphCheckpointMemory(config)

# Add conversation
await memory.add(
    session_id="session_123",
    query="What is AI?",
    response="AI is artificial intelligence...",
    metadata={"user_id": "user_456", "soeid": "john.doe"}
)

# Get history
history = await memory.get_history("session_123", limit=5)
for msg in history:
    print(f"Q: {msg['query']}")
    print(f"A: {msg['response']}")
```

### **API Usage with Memory**
```bash
# Chat with conversation history
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: john.doe" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did we discuss about machine learning?",
    "use_history": true,
    "use_chat_history": true,
    "chat_history_days": 14
  }'

# Get chat history
curl -X GET "http://localhost:8001/chat/history/john.doe" \
  -H "X-API-Key: test-api-key"
```

---

**Next Steps**: 
- [Configure Workflow Management](./workflow.md)
- [Use the Chatbot API](./api.md)
- [Set up PostgreSQL](../../memory/postgresql.md)
