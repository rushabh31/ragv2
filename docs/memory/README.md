# Memory Systems

## 🎯 Overview

Memory systems are the backbone of conversational AI, enabling the RAG chatbot to maintain context, remember previous interactions, and provide coherent responses across multiple conversations. This documentation covers the comprehensive memory architecture, including conversation history, cross-session chat history, and persistent storage solutions.

## 🧠 Memory Architecture

### **Memory Types**
The RAG system supports multiple memory types for different use cases:

1. **Session Memory**: Conversation history within a single session
2. **Cross-Session Memory**: Chat history across multiple sessions for the same user
3. **Long-Term Memory**: Persistent storage using PostgreSQL with LangGraph
4. **Cache Memory**: Temporary storage for performance optimization

### **Memory Hierarchy**
```
User (SOEID)
├── Session 1 (Thread ID: thread_abc123)
│   ├── Message 1: Query + Response + Metadata
│   ├── Message 2: Query + Response + Metadata
│   └── Message N: Query + Response + Metadata
├── Session 2 (Thread ID: thread_def456)
│   ├── Message 1: Query + Response + Metadata
│   └── Message N: Query + Response + Metadata
└── Session N...
```

## 🏗️ Memory Components

### **1. LangGraph Checkpoint Memory (Primary)**
Enterprise-grade memory system using LangGraph checkpointers with PostgreSQL backend.

**Features:**
- Persistent storage across application restarts
- User-specific conversation tracking (SOEID-based)
- Thread-based session management
- Cross-session chat history access
- Scalable PostgreSQL backend with pgvector support
- Automatic cleanup and maintenance

**Architecture:**
```python
# LangGraph Memory Structure
Checkpointer (PostgreSQL)
├── Checkpoints Table: Thread states and conversation data
├── Checkpoint Writes Table: Individual message writes
├── Checkpoint Blobs Table: Large data storage
└── Checkpoint Migrations Table: Schema versioning

Store (PostgreSQL)
├── Cross-session data storage
├── User preference storage
└── Long-term memory storage
```

### **2. Simple Memory (Development)**
In-memory storage for development and testing environments.

**Features:**
- Fast in-memory operations
- No external dependencies
- Session-based conversation tracking
- Automatic cleanup on restart

**Use Cases:**
- Development and testing
- Single-user applications
- Temporary conversations
- Quick prototyping

## 📋 Configuration

### **LangGraph PostgreSQL Memory**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    
    # PostgreSQL configuration
    postgres:
      connection_string: "postgresql://username:password@localhost:5432/ragdb"
      pool_size: 10                    # Connection pool size
      max_overflow: 20                 # Additional connections
      pool_timeout: 30                 # Connection timeout (seconds)
      
      # Table configuration
      table_prefix: "rag_"             # Table name prefix
      schema: "public"                 # Database schema
      
      # Performance settings
      batch_size: 100                  # Batch operations
      connection_timeout: 60           # Connection timeout
      query_timeout: 30                # Query timeout
      
      # Cleanup settings
      cleanup_enabled: true            # Enable automatic cleanup
      cleanup_interval: 86400          # Cleanup interval (seconds)
      max_session_age: 2592000         # Max session age (30 days)
      max_sessions_per_user: 100       # Max sessions per user
    
    # Cross-session chat history
    chat_history:
      enabled: true                    # Enable chat history
      default_days: 7                  # Default lookback days
      max_days: 365                    # Maximum lookback days
      max_messages: 100                # Maximum messages to retrieve
      
    # Session management
    session:
      auto_create: true                # Auto-create sessions
      session_timeout: 3600            # Session timeout (seconds)
      max_session_length: 50           # Max messages per session
```

### **In-Memory Development Setup**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "in_memory"
    
    # In-memory configuration
    in_memory:
      max_sessions: 1000               # Maximum stored sessions
      cleanup_interval: 3600           # Cleanup interval (seconds)
      session_timeout: 7200            # Session timeout (seconds)
      max_memory_usage_mb: 500         # Maximum memory usage
```

### **Simple Memory Setup**
```yaml
chatbot:
  memory:
    type: "simple"
    
    # Simple memory configuration
    simple:
      max_history_length: 20           # Max messages per session
      cleanup_interval: 1800           # Cleanup interval (seconds)
      session_timeout: 3600            # Session timeout (seconds)
```

## 🛠️ Memory Usage

### **Basic Memory Operations**
```python
from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory

# Initialize memory system
memory = LangGraphCheckpointMemory({
    "store_type": "postgres",
    "postgres": {
        "connection_string": "postgresql://user:pass@localhost:5432/ragdb"
    }
})

# Add a conversation message
await memory.add(
    session_id="session_123",
    query="What is machine learning?",
    response="Machine learning is a subset of artificial intelligence...",
    metadata={
        "user_id": "user_456",
        "soeid": "john.doe",
        "timestamp": "2024-01-15T10:30:00Z",
        "source_documents": ["doc1.pdf", "doc2.pdf"],
        "model_used": "gemini-1.5-pro-002",
        "processing_time": 2.34
    }
)

# Retrieve session conversation history
history = await memory.get_history(
    session_id="session_123",
    limit=10
)

print(f"Retrieved {len(history)} messages")
for msg in history:
    print(f"User: {msg['query']}")
    print(f"AI: {msg['response']}")
    print(f"Time: {msg['timestamp']}")
    print("---")
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

## 📊 Memory Performance

### **Performance Characteristics**
```python
# Performance by database size
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
CREATE INDEX idx_checkpoint_writes_timestamp ON rag_checkpoint_writes(created_at);

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
    
    # Compress old messages if needed
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

# Usage in API
@app.get("/chat/history/{soeid}/temporal")
async def get_temporal_chat_history(
    soeid: str,
    time_filter: str = "this_week",
    limit: int = 50
):
    history = await get_temporal_history(memory, soeid, time_filter)
    return {"soeid": soeid, "time_filter": time_filter, "messages": history}
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

# Usage in generation
relevant_history = get_relevant_history(chat_history, current_query)
if relevant_history:
    context = "Relevant previous discussions:\n"
    for msg in relevant_history:
        context += f"Q: {msg['query']}\nA: {msg['response'][:200]}...\n\n"
```

### **Smart Context Management**
```python
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

## 🔧 Memory Maintenance

### **Automatic Cleanup**
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
    
    async def archive_session(self, session_id):
        """Archive important session data."""
        
        # Get session data
        history = await self.memory.get_history(session_id, limit=None)
        
        # Create archive record
        archive_data = {
            "session_id": session_id,
            "archived_at": datetime.now().isoformat(),
            "message_count": len(history),
            "summary": self.create_session_summary(history),
            "messages": history
        }
        
        # Store in archive (implement based on your needs)
        await self.store_archive(archive_data)
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

## 📊 Memory Analytics

### **Usage Analytics**
```python
class MemoryAnalytics:
    def __init__(self, memory):
        self.memory = memory
    
    async def get_usage_analytics(self, days=30):
        """Get comprehensive memory usage analytics."""
        
        # Get basic stats
        stats = await self.memory.get_memory_stats()
        
        # Calculate additional metrics
        analytics = {
            "basic_stats": stats,
            "user_activity": await self.get_user_activity_stats(days),
            "session_patterns": await self.get_session_patterns(days),
            "content_analysis": await self.get_content_analysis(days),
            "performance_metrics": await self.get_performance_metrics(days)
        }
        
        return analytics
    
    async def get_user_activity_stats(self, days):
        """Analyze user activity patterns."""
        
        # Get active users by day
        active_users_by_day = {}
        
        for day in range(days):
            date = datetime.now() - timedelta(days=day)
            users = await self.memory.get_active_users_on_date(date)
            active_users_by_day[date.strftime('%Y-%m-%d')] = len(users)
        
        return {
            "active_users_by_day": active_users_by_day,
            "average_daily_users": sum(active_users_by_day.values()) / days,
            "peak_usage_day": max(active_users_by_day, key=active_users_by_day.get)
        }
    
    async def get_session_patterns(self, days):
        """Analyze session patterns."""
        
        sessions = await self.memory.get_recent_sessions(days)
        
        session_lengths = [s['message_count'] for s in sessions]
        session_durations = [s['duration_minutes'] for s in sessions if 'duration_minutes' in s]
        
        return {
            "total_sessions": len(sessions),
            "average_session_length": sum(session_lengths) / len(session_lengths) if session_lengths else 0,
            "average_session_duration": sum(session_durations) / len(session_durations) if session_durations else 0,
            "session_length_distribution": self.calculate_distribution(session_lengths),
            "most_active_hours": self.get_most_active_hours(sessions)
        }
```

## 🎯 Best Practices

### **Memory Design Principles**
1. **User Privacy**: Always sanitize sensitive information
2. **Performance**: Use appropriate indexes and query optimization
3. **Scalability**: Design for growth with cleanup and archiving
4. **Reliability**: Implement retry logic and error handling
5. **Observability**: Monitor memory usage and performance

### **Configuration Guidelines**
```yaml
# Production configuration
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    postgres:
      # Connection pool sized for expected load
      pool_size: 20
      max_overflow: 40
      
      # Aggressive cleanup for production
      cleanup_enabled: true
      cleanup_interval: 21600      # 6 hours
      max_session_age: 604800      # 7 days
      max_sessions_per_user: 25    # Limit per user
    
    chat_history:
      enabled: true
      default_days: 3              # Conservative default
      max_days: 30                 # Reasonable maximum
      max_messages: 50             # Limit context size
```

### **Monitoring and Alerting**
```python
class MemoryMonitoring:
    def __init__(self, memory):
        self.memory = memory
        self.alert_thresholds = {
            "max_db_size_gb": 10,
            "max_query_time_ms": 1000,
            "max_sessions_per_user": 50,
            "max_memory_usage_mb": 2000
        }
    
    async def check_health(self):
        """Check memory system health."""
        
        health_status = {
            "status": "healthy",
            "checks": {},
            "alerts": []
        }
        
        # Check database size
        db_size = await self.memory.get_database_size()
        health_status["checks"]["database_size_gb"] = db_size
        
        if db_size > self.alert_thresholds["max_db_size_gb"]:
            health_status["alerts"].append(f"Database size ({db_size}GB) exceeds threshold")
            health_status["status"] = "warning"
        
        # Check query performance
        avg_query_time = await self.memory.get_average_query_time()
        health_status["checks"]["avg_query_time_ms"] = avg_query_time
        
        if avg_query_time > self.alert_thresholds["max_query_time_ms"]:
            health_status["alerts"].append(f"Query time ({avg_query_time}ms) exceeds threshold")
            health_status["status"] = "warning"
        
        return health_status
```

## 📚 Related Documentation

- **[Chat History](./chat-history.md)** - Detailed chat history implementation
- **[PostgreSQL Setup](./postgresql.md)** - Database configuration and setup
- **[Memory API](./api.md)** - Memory management API endpoints
- **[Chatbot Memory](../rag/chatbot/memory.md)** - Chatbot-specific memory usage

## 🚀 Quick Examples

### **Basic Setup**
```python
# Initialize memory
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
```

### **Cross-Session Usage**
```python
# Get user's chat history across sessions
chat_history = await memory.get_chat_history_by_soeid_and_date(
    soeid="john.doe",
    days=7,
    limit=20
)

# Use in context
if chat_history:
    context = create_context_from_history(chat_history)
```

---

**Next Steps**: 
- [Set up PostgreSQL](./postgresql.md)
- [Configure Chat History](./chat-history.md)
- [Use Memory API](./api.md)
