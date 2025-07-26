# Chat History

## 🎯 Overview

Chat history enables the RAG system to maintain conversation context across multiple sessions for the same user. Unlike session-based memory that only remembers conversations within a single session, chat history provides cross-session memory, allowing users to reference previous conversations from days, weeks, or months ago. This feature is essential for creating truly conversational AI experiences.

## 🔄 How Chat History Works

### **Cross-Session Memory Flow**
1. **User Identification**: Each user is identified by their SOEID (Single Online Enterprise ID)
2. **Message Storage**: All conversations are stored with user identification and timestamps
3. **History Retrieval**: When enabled, the system retrieves relevant past conversations
4. **Context Integration**: Historical context is included in AI response generation
5. **Temporal Filtering**: History can be filtered by time periods (days, weeks, months)

```python
# Example chat history flow
Day 1, Session A:
User (john.doe): "What is machine learning?"
AI: "Machine learning is a subset of AI that enables computers to learn..."
→ Stored with SOEID: john.doe, timestamp: 2024-01-15T10:00:00Z

Day 3, Session B:
User (john.doe): "Can you give me examples of what we discussed about ML?"
→ System retrieves chat history for john.doe from last 7 days
→ AI: "In our previous discussion about machine learning, I explained that it's a subset of AI..."
```

## 🏗️ Chat History Architecture

### **Data Structure**
```python
# Chat history message structure
{
    "session_id": "session_abc123",
    "soeid": "john.doe",
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of artificial intelligence...",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
        "user_id": "user_456",
        "model_used": "gemini-1.5-pro-002",
        "retrieved_documents": 5,
        "processing_time": 2.34,
        "source_documents": ["ml_textbook.pdf", "ai_guide.pdf"]
    }
}
```

### **Storage Architecture**
```
PostgreSQL Database (LangGraph Checkpointer)
├── rag_checkpoints
│   ├── thread_id (session_id)
│   ├── checkpoint_data (conversation state)
│   ├── created_at (timestamp)
│   └── metadata (user info, SOEID)
├── rag_checkpoint_writes
│   ├── thread_id (session_id)
│   ├── write_data (individual messages)
│   ├── created_at (timestamp)
│   └── metadata (message metadata)
└── Indexes
    ├── idx_checkpoints_soeid
    ├── idx_checkpoints_timestamp
    └── idx_writes_timestamp
```

## 📋 Configuration

### **Enable Chat History**
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    
    # Chat history configuration
    chat_history:
      enabled: true                    # Enable cross-session chat history
      default_days: 7                  # Default lookback period
      max_days: 365                    # Maximum allowed lookback
      max_messages: 100                # Maximum messages to retrieve
      
      # Performance settings
      cache_enabled: true              # Cache frequent queries
      cache_ttl: 3600                  # Cache TTL in seconds
      batch_size: 50                   # Batch size for queries
      
      # Filtering options
      relevance_threshold: 0.1         # Minimum relevance score
      include_metadata: true           # Include message metadata
      exclude_system_messages: true    # Exclude system messages
      
      # Privacy settings
      sanitize_content: true           # Remove sensitive information
      max_content_length: 1000         # Truncate long messages
```

### **API Configuration**
```yaml
api:
  chat_history:
    default_enabled: false             # Default state for new requests
    max_days_allowed: 365              # Maximum days users can request
    rate_limit:
      enabled: true                    # Enable rate limiting
      requests_per_minute: 30          # Limit history requests
    
    # Response formatting
    include_session_metadata: true     # Include session info
    group_by_session: true             # Group messages by session
    sort_order: "desc"                 # newest first or "asc" for oldest first
```

## 🛠️ Implementation

### **Basic Chat History Usage**
```python
from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory

# Initialize memory with chat history enabled
memory = LangGraphCheckpointMemory({
    "store_type": "postgres",
    "postgres": {
        "connection_string": "postgresql://user:pass@localhost:5432/ragdb"
    },
    "chat_history": {
        "enabled": True,
        "default_days": 7,
        "max_messages": 50
    }
})

# Get chat history for a user
chat_history = await memory.get_chat_history_by_soeid_and_date(
    soeid="john.doe",
    days=14,  # Look back 14 days
    limit=30  # Maximum 30 messages
)

print(f"Retrieved {len(chat_history)} messages from chat history")
for msg in chat_history:
    print(f"Session: {msg['session_id']}")
    print(f"Time: {msg['timestamp']}")
    print(f"Query: {msg['query']}")
    print(f"Response: {msg['response'][:100]}...")
    print("---")
```

### **Advanced Chat History Retrieval**
```python
async def get_contextual_chat_history(memory, soeid, current_query, days=7):
    """Get chat history with contextual relevance filtering."""
    
    # Get raw chat history
    raw_history = await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid,
        days=days,
        limit=100
    )
    
    if not raw_history:
        return []
    
    # Filter for relevance to current query
    relevant_history = filter_relevant_messages(raw_history, current_query)
    
    # Sort by relevance and recency
    scored_history = score_message_relevance(relevant_history, current_query)
    
    # Return top relevant messages
    return scored_history[:20]

def filter_relevant_messages(history, query, min_relevance=0.1):
    """Filter messages based on relevance to current query."""
    
    query_words = set(query.lower().split())
    relevant_messages = []
    
    for msg in history:
        # Calculate word overlap
        msg_text = (msg['query'] + ' ' + msg['response']).lower()
        msg_words = set(msg_text.split())
        
        overlap = len(query_words & msg_words)
        relevance = overlap / len(query_words) if query_words else 0
        
        if relevance >= min_relevance:
            msg['relevance_score'] = relevance
            relevant_messages.append(msg)
    
    return relevant_messages

def score_message_relevance(messages, query):
    """Score and sort messages by relevance and recency."""
    
    # Combine relevance and recency scores
    for msg in messages:
        # Relevance score (0-1)
        relevance_score = msg.get('relevance_score', 0)
        
        # Recency score (0-1, newer = higher)
        timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
        days_ago = (datetime.now(timezone.utc) - timestamp).days
        recency_score = max(0, 1 - (days_ago / 30))  # Decay over 30 days
        
        # Combined score (weighted)
        combined_score = (relevance_score * 0.7) + (recency_score * 0.3)
        msg['combined_score'] = combined_score
    
    # Sort by combined score
    return sorted(messages, key=lambda x: x['combined_score'], reverse=True)
```

### **Chat History in Generation Context**
```python
async def generate_with_chat_history(generator, memory, query, session_id, soeid, 
                                   use_chat_history=False, chat_history_days=7):
    """Generate response with optional chat history context."""
    
    # Get session history (always included)
    session_history = await memory.get_history(session_id, limit=10)
    
    # Get chat history if enabled
    chat_history = []
    if use_chat_history and soeid:
        chat_history = await get_contextual_chat_history(
            memory, soeid, query, chat_history_days
        )
    
    # Create context from chat history
    context_parts = []
    
    if chat_history:
        context_parts.append("Previous discussions:")
        for msg in chat_history[-5:]:  # Last 5 relevant messages
            context_parts.append(
                f"Q: {msg['query']}\nA: {msg['response'][:200]}...\n"
            )
    
    if session_history:
        context_parts.append("Current session:")
        for msg in session_history[-3:]:  # Last 3 session messages
            context_parts.append(
                f"Q: {msg['query']}\nA: {msg['response'][:200]}...\n"
            )
    
    # Combine context
    conversation_context = "\n".join(context_parts)
    
    # Generate response with context
    response = await generator.generate(
        query=query,
        documents=[],  # Assume no document retrieval for this example
        conversation_history=session_history,
        chat_history=chat_history,
        additional_context=conversation_context
    )
    
    return response, {
        "used_chat_history": len(chat_history) > 0,
        "chat_history_messages": len(chat_history),
        "session_history_messages": len(session_history)
    }
```

## 📊 Chat History Analytics

### **Usage Analytics**
```python
class ChatHistoryAnalytics:
    def __init__(self, memory):
        self.memory = memory
    
    async def get_chat_history_usage_stats(self, days=30):
        """Get comprehensive chat history usage statistics."""
        
        # Get all users with chat history
        users_with_history = await self.memory.get_users_with_chat_history(days)
        
        # Calculate usage patterns
        stats = {
            "total_users": len(users_with_history),
            "users_with_multi_session": 0,
            "average_sessions_per_user": 0,
            "average_messages_per_user": 0,
            "most_active_users": [],
            "session_patterns": {},
            "temporal_patterns": {}
        }
        
        total_sessions = 0
        total_messages = 0
        user_activity = []
        
        for user in users_with_history:
            user_sessions = await self.memory.get_sessions_for_soeid(user['soeid'])
            user_messages = sum(s['message_count'] for s in user_sessions)
            
            total_sessions += len(user_sessions)
            total_messages += user_messages
            
            if len(user_sessions) > 1:
                stats["users_with_multi_session"] += 1
            
            user_activity.append({
                "soeid": user['soeid'],
                "sessions": len(user_sessions),
                "messages": user_messages,
                "avg_messages_per_session": user_messages / max(len(user_sessions), 1)
            })
        
        # Calculate averages
        stats["average_sessions_per_user"] = total_sessions / max(len(users_with_history), 1)
        stats["average_messages_per_user"] = total_messages / max(len(users_with_history), 1)
        
        # Most active users
        stats["most_active_users"] = sorted(
            user_activity, 
            key=lambda x: x['messages'], 
            reverse=True
        )[:10]
        
        return stats
    
    async def get_chat_history_effectiveness(self, days=7):
        """Analyze effectiveness of chat history feature."""
        
        # Get conversations with and without chat history
        with_history = await self.memory.get_conversations_with_chat_history(days)
        without_history = await self.memory.get_conversations_without_chat_history(days)
        
        # Calculate effectiveness metrics
        effectiveness = {
            "conversations_with_history": len(with_history),
            "conversations_without_history": len(without_history),
            "usage_rate": len(with_history) / max(len(with_history) + len(without_history), 1),
            "average_response_quality": {
                "with_history": self.calculate_response_quality(with_history),
                "without_history": self.calculate_response_quality(without_history)
            },
            "user_satisfaction": {
                "with_history": self.calculate_user_satisfaction(with_history),
                "without_history": self.calculate_user_satisfaction(without_history)
            }
        }
        
        return effectiveness
```

### **Performance Monitoring**
```python
class ChatHistoryPerformanceMonitor:
    def __init__(self, memory):
        self.memory = memory
        self.metrics = {
            "query_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0
        }
    
    async def monitor_chat_history_query(self, soeid, days, limit):
        """Monitor chat history query performance."""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"chat_history:{soeid}:{days}:{limit}"
            cached_result = await self.get_from_cache(cache_key)
            
            if cached_result:
                self.metrics["cache_hits"] += 1
                query_time = time.time() - start_time
                self.metrics["query_times"].append(query_time)
                return cached_result
            
            # Query database
            result = await self.memory.get_chat_history_by_soeid_and_date(
                soeid=soeid, days=days, limit=limit
            )
            
            # Cache result
            await self.cache_result(cache_key, result, ttl=3600)
            
            self.metrics["cache_misses"] += 1
            query_time = time.time() - start_time
            self.metrics["query_times"].append(query_time)
            self.metrics["total_queries"] += 1
            
            # Log slow queries
            if query_time > 1.0:  # Slower than 1 second
                logger.warning(f"Slow chat history query: {query_time:.2f}s for {soeid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Chat history query failed: {e}")
            raise
    
    def get_performance_report(self):
        """Get chat history performance report."""
        
        if not self.metrics["query_times"]:
            return {"status": "no_data"}
        
        query_times = self.metrics["query_times"]
        total_queries = self.metrics["total_queries"]
        
        return {
            "total_queries": total_queries,
            "cache_hit_rate": self.metrics["cache_hits"] / max(total_queries, 1),
            "average_query_time": sum(query_times) / len(query_times),
            "median_query_time": sorted(query_times)[len(query_times) // 2],
            "95th_percentile": sorted(query_times)[int(len(query_times) * 0.95)],
            "slow_queries": len([t for t in query_times if t > 1.0]),
            "performance_grade": self.calculate_performance_grade(query_times)
        }
    
    def calculate_performance_grade(self, query_times):
        """Calculate performance grade based on query times."""
        
        avg_time = sum(query_times) / len(query_times)
        
        if avg_time < 0.1:
            return "A"  # Excellent
        elif avg_time < 0.5:
            return "B"  # Good
        elif avg_time < 1.0:
            return "C"  # Acceptable
        elif avg_time < 2.0:
            return "D"  # Poor
        else:
            return "F"  # Unacceptable
```

## 🔍 Advanced Features

### **Semantic Chat History Search**
```python
async def semantic_chat_history_search(memory, soeid, query, days=30, limit=20):
    """Search chat history using semantic similarity."""
    
    # Get embeddings for the current query
    from src.models.embedding import EmbeddingModelFactory
    
    embedder = EmbeddingModelFactory.create_model("vertex_ai")
    query_embedding = await embedder.embed_single(query)
    
    # Get chat history
    chat_history = await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid, days=days, limit=limit * 3  # Get more for filtering
    )
    
    if not chat_history:
        return []
    
    # Calculate semantic similarity for each message
    similar_messages = []
    
    for msg in chat_history:
        # Combine query and response for embedding
        msg_text = f"{msg['query']} {msg['response']}"
        msg_embedding = await embedder.embed_single(msg_text)
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(query_embedding, msg_embedding)
        
        if similarity > 0.3:  # Threshold for relevance
            msg['semantic_similarity'] = similarity
            similar_messages.append(msg)
    
    # Sort by similarity and return top results
    similar_messages.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    return similar_messages[:limit]

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
```

### **Topic-Based Chat History**
```python
async def get_topic_based_chat_history(memory, soeid, topic, days=30):
    """Get chat history filtered by topic."""
    
    # Define topic keywords
    topic_keywords = {
        "machine_learning": ["machine learning", "ml", "algorithm", "model", "training", "prediction"],
        "artificial_intelligence": ["artificial intelligence", "ai", "neural network", "deep learning"],
        "data_science": ["data science", "analytics", "statistics", "visualization", "dataset"],
        "programming": ["programming", "code", "python", "javascript", "development", "software"]
    }
    
    keywords = topic_keywords.get(topic.lower(), [topic.lower()])
    
    # Get chat history
    chat_history = await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid, days=days, limit=100
    )
    
    # Filter by topic
    topic_messages = []
    for msg in chat_history:
        msg_text = (msg['query'] + ' ' + msg['response']).lower()
        
        # Check if any topic keywords are present
        if any(keyword in msg_text for keyword in keywords):
            # Calculate topic relevance score
            keyword_count = sum(1 for keyword in keywords if keyword in msg_text)
            msg['topic_relevance'] = keyword_count / len(keywords)
            topic_messages.append(msg)
    
    # Sort by topic relevance and recency
    topic_messages.sort(
        key=lambda x: (x['topic_relevance'], x['timestamp']), 
        reverse=True
    )
    
    return topic_messages
```

### **Conversation Thread Reconstruction**
```python
async def reconstruct_conversation_threads(memory, soeid, days=7):
    """Reconstruct conversation threads from chat history."""
    
    # Get chat history
    chat_history = await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid, days=days, limit=200
    )
    
    if not chat_history:
        return []
    
    # Group by session
    sessions = {}
    for msg in chat_history:
        session_id = msg['session_id']
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(msg)
    
    # Reconstruct threads
    threads = []
    for session_id, messages in sessions.items():
        # Sort messages by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        
        # Create thread summary
        thread = {
            "session_id": session_id,
            "message_count": len(messages),
            "start_time": messages[0]['timestamp'],
            "end_time": messages[-1]['timestamp'],
            "duration_minutes": calculate_session_duration(messages),
            "topics": extract_conversation_topics(messages),
            "summary": create_conversation_summary(messages),
            "messages": messages
        }
        
        threads.append(thread)
    
    # Sort threads by recency
    threads.sort(key=lambda x: x['end_time'], reverse=True)
    
    return threads

def extract_conversation_topics(messages):
    """Extract main topics from conversation messages."""
    
    # Combine all text
    all_text = ' '.join([msg['query'] + ' ' + msg['response'] for msg in messages])
    
    # Simple keyword extraction (in practice, use NLP libraries)
    words = all_text.lower().split()
    word_freq = {}
    
    # Count word frequency (excluding common words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    for word in words:
        if len(word) > 3 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top topics
    topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [topic[0] for topic in topics[:5]]

def create_conversation_summary(messages):
    """Create a summary of the conversation."""
    
    if not messages:
        return ""
    
    # Simple summary (in practice, use summarization models)
    topics = extract_conversation_topics(messages)
    message_count = len(messages)
    
    return f"Conversation with {message_count} messages covering topics: {', '.join(topics[:3])}"
```

## 🚨 Common Issues and Solutions

### **Performance Issues**
```python
# Issue: Slow chat history queries
# Solution: Implement caching and indexing

class OptimizedChatHistory:
    def __init__(self, memory):
        self.memory = memory
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_cached_chat_history(self, soeid, days, limit):
        """Get chat history with caching."""
        
        cache_key = f"{soeid}:{days}:{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        # Query database
        result = await self.memory.get_chat_history_by_soeid_and_date(
            soeid=soeid, days=days, limit=limit
        )
        
        # Cache result
        self.cache[cache_key] = (result, time.time())
        
        return result
```

### **Memory Usage Issues**
```python
# Issue: Large chat history consuming too much memory
# Solution: Implement pagination and truncation

async def get_paginated_chat_history(memory, soeid, days=7, page=1, page_size=20):
    """Get paginated chat history to manage memory usage."""
    
    offset = (page - 1) * page_size
    
    # Get total count first
    total_count = await memory.get_chat_history_count(soeid, days)
    
    # Get paginated results
    chat_history = await memory.get_chat_history_by_soeid_and_date(
        soeid=soeid,
        days=days,
        limit=page_size,
        offset=offset
    )
    
    return {
        "messages": chat_history,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": (total_count + page_size - 1) // page_size,
            "has_next": page * page_size < total_count,
            "has_previous": page > 1
        }
    }
```

### **Privacy and Security**
```python
# Issue: Sensitive information in chat history
# Solution: Implement data sanitization

def sanitize_chat_history(chat_history):
    """Remove sensitive information from chat history."""
    
    sanitized_history = []
    
    for msg in chat_history:
        sanitized_msg = msg.copy()
        
        # Sanitize query and response
        sanitized_msg['query'] = sanitize_text(msg['query'])
        sanitized_msg['response'] = sanitize_text(msg['response'])
        
        # Remove sensitive metadata
        if 'metadata' in sanitized_msg:
            sensitive_fields = ['api_key', 'password', 'token', 'secret']
            for field in sensitive_fields:
                sanitized_msg['metadata'].pop(field, None)
        
        sanitized_history.append(sanitized_msg)
    
    return sanitized_history

def sanitize_text(text):
    """Remove sensitive patterns from text."""
    import re
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    
    # Remove credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
    
    return text
```

## 🎯 Best Practices

### **Chat History Design Guidelines**
1. **User Control**: Always allow users to control chat history usage
2. **Privacy First**: Implement strong data sanitization and retention policies
3. **Performance**: Use caching and pagination for large datasets
4. **Relevance**: Filter history by relevance to current context
5. **Transparency**: Clearly indicate when chat history is being used

### **Configuration Best Practices**
```yaml
# Production-ready chat history configuration
chatbot:
  memory:
    chat_history:
      enabled: true
      default_days: 3              # Conservative default
      max_days: 30                 # Reasonable maximum
      max_messages: 20             # Limit context size
      
      # Performance optimization
      cache_enabled: true
      cache_ttl: 1800              # 30 minutes
      batch_size: 25               # Reasonable batch size
      
      # Privacy protection
      sanitize_content: true
      max_content_length: 500      # Truncate long messages
      exclude_system_messages: true
```

### **API Usage Guidelines**
```python
# Best practice API usage
async def chat_with_history_best_practice(query, soeid, session_id):
    """Example of best practice chat history usage."""
    
    # Start with conservative settings
    use_chat_history = False
    chat_history_days = 3
    
    # Only enable for follow-up questions
    follow_up_indicators = [
        "what we discussed", "previous conversation", "earlier", 
        "before", "last time", "you mentioned"
    ]
    
    if any(indicator in query.lower() for indicator in follow_up_indicators):
        use_chat_history = True
        chat_history_days = 7  # Expand for follow-ups
    
    # Make API request
    response = await chat_api.send_message(
        query=query,
        session_id=session_id,
        use_chat_history=use_chat_history,
        chat_history_days=chat_history_days,
        headers={"soeid": soeid}
    )
    
    return response
```

## 📚 Related Documentation

- **[Memory Systems Overview](./README.md)** - Complete memory architecture
- **[PostgreSQL Setup](./postgresql.md)** - Database configuration
- **[Memory API](./api.md)** - Memory management endpoints
- **[Chatbot Memory](../rag/chatbot/memory.md)** - Chatbot-specific memory usage

## 🚀 Quick Examples

### **Enable Chat History**
```python
# In API request
response = await chat_api.send_message(
    query="What did we discuss about machine learning?",
    use_chat_history=True,
    chat_history_days=7,
    headers={"soeid": "john.doe"}
)
```

### **Get User Chat History**
```bash
# Get chat history via API
curl -X GET "http://localhost:8001/chat/history/john.doe?days=14&limit=20" \
  -H "X-API-Key: test-api-key"
```

---

**Next Steps**: 
- [Set up PostgreSQL](./postgresql.md)
- [Configure Memory API](./api.md)
- [Use Memory Systems](./README.md)
