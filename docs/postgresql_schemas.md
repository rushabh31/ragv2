# PostgreSQL Database Schemas for RAG System

This document provides comprehensive PostgreSQL table schemas required for the RAG (Retrieval-Augmented Generation) system, covering both ingestion and chatbot components.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Ingestion System Tables](#ingestion-system-tables)
- [Chatbot Memory System Tables](#chatbot-memory-system-tables)
- [Performance Indexes](#performance-indexes)
- [Database Setup](#database-setup)
- [Configuration Examples](#configuration-examples)

## 🔧 Prerequisites

### Required PostgreSQL Extensions

```sql
-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation (optional, for unique identifiers)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable JSON operations (usually included by default)
CREATE EXTENSION IF NOT EXISTS btree_gin;
```

### Minimum Requirements
- **PostgreSQL**: 12+ (recommended 15+ for HNSW indexes)
- **pgvector**: 0.5.0+ (for HNSW index support)
- **Memory**: 4GB+ RAM for production workloads
- **Storage**: SSD recommended for vector operations

---

## 🗄️ Ingestion System Tables

### 1. Document Embeddings Table (PgVector Store)

This table stores document chunks and their vector embeddings for similarity search.

```sql
-- Main document embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    soeid TEXT,                                    -- User identifier
    session_id TEXT,                               -- Session identifier
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,                         -- Document chunk text
    metadata JSONB,                                -- Additional metadata
    embedding vector(768),                         -- Vector embedding (768-dim for text-embedding-004)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    page_number INTEGER,                           -- Source page number
    file_name TEXT,                                -- Source file name
    document_id TEXT,                              -- Source document identifier
    chunk_index INTEGER                            -- Chunk position in document
);
```

### 2. Vector Indexes for Similarity Search

```sql
-- HNSW index for fast approximate nearest neighbor search (recommended)
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_hnsw_idx 
ON document_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (for older pgvector versions)
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

### 3. Supporting Indexes for Document Queries

```sql
-- Indexes for efficient filtering and sorting
CREATE INDEX IF NOT EXISTS idx_document_embeddings_soeid 
ON document_embeddings(soeid);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_session_id 
ON document_embeddings(session_id);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id 
ON document_embeddings(document_id);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_upload_timestamp 
ON document_embeddings(upload_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_file_name 
ON document_embeddings(file_name);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_document_embeddings_soeid_timestamp 
ON document_embeddings(soeid, upload_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_chunk 
ON document_embeddings(document_id, chunk_index);

-- GIN index for metadata queries
CREATE INDEX IF NOT EXISTS idx_document_embeddings_metadata 
ON document_embeddings USING GIN (metadata);
```

---

## 💬 Chatbot Memory System Tables

### 1. LangGraph Checkpoints Table

Stores conversation state and history using LangGraph's checkpoint system.

```sql
-- Main checkpoints table (conversation state)
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,                       -- Conversation thread ID
    checkpoint_ns TEXT NOT NULL DEFAULT '',        -- Namespace for checkpoints
    checkpoint_id TEXT NOT NULL,                   -- Unique checkpoint ID
    parent_checkpoint_id TEXT,                     -- Parent checkpoint reference
    type TEXT,                                     -- Checkpoint type
    checkpoint JSONB NOT NULL,                     -- Checkpoint data
    metadata JSONB NOT NULL DEFAULT '{}',          -- User metadata (includes SOEID)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
```

### 2. Checkpoint Writes Table

Stores individual message writes and state changes.

```sql
-- Checkpoint writes table (individual message writes)
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,                       -- Conversation thread ID
    checkpoint_ns TEXT NOT NULL DEFAULT '',        -- Namespace for checkpoints
    checkpoint_id TEXT NOT NULL,                   -- Associated checkpoint ID
    task_id TEXT NOT NULL,                         -- Task identifier
    idx INTEGER NOT NULL,                          -- Write index
    channel TEXT NOT NULL,                         -- Channel name
    type TEXT,                                     -- Write type
    value JSONB,                                   -- Message/state data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

### 3. Checkpoint Blobs Table

Stores large data objects that don't fit in regular checkpoint data.

```sql
-- Checkpoint blobs table (large data storage)
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,                       -- Conversation thread ID
    checkpoint_ns TEXT NOT NULL DEFAULT '',        -- Namespace for checkpoints
    checkpoint_id TEXT NOT NULL,                   -- Associated checkpoint ID
    channel TEXT NOT NULL,                         -- Channel name
    type TEXT NOT NULL,                            -- Blob type
    blob BYTEA,                                    -- Binary data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, type)
);
```

### 4. Checkpoint Migrations Table

Tracks schema migrations for the checkpoint system.

```sql
-- Migrations table (schema versioning)
CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    version INTEGER PRIMARY KEY,                   -- Migration version
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 🚀 Performance Indexes

### 1. Essential Indexes for Checkpoints

```sql
-- Basic performance indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id 
ON checkpoints(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at 
ON checkpoints(created_at);

CREATE INDEX IF NOT EXISTS idx_checkpoints_metadata_soeid 
ON checkpoints USING GIN ((metadata->>'soeid'));

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id 
ON checkpoint_writes(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_created_at 
ON checkpoint_writes(created_at);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_channel 
ON checkpoint_writes(channel);
```

### 2. Composite Indexes for Common Queries

```sql
-- Composite indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_created 
ON checkpoints(thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_writes_thread_checkpoint 
ON checkpoint_writes(thread_id, checkpoint_id);

-- SOEID-based queries for cross-session history
CREATE INDEX IF NOT EXISTS idx_checkpoints_soeid_created 
ON checkpoints((metadata->>'soeid'), created_at DESC) 
WHERE metadata->>'soeid' IS NOT NULL;
```

### 3. Temporal and Optimization Indexes

```sql
-- Temporal queries for date-based filtering
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_date 
ON checkpoints(DATE(created_at));

-- Partial indexes for active sessions (performance optimization)
CREATE INDEX IF NOT EXISTS idx_checkpoints_recent 
ON checkpoints(thread_id, created_at DESC) 
WHERE created_at > NOW() - INTERVAL '30 days';

-- Partial index for non-null SOEID values
CREATE INDEX IF NOT EXISTS idx_checkpoints_soeid_notnull 
ON checkpoints(metadata->>'soeid', created_at DESC) 
WHERE metadata->>'soeid' IS NOT NULL;
```

---

## 🛠️ Database Setup

### 1. Complete Database Initialization Script

```sql
-- Create database (run as superuser)
CREATE DATABASE langgraph_rag_db;

-- Connect to the database
\c langgraph_rag_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create ingestion tables
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    soeid TEXT,
    session_id TEXT,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    page_number INTEGER,
    file_name TEXT,
    document_id TEXT,
    chunk_index INTEGER
);

-- Create chatbot memory tables
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, type)
);

CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create all performance indexes (see sections above)
-- ... (include all indexes from previous sections)
```

### 2. User and Permissions Setup

```sql
-- Create application user
CREATE USER rag_app_user WITH PASSWORD 'secure_password_here';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE langgraph_rag_db TO rag_app_user;
GRANT USAGE ON SCHEMA public TO rag_app_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_app_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_app_user;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
GRANT ALL PRIVILEGES ON TABLES TO rag_app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
GRANT ALL PRIVILEGES ON SEQUENCES TO rag_app_user;
```

---

## 💭 Memory System Options

The RAG system supports multiple memory implementations, each designed for different use cases:

### 🏪 **PostgreSQL Persistent Memory (`langgraph_checkpoint` + `postgres`)**
**Best for: Production environments, enterprise applications**

- **Persistence**: Full conversation history stored in PostgreSQL
- **Cross-session**: Users can reference conversations from other sessions
- **Scalability**: Handles high-volume conversations with proper indexing
- **SOEID Support**: Full user tracking across all sessions
- **Performance**: Optimized with indexes for fast retrieval
- **Backup**: Standard PostgreSQL backup/restore procedures

### 🧠 **In-Memory Checkpointing (`langgraph_checkpoint` + `in_memory`)**
**Best for: Development, testing, temporary deployments**

- **Persistence**: Conversation history lost on application restart
- **Performance**: Fastest memory operations (no database I/O)
- **Resource Usage**: Uses application memory for storage
- **Limitations**: Not suitable for production or multi-instance deployments
- **Use Cases**: Local development, unit testing, proof-of-concepts

### 🚫 **No Checkpointing (`no_checkpoint`)**
**Best for: Stateless APIs, privacy-sensitive applications, high-performance scenarios**

- **Persistence**: No conversation history stored anywhere
- **Privacy**: Maximum privacy - no data retention
- **Performance**: Lowest latency and resource usage
- **Stateless**: Each conversation is independent
- **Use Cases**: Public APIs, privacy-compliant applications, high-volume stateless interactions
- **Benefits**: No database requirements, infinite scalability

### 📝 **Simple Memory (`simple`)**
**Best for: Legacy compatibility, basic applications**

- **Persistence**: In-memory storage with configurable history limit
- **Limitations**: Basic functionality, no cross-session support
- **Use Cases**: Backward compatibility, simple chatbots

---

## ⚙️ Configuration Examples

### 1. Ingestion Configuration (config.yaml)

```yaml
vector_store:
  provider: "pgvector"
  config:
    connection_string: "postgresql://rag_app_user:password@localhost:5432/langgraph_rag_db"
    table_name: "document_embeddings"
    dimension: 768
    index_method: "hnsw"  # or "ivfflat"
    distance_metric: "cosine"
```

### 2. Chatbot Memory Configuration (config.yaml)

#### Option A: PostgreSQL Persistent Memory (Recommended for Production)
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    postgres:
      connection_string: "postgresql://rag_app_user:password@localhost:5432/langgraph_rag_db"
```

#### Option B: In-Memory Checkpointing (Development/Testing)
```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "in_memory"
```

#### Option C: No Checkpointing - Stateless Conversations (High Performance/Privacy)
```yaml
chatbot:
  memory:
    type: "no_checkpoint"
    enabled: true  # Set to false to completely disable memory operations
```

#### Option D: Simple Memory (Legacy)
```yaml
chatbot:
  memory:
    type: "simple"
    max_history: 10
```

### 3. Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql://rag_app_user:password@localhost:5432/langgraph_rag_db

# Connection pool settings
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
DB_POOL_MAX_QUERIES=50000
DB_POOL_MAX_INACTIVE_CONNECTION_LIFETIME=300

# Performance settings
DB_QUERY_TIMEOUT=30
DB_COMMAND_TIMEOUT=60
```

---

## 📊 Table Relationships

```mermaid
erDiagram
    document_embeddings {
        serial id PK
        text chunk_id UK
        text soeid
        text session_id
        timestamp upload_timestamp
        text content
        jsonb metadata
        vector embedding
        timestamp created_at
        integer page_number
        text file_name
        text document_id
        integer chunk_index
    }
    
    checkpoints {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text parent_checkpoint_id
        text type
        jsonb checkpoint
        jsonb metadata
        timestamptz created_at
    }
    
    checkpoint_writes {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text task_id PK
        integer idx PK
        text channel
        text type
        jsonb value
        timestamptz created_at
    }
    
    checkpoint_blobs {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text channel PK
        text type PK
        bytea blob
        timestamptz created_at
    }
    
    checkpoint_migrations {
        integer version PK
        timestamptz applied_at
    }
    
    checkpoints ||--o{ checkpoint_writes : "has"
    checkpoints ||--o{ checkpoint_blobs : "has"
```

---

## 🔍 Query Examples

### 1. Vector Similarity Search

```sql
-- Find similar documents using cosine similarity
SELECT 
    chunk_id,
    content,
    file_name,
    page_number,
    1 - (embedding <=> $1::vector) as similarity_score
FROM document_embeddings 
WHERE soeid = $2
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

### 2. Chat History Retrieval

```sql
-- Get chat history for a user within date range
SELECT 
    thread_id,
    checkpoint,
    metadata,
    created_at
FROM checkpoints 
WHERE metadata->>'soeid' = $1 
  AND created_at >= $2 
  AND created_at <= $3
ORDER BY created_at DESC
LIMIT 50;
```

### 3. Document Statistics

```sql
-- Get document ingestion statistics
SELECT 
    soeid,
    COUNT(*) as total_chunks,
    COUNT(DISTINCT document_id) as total_documents,
    COUNT(DISTINCT file_name) as total_files,
    MIN(upload_timestamp) as first_upload,
    MAX(upload_timestamp) as last_upload
FROM document_embeddings 
GROUP BY soeid
ORDER BY total_chunks DESC;
```

---

## 🚨 Important Notes

1. **Vector Dimensions**: Ensure the vector dimension (768) matches your embedding model
2. **Index Selection**: Use HNSW for better performance with pgvector 0.5.0+
3. **Connection Pooling**: Configure appropriate connection pool sizes for your workload
4. **Backup Strategy**: Implement regular backups, especially for checkpoint tables
5. **Monitoring**: Monitor query performance and adjust indexes as needed
6. **Security**: Use strong passwords and limit database access appropriately

---

## 📚 Additional Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [LangGraph Checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Vector Database Best Practices](https://www.pinecone.io/learn/vector-database/)
