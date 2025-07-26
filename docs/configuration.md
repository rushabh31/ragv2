# Configuration Guide

## 🎯 Overview

The RAG system uses YAML configuration files to customize behavior without code changes. This guide covers all configuration options with practical examples using Vertex AI as the primary provider.

## 📁 Configuration File Structure

```
controlsgenai/
├── config.yaml                           # Main system configuration
├── examples/rag/ingestion/config.yaml    # Ingestion service config
├── examples/rag/chatbot/config.yaml      # Chatbot service config
└── .env                                  # Environment variables
```

## 🔧 Main Configuration (`config.yaml`)

### **Complete Configuration Template**

```yaml
# Main RAG System Configuration
ingestion:
  parsing:
    provider: "vision_parser"
    vision:
      model: "gemini-1.5-pro-002"
      region: "us-central1"
      max_pages: 100
      max_concurrent_pages: 5
  
  chunking:
    provider: "semantic"
    semantic:
      chunk_size: 1000
      overlap: 200
      use_llm_boundary: true
  
  embedding:
    provider: "vertex"
    vertex:
      model: "text-embedding-004"
      batch_size: 100
      region: "us-central1"
  
  vector_store:
    provider: "faiss"
    faiss:
      index_type: "HNSW"
      dimension: 768
      metric: "cosine"

chatbot:
  generation:
    provider: "vertex"
    vertex:
      model_name: "gemini-1.5-pro-002"
      max_tokens: 1000
      temperature: 0.7
      region: "us-central1"
  
  retrieval:
    provider: "vector"
    vector:
      top_k: 10
      score_threshold: 0.7
  
  reranking:
    provider: "custom"
    custom:
      top_k: 5
  
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    postgres:
      connection_string: "postgresql://user:pass@localhost:5432/ragdb"

# Vision processing configuration
vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"
    region: "us-central1"

# Generation model configuration
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    max_tokens: 1000
    temperature: 0.7

# Embedding model configuration
embedding:
  provider: "vertex_ai"
  config:
    model: "text-embedding-004"
    batch_size: 100
```

## 🏭 Provider Configuration

### **Vertex AI Configuration (Recommended)**

```yaml
# Generation with Vertex AI Gemini
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"    # Latest Gemini Pro model
    max_tokens: 1000                     # Response length limit
    temperature: 0.7                     # Creativity (0.0-1.0)
    top_p: 0.9                          # Nucleus sampling
    top_k: 40                           # Top-k sampling
    region: "us-central1"               # GCP region

# Embedding with Vertex AI
embedding:
  provider: "vertex_ai"
  config:
    model: "text-embedding-004"         # Latest embedding model
    batch_size: 100                     # Batch processing size
    region: "us-central1"               # GCP region

# Vision with Vertex AI
vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"        # Vision-capable model
    region: "us-central1"               # GCP region
```

### **OpenAI Configuration (Alternative)**

```yaml
# Generation with OpenAI
generation:
  provider: "openai"
  config:
    model: "gpt-4"                      # GPT-4 model
    max_tokens: 1000                    # Response length
    temperature: 0.7                    # Creativity
    api_base: "https://api.openai.com/v1"

# Embedding with OpenAI
embedding:
  provider: "openai_universal"
  config:
    model: "text-embedding-3-large"    # Latest embedding model
    batch_size: 100                    # Batch size
```

## 📥 Ingestion Configuration

### **Document Parsing**

```yaml
ingestion:
  parsing:
    # Vision Parser (OCR) - Best for PDFs with images/scans
    provider: "vision_parser"
    vision:
      model: "gemini-1.5-pro-002"
      max_pages: 100                    # Maximum pages to process
      max_concurrent_pages: 5           # Parallel processing
      prompt_template: "Extract all text content from this document image."
    
    # Simple Text Parser - Fast for text-only documents
    # provider: "simple_text"
    # simple_text:
    #   encoding: "utf-8"
```

### **Text Chunking**

```yaml
ingestion:
  chunking:
    # Semantic Chunking - AI-powered boundary detection
    provider: "semantic"
    semantic:
      chunk_size: 1000                  # Target chunk size
      overlap: 200                      # Overlap between chunks
      use_llm_boundary: true            # Use AI for boundaries
      min_chunk_size: 100               # Minimum chunk size
      max_chunk_size: 2000              # Maximum chunk size
    
    # Sliding Window - Simple overlap-based chunking
    # provider: "sliding_window"
    # sliding_window:
    #   chunk_size: 1000
    #   overlap: 200
```

### **Vector Storage**

```yaml
ingestion:
  vector_store:
    # FAISS - Fast similarity search
    provider: "faiss"
    faiss:
      index_type: "HNSW"               # Index algorithm
      dimension: 768                   # Embedding dimension
      metric: "cosine"                 # Distance metric
      ef_construction: 200             # Build-time parameter
      ef_search: 100                   # Search-time parameter
      max_connections: 16              # Graph connectivity
```

## 🤖 Chatbot Configuration

### **Response Generation**

```yaml
chatbot:
  generation:
    provider: "vertex"
    vertex:
      model_name: "gemini-1.5-pro-002"
      max_tokens: 1000                  # Response length
      temperature: 0.7                  # Creativity (0.0-1.0)
      top_p: 0.9                       # Nucleus sampling
      top_k: 40                        # Top-k sampling
      system_prompt: |                 # Custom system prompt
        You are a helpful AI assistant that answers questions
        based on the provided context. Be accurate and concise.
```

### **Document Retrieval**

```yaml
chatbot:
  retrieval:
    provider: "vector"
    vector:
      top_k: 10                        # Number of documents to retrieve
      score_threshold: 0.7             # Minimum similarity score
      filter_metadata: {}              # Metadata filters
      search_type: "similarity"        # Search algorithm
```

### **Result Reranking**

```yaml
chatbot:
  reranking:
    provider: "custom"
    custom:
      top_k: 5                         # Final number of documents
      boost_recent: true               # Boost recent documents
      boost_keywords: ["important"]    # Boost specific keywords
```

## 🧠 Memory Configuration

### **PostgreSQL Memory (Production)**

```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    postgres:
      connection_string: "postgresql://user:pass@localhost:5432/ragdb"
      pool_size: 10                    # Connection pool size
      max_overflow: 20                 # Additional connections
      pool_timeout: 30                 # Connection timeout
```

### **In-Memory Storage (Development)**

```yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "in_memory"
    max_sessions: 1000                 # Maximum stored sessions
    cleanup_interval: 3600             # Cleanup interval (seconds)
```

### **Simple Memory (Basic)**

```yaml
chatbot:
  memory:
    type: "simple"
    max_history: 10                    # Messages to remember
    persist: false                     # Don't persist to disk
```

## 🌍 Environment Variables (`.env`)

### **Required Variables**

```env
# Google Cloud Project
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Authentication (OAuth2)
COIN_CONSUMER_ENDPOINT_URL=https://your-oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# PostgreSQL Database
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/ragdb
```

### **Optional Variables**

```env
# OpenAI (if using OpenAI provider)
OPENAI_API_KEY=your-openai-api-key

# Azure OpenAI (if using Azure provider)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
MAX_WORKERS=4
TIMEOUT_SECONDS=300
```

## ⚡ Performance Tuning

### **High-Performance Configuration**

```yaml
# For large document processing
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 10         # Increase parallel processing
  embedding:
    vertex:
      batch_size: 200                  # Larger batches
  vector_store:
    faiss:
      ef_construction: 400             # Better index quality
      ef_search: 200                   # More thorough search

# For fast response times
chatbot:
  retrieval:
    vector:
      top_k: 5                         # Fewer documents
  reranking:
    custom:
      top_k: 3                         # Fewer final results
  generation:
    vertex:
      max_tokens: 500                  # Shorter responses
```

### **Memory-Optimized Configuration**

```yaml
# For limited memory environments
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 2          # Reduce parallelism
  embedding:
    vertex:
      batch_size: 20                   # Smaller batches
  chunking:
    semantic:
      chunk_size: 500                  # Smaller chunks

chatbot:
  memory:
    type: "simple"                     # Use simple memory
    max_history: 5                     # Limited history
```

## 🔧 Service-Specific Configuration

### **Ingestion Service (`examples/rag/ingestion/config.yaml`)**

```yaml
# Ingestion-specific settings
parser:
  provider: "vision_parser"
  config:
    model: "gemini-1.5-pro-002"
    max_pages: 50
    max_concurrent_pages: 5

chunker:
  provider: "semantic"
  config:
    chunk_size: 1000
    overlap: 200

embedder:
  provider: "vertex"
  config:
    model: "text-embedding-004"
    batch_size: 100

vector_store:
  provider: "faiss"
  config:
    index_type: "HNSW"
    dimension: 768
    storage_path: "./vector_storage"

api:
  host: "0.0.0.0"
  port: 8000
  api_key: "test-api-key"
  cors_origins: ["*"]
  max_file_size: 50000000              # 50MB limit
```

### **Chatbot Service (`examples/rag/chatbot/config.yaml`)**

```yaml
# Chatbot-specific settings
generator:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    max_tokens: 1000
    temperature: 0.7

retriever:
  provider: "vector"
  config:
    vector_store_path: "../ingestion/vector_storage"
    top_k: 10
    score_threshold: 0.7

reranker:
  provider: "custom"
  config:
    top_k: 5

memory:
  type: "langgraph_checkpoint"
  store_type: "postgres"
  postgres:
    connection_string: "postgresql://user:pass@localhost:5432/ragdb"

api:
  host: "0.0.0.0"
  port: 8001
  api_key: "test-api-key"
  cors_origins: ["*"]
  session_timeout: 3600                # 1 hour
```

## 🚨 Troubleshooting Configuration

### **Validation Commands**

```bash
# Test configuration validity
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Configuration is valid YAML')
"

# Test environment variables
python -c "
import os
required = ['PROJECT_ID', 'VERTEXAI_API_ENDPOINT']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'❌ Missing: {missing}')
else:
    print('✅ Environment variables are set')
"

# Test provider authentication
python -c "
from src.models.generation import VertexGenAI
model = VertexGenAI()
status = model.get_auth_health_status()
print(f'Auth status: {status}')
"
```

### **Common Configuration Errors**

**Error**: `Provider 'vertex' not found`
```yaml
# ❌ Wrong
generation:
  provider: "vertex_ai"  # Should be "vertex"

# ✅ Correct
generation:
  provider: "vertex"
```

**Error**: `Model not found`
```yaml
# ❌ Wrong
generation:
  config:
    model: "gemini-1.5-pro-002"  # Should be "model_name"

# ✅ Correct
generation:
  config:
    model_name: "gemini-1.5-pro-002"
```

**Error**: `Database connection failed`
```env
# ❌ Wrong format
POSTGRES_CONNECTION_STRING=localhost:5432/ragdb

# ✅ Correct format
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/ragdb
```

## 📚 Configuration Examples

### **Development Environment**

```yaml
# Fast setup for development
ingestion:
  parsing:
    provider: "simple_text"            # Fast text parsing
  chunking:
    provider: "sliding_window"         # Simple chunking
  embedding:
    provider: "vertex"
  vector_store:
    provider: "faiss"

chatbot:
  generation:
    provider: "vertex"
  memory:
    type: "simple"                     # No database required
```

### **Production Environment**

```yaml
# Robust setup for production
ingestion:
  parsing:
    provider: "vision_parser"          # Full OCR capability
    vision:
      max_concurrent_pages: 8
  chunking:
    provider: "semantic"               # AI-powered chunking
  embedding:
    provider: "vertex"
    vertex:
      batch_size: 200
  vector_store:
    provider: "faiss"
    faiss:
      ef_construction: 400

chatbot:
  generation:
    provider: "vertex"
    vertex:
      temperature: 0.3                 # More conservative
  memory:
    type: "langgraph_checkpoint"       # Persistent memory
    store_type: "postgres"
```

---

**Next Steps**: 
- [Ingestion Pipeline Configuration](./rag/ingestion/README.md)
- [Chatbot Service Configuration](./rag/chatbot/README.md)
- [Complete Walkthrough](./tutorials/complete-walkthrough.md)
