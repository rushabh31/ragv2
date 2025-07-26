# Chatbot Service Example

This example demonstrates how to use the controlsgenai RAG library to build a conversational AI chatbot with document retrieval capabilities.

## Overview

The chatbot service provides REST API endpoints for:
- Natural language conversations with context memory
- Document retrieval from vector databases
- Re-ranking of retrieved documents for better relevance
- Response generation using large language models
- Session management and conversation history

## Quick Start

### 1. Install Dependencies

Make sure you have the controlsgenai package installed:

```bash
# From the project root
pip install -e .
```

### 2. Configure Environment Variables

Set up your authentication and API keys in your environment:

```bash
# Universal Authentication (Required)
export COIN_CONSUMER_ENDPOINT_URL="https://your-oauth-server/oauth2/token"
export COIN_CONSUMER_CLIENT_ID="your-client-id"
export COIN_CONSUMER_CLIENT_SECRET="your-client-secret"
export COIN_CONSUMER_SCOPE="https://www.googleapis.com/auth/cloud-platform"
export PROJECT_ID="your-gcp-project-id"
export VERTEXAI_API_ENDPOINT="us-central1-aiplatform.googleapis.com"

# PostgreSQL for Memory (Optional - for persistent memory)
export POSTGRES_CONNECTION_STRING="postgresql://username:password@localhost:5432/langgraph_db"

# API Authentication (Optional)
export API_KEY="your_api_key_here"  # For API authentication
export ADMIN_API_KEY="your_admin_key_here"  # For admin access
```

### 3. Start the Service

From this directory (`examples/rag/chatbot/`):

```bash
python run_chatbot.py
```

The service will start on `http://localhost:8001`

### 4. Test the Service

#### Basic Chat (No Retrieval)

```bash
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{
    "query": "Hello, how are you?",
    "session_id": "test_session",
    "retrieval_enabled": false
  }'
```

#### Chat with Document Retrieval

First, make sure you have documents ingested using the [Ingestion Service](../ingestion/README.md).

```bash
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{
    "query": "What is machine learning?",
    "session_id": "test_session",
    "retrieval_enabled": true
  }'
```

Expected response:
```json
{
  "session_id": "test_session",
  "soeid": "test-user",
  "response": "Based on the documents, machine learning is...",
  "created_at": "2025-07-25T15:00:00",
  "query": "What is machine learning?",
  "retrieved_documents": [
    {
      "content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {...},
      "score": 0.85
    }
  ],
  "metadata": {
    "metrics": {
      "retrieval": {"time_seconds": 0.5, "document_count": 3},
      "reranking": {"time_seconds": 0.2, "documents_reranked": 3},
      "generation": {"time_seconds": 1.2}
    }
  }
}
```

#### Get Chat History

```bash
curl -X GET 'http://localhost:8001/chat/history/test_session' \
  -H 'soeid: test-user'
```

## Configuration

The service uses `config.yaml` in this directory. All configuration parameters are verified from the source code:

### Generation Configuration
- **Default Provider**: `vertex` (Vertex AI Gemini)
- **Available Providers**: `vertex`, `anthropic_vertex`, `openai`, `azure_openai`
- **Model**: `gemini-1.5-pro-002`
- **Temperature**: 0.7 (configurable in `chatbot.generation.config.temperature`)
- **Max Tokens**: 2048 (configurable in `chatbot.generation.config.max_tokens`)
- **Prompt Template**: `./templates/rag_prompt.jinja2`

### Retrieval Configuration
- **Vector Store**: Must match ingestion config (default: `faiss`)
- **Embedding Provider**: Must match ingestion config (default: `vertex`)
- **Embedding Model**: `text-embedding-004` (768 dimensions)
- **Top K**: 10 documents retrieved (configurable in `chatbot.retrieval.top_k`)
- **Similarity Threshold**: 0.7 (configurable in `chatbot.retrieval.similarity_threshold`)

### Reranking Configuration
- **Default Type**: `custom` (TF-IDF based)
- **Available Types**: `custom`, `cross_encoder`
- **Top K After Reranking**: 5 documents (configurable in `chatbot.reranking.config.top_k`)
- **Boost Recent**: Enabled (configurable in `chatbot.reranking.config.boost_recent`)

### Memory Configuration
- **Default Type**: `langgraph_checkpoint` (LangGraph with PostgreSQL)
- **Available Types**: `simple`, `mem0`, `langgraph`, `langgraph_checkpoint`
- **Store Type**: `postgres` or `in_memory`
- **Max History**: 10 messages (configurable in `chatbot.memory.max_history`)
- **PostgreSQL**: Connection string required for persistent memory

### API Configuration
- **Host**: `0.0.0.0`
- **Port**: 8001 (different from ingestion port 8000)
- **CORS**: Enabled for all origins
- **API Key**: Required by default

## API Endpoints

### Send Chat Message
- **POST** `/chat/message/json`
- **Headers**: `soeid: <user_id>`, `Content-Type: application/json`
- **Body**: JSON with `query`, `session_id`, `retrieval_enabled`

### Get Chat History
- **GET** `/chat/history/{session_id}`
- **Headers**: `soeid: <user_id>`

### Health Check
- **GET** `/health`

## Features

### 1. Document Retrieval
- Semantic search using vector embeddings
- Configurable similarity thresholds
- Support for multiple vector stores (pgvector, FAISS, ChromaDB)

### 2. Re-ranking
- **Custom Reranker**: Uses TF-IDF, keyword matching, and position scoring
- **Cross-Encoder Reranker**: Uses transformer models (optional)
- Configurable weights and thresholds

### 3. Memory Management
- **LangGraph Memory**: Advanced conversation memory with persistence
- **Simple Memory**: Basic in-memory conversation history
- Session-based conversation tracking

### 4. Response Generation
- **Groq**: Fast inference with Llama models
- **OpenAI**: GPT models (configurable)
- **Vertex AI**: Google's PaLM models (configurable)

## Advanced Usage

### Custom Prompts

The system uses Jinja2 templates for prompts. Default template location:
```
config/prompts/chatbot/rag_prompt.jinja2
```

### Retrieval Configuration

Adjust retrieval settings in `config.yaml`:

```yaml
chatbot:
  retrieval:
    top_k: 5  # Number of documents to retrieve
    similarity_threshold: 0.7  # Minimum similarity score
  reranking:
    enabled: true
    type: custom  # or "cross_encoder"
    top_k: 3  # Number of documents after reranking
```

### Memory Configuration

Configure conversation memory:

```yaml
chatbot:
  memory:
    type: langgraph  # or "simple"
    store_type: in_memory  # or "postgres"
    max_history: 20  # Number of messages to remember
```

## Testing Scenarios

### 1. Basic Conversation
```bash
# Test basic chat without retrieval
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{"query": "Hello!", "session_id": "test", "retrieval_enabled": false}'
```

### 2. Knowledge-Based Questions
```bash
# Ask about ingested documents
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{"query": "What topics are covered in the documents?", "session_id": "test", "retrieval_enabled": true}'
```

### 3. Follow-up Questions
```bash
# Test conversation memory
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{"query": "Can you explain that in more detail?", "session_id": "test", "retrieval_enabled": true}'
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill existing processes
   lsof -ti:8001 | xargs kill -9
   ```

2. **No documents retrieved**
   - Ensure documents are ingested via the ingestion service
   - Check vector store connection
   - Lower similarity threshold in config

3. **LLM generation errors**
   - Verify GROQ_API_KEY is set correctly
   - Check API rate limits
   - Try switching to a different provider in config

4. **Memory errors**
   - Switch to simple memory type in config
   - Check PostgreSQL connection if using postgres store type

### Performance Tuning

1. **Retrieval Speed**
   - Use FAISS for faster in-memory search
   - Reduce `top_k` values
   - Use HNSW index for pgvector

2. **Generation Speed**
   - Use Groq for fastest inference
   - Reduce `max_output_tokens`
   - Lower `temperature` for more focused responses

## Development

To modify the chatbot service:

1. **Core Logic**: Edit files in `../../../src/rag/chatbot/`
2. **API Layer**: Edit files in `api/`
3. **Configuration**: Modify `config.yaml`
4. **Prompts**: Edit templates in `../../../config/prompts/chatbot/`

## Integration

This chatbot service is designed to work with:
- [Ingestion Service](../ingestion/README.md) for document processing
- PostgreSQL + pgvector for vector storage
- Groq API for fast LLM inference
- Any vector store supported by the core library

## Next Steps

1. **Deploy**: Use the configuration as a template for production deployment
2. **Customize**: Modify prompts, add new endpoints, or integrate with other systems
3. **Scale**: Add load balancing, caching, and monitoring for production use
