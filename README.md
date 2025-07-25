# RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with flexible AI providers, efficient caching, and scalable architecture.

## Features

- **Multiple AI Provider Support**
  - Generation: Vertex AI (Google) and Groq
  - Embeddings: Vertex AI (Google) and Sentence Transformers (local)
  - Runtime provider switching through configuration

- **Efficient Caching**
  - Local in-memory caching with LRU eviction and TTL expiration
  - Optional Redis caching for distributed deployments
  - Unified cache interface abstraction

- **Comprehensive API**
  - Document ingestion and processing
  - Conversation management
  - Vector search and retrieval
  - Rate limiting and authentication

- **Production-Ready Architecture**
  - Asynchronous processing
  - Middleware support for CORS, authentication, rate limiting
  - Proper error handling and logging

## Prerequisites

- Python 3.9+
- API keys for chosen AI providers (Vertex AI and/or Groq)
- (Optional) Redis server for distributed caching

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Configuration

The system uses a YAML configuration file with the following major sections:

### Basic Configuration

Create a `config.yaml` file based on the provided `config_sample.yaml`:

```yaml
# API settings
api:
  chatbot:
    host: 0.0.0.0
    port: 8000
    cors_origins:
      - "*"
    require_api_key: true
    api_keys:
      - name: default
        key: "${API_KEY}"
        roles:
          - user
      - name: admin
        key: "${ADMIN_API_KEY}"
        roles:
          - user
          - admin
    rate_limiting:
      enabled: true
      provider: local
      requests: 100
      period_seconds: 60

# Cache settings
cache:
  enabled: true
  provider: local  # Options: "local" or "redis"
  default_ttl_seconds: 3600
  max_size: 10000
  cleanup_interval_seconds: 300

# Ingestion API settings
ingestion:
  chunking:
    chunk_size: 1000
    chunk_overlap: 200
  embedding:
    provider: sentence_transformer  # Options: "vertex" or "sentence_transformer"
    model_name: all-MiniLM-L6-v2
    dimension: 384
    batch_size: 8

# Chatbot settings
chatbot:
  generation:
    provider: groq  # Options: "vertex" or "groq"
    model_name: llama3-8b-8192
    temperature: 0.7
    max_tokens: 1024
    top_p: 1.0
  retrieval:
    retriever_type: vector
    top_k: 5
    min_score: 0.7
  reranking:
    enabled: true
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_k: 3
```

### Environment Variables

Create a `.env` file with the necessary API keys:

```
API_KEY=your_api_key_here
ADMIN_API_KEY=your_admin_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-credentials.json
```

### Provider Configuration

#### Generation Providers

The system supports the following generation providers:

1. **Vertex AI (Google)**
   ```yaml
   chatbot:
     generation:
       provider: vertex
       model_name: gemini-1.0-pro
   ```

2. **Groq**
   ```yaml
   chatbot:
     generation:
       provider: groq
       model_name: llama3-8b-8192
   ```

#### Embedding Providers

The system supports the following embedding providers:

1. **Vertex AI (Google)**
   ```yaml
   ingestion:
     embedding:
       provider: vertex
       model_name: textembedding-gecko@001
   ```

2. **Sentence Transformers (Local)**
   ```yaml
   ingestion:
     embedding:
       provider: sentence_transformer
       model_name: all-MiniLM-L6-v2
   ```

#### Cache Providers

The system supports the following cache providers:

1. **Local Cache (Default)**
   ```yaml
   cache:
     provider: local
     max_size: 10000
     cleanup_interval_seconds: 300
   ```

2. **Redis Cache**
   ```yaml
   cache:
     provider: redis
     redis_url: "${REDIS_URL}"
   ```

## Usage

### Running the Services

1. Start the ingestion API:
   ```bash
   python -m controlsgenai.funcs.rag.src.ingestion.api.main
   ```

2. Start the chatbot API:
   ```bash
   python -m controlsgenai.funcs.rag.src.chatbot.api.main
   ```

### API Endpoints

#### Ingestion API

- **Upload Document**
  ```
  POST /documents/upload
  ```
  Upload and process a document for ingestion into the vector store.

- **Batch Upload**
  ```
  POST /documents/batch
  ```
  Upload and process multiple documents in a single request.

- **Delete Document**
  ```
  DELETE /documents/{document_id}
  ```
  Remove a document from the system.

#### Chatbot API

- **Process Chat**
  ```
  POST /chat
  ```
  Process a chat message with optional retrieval and memory.

- **Create Session**
  ```
  POST /sessions
  ```
  Create a new chat session.

- **Get Session History**
  ```
  GET /sessions/{session_id}
  ```
  Retrieve the history of a chat session.

### Example API Usage

#### Document Upload
```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@path/to/document.pdf" \
  -F "metadata={\"title\": \"My Document\", \"author\": \"John Doe\"}"
```

#### Chat Request
```bash
curl -X POST http://localhost:8001/chat \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "query": "What is RAG?",
    "session_id": "session456",
    "use_retrieval": true,
    "use_history": true
  }'
```

## Architecture

### Components

1. **API Layer**
   - Ingestion API for document processing
   - Chatbot API for conversation handling

2. **Service Layer**
   - Ingestion Service for document processing and embedding
   - Chatbot Service for retrieval and generation

3. **Core Components**
   - Embedders (Vertex AI, Sentence Transformer)
   - Generators (Vertex AI, Groq)
   - Vector Store for efficient retrieval
   - Memory System for conversation context

4. **Infrastructure**
   - Cache Manager (Local or Redis)
   - Middleware Factory for API security and performance

### Flow Diagrams

#### Document Ingestion Flow
```
Document → Chunking → Embedding → Vector Store → Metadata Storage
```

#### Query Processing Flow
```
Query → Embedding → Vector Retrieval → Reranking → Context Assembly → Generation → Response
```

## Development

### Adding a New Provider

1. Create a new provider implementation in the appropriate directory
2. Implement the required interface (generator or embedder)
3. Update the factory class to support the new provider
4. Add configuration options to the sample config

### Testing

Run the test suite with:
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in your `.env` file
   - Check that the environment variables are properly loaded

2. **Model Availability**
   - Verify the selected models are available in your provider account
   - Check for any usage quotas or limitations

3. **Memory Usage**
   - Monitor memory usage when using local cache with large `max_size` values
   - Adjust cache settings based on available system resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
