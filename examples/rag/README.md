# ControlsGenAI RAG Examples

This directory contains practical examples demonstrating how to use the controlsgenai RAG library to build production-ready applications.

## Overview

The examples showcase a complete Retrieval-Augmented Generation (RAG) system with two main services:

1. **[Ingestion Service](ingestion/)** - Document processing and vector storage
2. **[Chatbot Service](chatbot/)** - Conversational AI with document retrieval

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Ingestion API   │───▶│  Vector Store   │
│  (.txt, .pdf)   │    │  (Port 8000)     │    │  (PostgreSQL)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│    Users        │◀───│   Chatbot API    │◀────────────┘
│  (Chat Queries) │    │  (Port 8001)     │
└─────────────────┘    └──────────────────┘
```

## Quick Start

### Prerequisites

1. **Python Environment**
   ```bash
   # Install the controlsgenai package
   pip install -e .
   ```

2. **Environment Variables**
   ```bash
   export API_KEY="optional_api_key"
   export ADMIN_API_KEY="optional_admin_key"
   ```

3. **PostgreSQL with pgvector** (Optional - can use FAISS instead)
   ```bash
   # Install PostgreSQL and pgvector extension
   # Or modify config.yaml to use FAISS vector store
   ```

### Running the Complete System

#### 1. Start Ingestion Service
```bash
cd examples/rag/ingestion/
python run_ingestion.py
```
Service available at: `http://localhost:8000`

#### 2. Start Chatbot Service
```bash
cd examples/rag/chatbot/
python run_chatbot.py
```
Service available at: `http://localhost:8001`

#### 3. Test the System

**Upload a document:**
```bash
echo "Machine learning is a subset of artificial intelligence." > test_doc.txt

curl -X POST 'http://localhost:8000/ingest/upload' \
  -H 'soeid: test-user' \
  -F 'file=@test_doc.txt' \
  -F 'options={"file_name": "ml_guide.txt", "session_id": "test"}'
```

**Query the chatbot:**
```bash
curl -X POST 'http://localhost:8001/chat/message/json' \
  -H 'Content-Type: application/json' \
  -H 'soeid: test-user' \
  -d '{
    "query": "What is machine learning?",
    "session_id": "test",
    "retrieval_enabled": true
  }'
```

## Features Demonstrated

### Document Processing
- **Multiple Formats**: Text, PDF, images
- **Intelligent Chunking**: Page-based and semantic chunking
- **Vector Embeddings**: Using sentence transformers
- **Metadata Enrichment**: File info, timestamps, user tracking

### Conversational AI
- **Context-Aware**: Maintains conversation history
- **Document Retrieval**: Semantic search across ingested documents
- **Re-ranking**: Improves relevance of retrieved content
- **Multiple LLM Providers**: Groq, OpenAI, Vertex AI support

### Production Features
- **REST APIs**: Well-documented endpoints
- **Configuration Management**: YAML-based configuration
- **Error Handling**: Comprehensive error responses
- **Logging**: Structured logging for debugging
- **Health Checks**: Service monitoring endpoints

## Configuration

Each service has its own `config.yaml` file with settings for:

- **API Configuration**: Ports, CORS, rate limiting
- **Vector Stores**: PostgreSQL, FAISS, ChromaDB
- **LLM Providers**: Groq, OpenAI, Vertex AI
- **Memory Systems**: Simple, LangGraph
- **Security**: API keys, authentication

## Development Workflow

### 1. Core Library Development
Modify core RAG components in `../../src/rag/`:
- `ingestion/` - Document processing logic
- `chatbot/` - Conversation and retrieval logic
- `shared/` - Common utilities and models

### 2. API Development
Customize API layers in each service:
- `api/main.py` - FastAPI application setup
- `api/router.py` - Endpoint definitions
- `api/service.py` - Business logic

### 3. Configuration
Adjust settings in each service's `config.yaml`:
- Change vector store types
- Switch LLM providers
- Modify chunking strategies
- Tune retrieval parameters

## Deployment

### Local Development
- Use the provided run scripts
- SQLite/FAISS for simple setup
- In-memory configurations for testing

### Production
- PostgreSQL with pgvector for scalability
- Load balancers for high availability
- Environment-specific configurations
- Monitoring and logging integration

## Troubleshooting

### Common Issues

1. **Services won't start**
   - Check port availability: `lsof -i :8000` and `lsof -i :8001`
   - Verify environment variables are set
   - Check configuration file syntax

2. **No documents retrieved**
   - Ensure ingestion completed successfully
   - Check vector store connection
   - Verify embedding dimensions match

3. **LLM errors**
   - Validate API keys
   - Check rate limits
   - Try different providers in config

### Debug Mode

Enable detailed logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

### Health Checks

Monitor service health:
```bash
curl http://localhost:8000/health  # Ingestion service
curl http://localhost:8001/health  # Chatbot service
```

## Examples Directory Structure

```
examples/rag/
├── README.md                 # This file
├── ingestion/
│   ├── README.md            # Ingestion service guide
│   ├── config.yaml          # Ingestion configuration
│   ├── run_ingestion.py     # Service runner
│   └── api/                 # API implementation
├── chatbot/
│   ├── README.md            # Chatbot service guide
│   ├── config.yaml          # Chatbot configuration
│   ├── run_chatbot.py       # Service runner
│   └── api/                 # API implementation
└── __init__.py
```

## Next Steps

1. **Explore Individual Services**: Check out the detailed READMEs in each service directory
2. **Customize Configuration**: Modify `config.yaml` files to suit your needs
3. **Extend Functionality**: Add new endpoints, parsers, or LLM providers
4. **Deploy to Production**: Use these examples as templates for your deployment

## Support

For issues and questions:
1. Check the individual service READMEs
2. Review configuration options
3. Enable debug logging for detailed error information
4. Refer to the core library documentation
