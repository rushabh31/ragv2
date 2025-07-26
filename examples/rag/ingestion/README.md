# Ingestion Service Example

This example demonstrates how to use the controlsgenai RAG library to build a document ingestion service.

## Overview

The ingestion service provides REST API endpoints for:
- Uploading and processing documents
- Parsing various document formats (PDF, TXT, etc.)
- Chunking documents into smaller pieces
- Generating embeddings for document chunks
- Storing embeddings in vector databases

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

# API Authentication (Optional)
export API_KEY="your_api_key_here"  # For API authentication
export ADMIN_API_KEY="your_admin_key_here"  # For admin access
```

### 3. Start the Service

From this directory (`examples/rag/ingestion/`):

```bash
python run_ingestion.py
```

The service will start on `http://localhost:8000`

### 4. Test the Service

#### Upload a Document

```bash
# Create a test document
echo "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data." > test_document.txt

# Upload the document
curl -X POST 'http://localhost:8000/ingest/upload' \
  -H 'soeid: test-user' \
  -F 'file=@test_document.txt' \
  -F 'options={"file_name": "ml_guide.txt", "session_id": "test_session"}'
```

Expected response:
```json
{
  "job_id": "uuid-here",
  "status": "pending",
  "document_id": "uuid-here",
  "message": "Document upload accepted for processing"
}
```

#### Check Processing Status

```bash
curl -X GET 'http://localhost:8000/ingest/documents/user/test-user' \
  -H 'soeid: test-user'
```

## Configuration

The service uses `config.yaml` in this directory. All configuration parameters are verified from the source code:

### Document Processing
- **Chunk Size**: 1000 characters (configurable in `document_processing.chunk_size`)
- **Chunk Overlap**: 200 characters (configurable in `document_processing.chunk_overlap`)
- **Max Chunks**: 100 per document (configurable in `document_processing.max_chunks_per_document`)

### Parser Configuration
- **Default Parser**: `vision_parser` (Vertex AI Gemini Vision)
- **Available Parsers**: `vision_parser`, `groq_vision_parser`, `openai_vision`, `simple_text`
- **Vision Model**: `gemini-1.5-pro-002`
- **Parallel Processing**: 5 concurrent pages (configurable in `parser.config.max_concurrent_pages`)

### Embedding Configuration
- **Default Provider**: `vertex` (Vertex AI)
- **Available Providers**: `vertex`, `vertex_ai`, `openai`, `openai_universal`, `azure_openai`
- **Model**: `text-embedding-004` (768 dimensions)
- **Batch Size**: 100 (configurable in `embedding.config.batch_size`)

### Vector Store Configuration
- **Default Provider**: `faiss` (FAISS with HNSW index)
- **Available Providers**: `faiss`, `pgvector`, `chromadb`
- **Index Path**: `./data/faiss.index`
- **Metadata Path**: `./data/metadata.pickle`

### API Configuration
- **Host**: `0.0.0.0`
- **Port**: 8000
- **CORS**: Enabled for all origins
- **API Key**: Required by default

## API Endpoints

### Document Upload
- **POST** `/ingest/upload`
- **Headers**: `soeid: <user_id>`
- **Body**: Multipart form with `file` and `options`

### List User Documents
- **GET** `/ingest/documents/user/{soeid}`
- **Headers**: `soeid: <user_id>`

### Health Check
- **GET** `/health`

## Supported Document Types

- **Text files**: `.txt`, `.md`
- **PDF files**: `.pdf` (requires additional parser configuration)
- **Images**: `.jpg`, `.png` (requires vision parser configuration)

## Vector Store Options

The service supports multiple vector stores:

1. **PostgreSQL + pgvector** (default)
   - Requires PostgreSQL with pgvector extension
   - Connection: `postgresql://localhost:5432/postgres`

2. **FAISS** (in-memory)
   - Change `ingestion.vector_store.type` to `faiss`

3. **ChromaDB**
   - Change `ingestion.vector_store.type` to `chromadb`

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill existing processes
   lsof -ti:8000 | xargs kill -9
   ```

2. **PostgreSQL connection error**
   - Ensure PostgreSQL is running
   - Install pgvector extension: `CREATE EXTENSION vector;`
   - Or switch to FAISS in config.yaml

3. **Import errors**
   - Make sure you're running from this directory
   - Ensure controlsgenai is installed: `pip install -e .`

### Logs

The service logs to stdout. Look for:
- Configuration loading messages
- Document processing status
- Error messages with stack traces

## Development

To modify the ingestion service:

1. **Core Logic**: Edit files in `../../../src/rag/ingestion/`
2. **API Layer**: Edit files in `api/`
3. **Configuration**: Modify `config.yaml`
4. **Dependencies**: Update `../../../requirements.txt`

## Next Steps

After successfully ingesting documents, try the [Chatbot Service](../chatbot/README.md) to query your documents with natural language.
