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

Set up your API keys in your environment:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export API_KEY="your_api_key_here"  # Optional: for API authentication
export ADMIN_API_KEY="your_admin_key_here"  # Optional: for admin access
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

The service uses `config.yaml` in this directory. Key settings:

- **Port**: 8000 (configurable in `ingestion.port`)
- **Vector Store**: PostgreSQL with pgvector (configurable in `ingestion.vector_store`)
- **Embedding Model**: all-MiniLM-L6-v2 (configurable in `ingestion.embedding`)
- **Parsers**: Simple text parser enabled by default

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
