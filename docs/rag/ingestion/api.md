# Ingestion API

## 🎯 Overview

The Ingestion API provides REST endpoints for uploading, processing, and managing documents in the RAG system. It handles the complete ingestion pipeline from document upload to vector storage, making your documents searchable by the chatbot.

## 🚀 Quick Start

### **Start the Ingestion Service**
```bash
cd examples/rag/ingestion
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### **Test the Service**
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "parser": "ready",
    "embedder": "ready", 
    "vector_store": "ready"
  }
}
```

## 📋 API Endpoints

### **Document Upload**

#### **Single File Upload**
```http
POST /ingest/file
Content-Type: multipart/form-data
X-API-Key: test-api-key

Parameters:
- file: File (required) - Document to upload
- metadata: JSON string (optional) - Document metadata
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@document.pdf" \
  -F "metadata={\"source\": \"user_upload\", \"category\": \"manual\", \"priority\": \"high\"}"
```

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "processing",
  "message": "Document upload successful, processing started",
  "file_info": {
    "filename": "document.pdf",
    "size": 1048576,
    "content_type": "application/pdf"
  },
  "estimated_processing_time": 120
}
```

#### **Batch File Upload**
```http
POST /ingest/batch
Content-Type: multipart/form-data
X-API-Key: test-api-key

Parameters:
- files: List[File] (required) - Multiple documents
- metadata: JSON string (optional) - Shared metadata for all files
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "X-API-Key: test-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.txt" \
  -F "metadata={\"batch_id\": \"batch_001\", \"source\": \"bulk_upload\"}"
```

**Response:**
```json
{
  "batch_id": "batch_123456",
  "status": "processing",
  "message": "Batch upload successful, processing started",
  "files": [
    {
      "filename": "doc1.pdf",
      "job_id": "job_123457",
      "status": "queued"
    },
    {
      "filename": "doc2.pdf", 
      "job_id": "job_123458",
      "status": "queued"
    },
    {
      "filename": "doc3.txt",
      "job_id": "job_123459", 
      "status": "queued"
    }
  ],
  "estimated_total_time": 300
}
```

#### **URL Processing**
```http
POST /ingest/url
Content-Type: application/json
X-API-Key: test-api-key

Body:
{
  "url": "https://example.com/document.pdf",
  "metadata": {
    "source": "web",
    "category": "external"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest/url" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/whitepaper.pdf",
    "metadata": {
      "source": "website",
      "category": "research",
      "author": "external"
    }
  }'
```

### **Processing Status**

#### **Job Status**
```http
GET /ingest/status/{job_id}
X-API-Key: test-api-key
```

**Example:**
```bash
curl -X GET "http://localhost:8000/ingest/status/job_123456" \
  -H "X-API-Key: test-api-key"
```

**Response (Processing):**
```json
{
  "job_id": "job_123456",
  "status": "processing",
  "progress": {
    "current_step": "embedding",
    "steps_completed": 2,
    "total_steps": 4,
    "percentage": 50
  },
  "processing_info": {
    "pages_processed": 15,
    "total_pages": 30,
    "chunks_created": 45,
    "embeddings_generated": 45
  },
  "estimated_remaining_time": 60
}
```

**Response (Completed):**
```json
{
  "job_id": "job_123456",
  "status": "completed",
  "result": {
    "chunks_created": 89,
    "embeddings_generated": 89,
    "processing_time": 125.5,
    "pages_processed": 30,
    "storage_location": "vector_store_faiss"
  },
  "metadata": {
    "filename": "document.pdf",
    "file_size": 1048576,
    "content_type": "application/pdf",
    "upload_time": "2024-01-15T10:30:00Z",
    "completion_time": "2024-01-15T10:32:05Z"
  }
}
```

**Response (Failed):**
```json
{
  "job_id": "job_123456",
  "status": "failed",
  "error": {
    "code": "PARSING_ERROR",
    "message": "Failed to extract text from PDF",
    "details": "Document appears to be corrupted or password protected",
    "step": "parsing",
    "retry_possible": true
  },
  "partial_results": {
    "pages_processed": 5,
    "chunks_created": 12
  }
}
```

#### **Batch Status**
```http
GET /ingest/batch/{batch_id}
X-API-Key: test-api-key
```

**Example:**
```bash
curl -X GET "http://localhost:8000/ingest/batch/batch_123456" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "batch_id": "batch_123456",
  "status": "processing",
  "progress": {
    "completed_jobs": 2,
    "total_jobs": 3,
    "percentage": 67
  },
  "jobs": [
    {
      "job_id": "job_123457",
      "filename": "doc1.pdf",
      "status": "completed",
      "chunks_created": 45
    },
    {
      "job_id": "job_123458", 
      "filename": "doc2.pdf",
      "status": "completed",
      "chunks_created": 32
    },
    {
      "job_id": "job_123459",
      "filename": "doc3.txt", 
      "status": "processing",
      "progress": 75
    }
  ]
}
```

### **Document Management**

#### **List Documents**
```http
GET /ingest/documents
X-API-Key: test-api-key

Query Parameters:
- limit: int (default: 50) - Number of documents to return
- offset: int (default: 0) - Pagination offset
- filter: JSON string - Metadata filters
- sort_by: string - Sort field (upload_time, filename, size)
- sort_order: string - Sort order (asc, desc)
```

**Example:**
```bash
curl -X GET "http://localhost:8000/ingest/documents?limit=10&sort_by=upload_time&sort_order=desc" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_123456",
      "filename": "manual.pdf",
      "upload_time": "2024-01-15T10:30:00Z",
      "file_size": 1048576,
      "chunks_count": 89,
      "status": "completed",
      "metadata": {
        "source": "user_upload",
        "category": "manual"
      }
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 10,
    "offset": 0,
    "has_more": true
  }
}
```

#### **Get Document Details**
```http
GET /ingest/documents/{document_id}
X-API-Key: test-api-key
```

**Example:**
```bash
curl -X GET "http://localhost:8000/ingest/documents/doc_123456" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "document_id": "doc_123456",
  "filename": "manual.pdf",
  "upload_time": "2024-01-15T10:30:00Z",
  "processing_time": 125.5,
  "file_info": {
    "size": 1048576,
    "content_type": "application/pdf",
    "pages": 30
  },
  "processing_results": {
    "chunks_created": 89,
    "embeddings_generated": 89,
    "parsing_method": "vision_parser",
    "chunking_method": "semantic",
    "embedding_model": "text-embedding-004"
  },
  "metadata": {
    "source": "user_upload",
    "category": "manual",
    "priority": "high"
  },
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "content_preview": "This manual describes the operation of...",
      "chunk_size": 987,
      "page_numbers": [1, 2]
    }
  ]
}
```

#### **Delete Document**
```http
DELETE /ingest/documents/{document_id}
X-API-Key: test-api-key
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/ingest/documents/doc_123456" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "doc_123456",
  "chunks_removed": 89,
  "embeddings_removed": 89
}
```

### **System Information**

#### **Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "parser": "ready",
    "embedder": "ready",
    "vector_store": "ready",
    "database": "connected"
  },
  "system_info": {
    "cpu_usage": 25.5,
    "memory_usage": 1024,
    "disk_usage": 45.2,
    "active_jobs": 3
  }
}
```

#### **System Statistics**
```http
GET /ingest/stats
X-API-Key: test-api-key
```

**Response:**
```json
{
  "statistics": {
    "total_documents": 1250,
    "total_chunks": 45000,
    "total_embeddings": 45000,
    "storage_size_mb": 2048,
    "processing_stats": {
      "documents_today": 25,
      "documents_this_week": 150,
      "average_processing_time": 95.5,
      "success_rate": 0.98
    },
    "queue_stats": {
      "pending_jobs": 5,
      "active_jobs": 2,
      "failed_jobs_today": 1
    }
  }
}
```

## 🔧 Configuration

### **API Configuration**
```yaml
# examples/rag/ingestion/config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  api_key: "test-api-key"              # Change in production
  cors_origins: ["*"]                  # Restrict in production
  max_file_size: 50000000              # 50MB limit
  max_batch_size: 10                   # Maximum files per batch
  upload_timeout: 300                  # Upload timeout (seconds)
  processing_timeout: 1800             # Processing timeout (seconds)

# Rate limiting
rate_limiting:
  enabled: true
  requests_per_minute: 100
  burst_size: 20

# File validation
file_validation:
  allowed_extensions: [".pdf", ".txt", ".docx", ".md"]
  max_pages: 100                       # Maximum pages per document
  scan_for_viruses: false              # Enable virus scanning
```

### **Processing Configuration**
```yaml
# Processing pipeline settings
parser:
  provider: "vision_parser"
  config:
    model: "gemini-1.5-pro-002"
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
```

## 🔐 Authentication

### **API Key Authentication**
```bash
# Include API key in all requests
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"
```

### **Production Security**
```yaml
# Secure configuration for production
api:
  api_key: "${API_KEY}"                # Use environment variable
  cors_origins: ["https://yourdomain.com"]  # Restrict origins
  https_only: true                     # Require HTTPS
  rate_limiting:
    enabled: true
    requests_per_minute: 60            # Stricter limits
```

## 🚨 Error Handling

### **Common Error Responses**

#### **Authentication Error (401)**
```json
{
  "detail": "Invalid API key",
  "error_code": "INVALID_API_KEY"
}
```

#### **File Too Large (413)**
```json
{
  "detail": "File size exceeds maximum allowed size of 50MB",
  "error_code": "FILE_TOO_LARGE",
  "max_size": 50000000
}
```

#### **Unsupported File Type (415)**
```json
{
  "detail": "File type not supported",
  "error_code": "UNSUPPORTED_FILE_TYPE",
  "supported_types": [".pdf", ".txt", ".docx", ".md"]
}
```

#### **Processing Error (500)**
```json
{
  "detail": "Document processing failed",
  "error_code": "PROCESSING_ERROR",
  "job_id": "job_123456",
  "retry_possible": true,
  "error_details": {
    "step": "parsing",
    "message": "Failed to extract text from page 5"
  }
}
```

### **Error Recovery**
```bash
# Retry failed job
curl -X POST "http://localhost:8000/ingest/retry/job_123456" \
  -H "X-API-Key: test-api-key"

# Get error details
curl -X GET "http://localhost:8000/ingest/errors/job_123456" \
  -H "X-API-Key: test-api-key"
```

## 📊 Monitoring and Logging

### **Request Logging**
```python
# All requests are logged with:
- Timestamp
- Request method and path
- Response status code
- Processing time
- File information (if applicable)
- Error details (if applicable)

# Example log entry
2024-01-15 10:30:00 INFO POST /ingest/file status=202 time=1.2s file=document.pdf size=1MB job_id=job_123456
```

### **Performance Metrics**
```bash
# Get performance metrics
curl -X GET "http://localhost:8000/ingest/metrics" \
  -H "X-API-Key: test-api-key"
```

**Response:**
```json
{
  "metrics": {
    "requests_per_minute": 15.5,
    "average_response_time": 1250,
    "success_rate": 0.98,
    "active_connections": 5,
    "queue_length": 3,
    "processing_times": {
      "parsing": 45.2,
      "chunking": 12.8,
      "embedding": 67.5,
      "storage": 8.1
    }
  }
}
```

## 🎯 Best Practices

### **File Upload Optimization**
1. **Batch uploads**: Use batch endpoint for multiple files
2. **File preparation**: Optimize PDFs before upload
3. **Metadata**: Include relevant metadata for better organization
4. **Error handling**: Implement retry logic for failed uploads

### **API Usage Patterns**
```python
# Python client example
import requests
import time

class IngestionClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def upload_file(self, file_path, metadata=None):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps(metadata)} if metadata else {}
            
            response = requests.post(
                f"{self.base_url}/ingest/file",
                headers=self.headers,
                files=files,
                data=data
            )
            
            return response.json()
    
    def wait_for_completion(self, job_id, timeout=300):
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            time.sleep(5)  # Poll every 5 seconds
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def get_job_status(self, job_id):
        response = requests.get(
            f"{self.base_url}/ingest/status/{job_id}",
            headers=self.headers
        )
        return response.json()

# Usage
client = IngestionClient("http://localhost:8000", "test-api-key")

# Upload and wait for completion
result = client.upload_file("document.pdf", {"source": "api_client"})
job_id = result['job_id']

final_status = client.wait_for_completion(job_id)
print(f"Processing completed: {final_status['result']['chunks_created']} chunks created")
```

### **Production Deployment**
1. **Security**: Use strong API keys and HTTPS
2. **Rate limiting**: Implement appropriate rate limits
3. **Monitoring**: Set up health checks and alerting
4. **Scaling**: Use load balancers for multiple instances

## 📚 Related Documentation

- **[Document Parsers](./parsers.md)** - Configure document parsing
- **[Text Chunkers](./chunkers.md)** - Configure text chunking
- **[Embedding Models](./embedders.md)** - Configure embedding generation
- **[Vector Stores](./vector-stores.md)** - Configure vector storage
- **[Configuration Guide](../../configuration.md)** - Complete configuration reference

## 🚀 Quick Examples

### **Upload and Process Document**
```bash
# Upload a document
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@manual.pdf" \
  -F "metadata={\"category\": \"user_manual\", \"version\": \"2.1\"}"

# Check processing status
curl -X GET "http://localhost:8000/ingest/status/job_123456" \
  -H "X-API-Key: test-api-key"

# List all documents
curl -X GET "http://localhost:8000/ingest/documents" \
  -H "X-API-Key: test-api-key"
```

### **Batch Processing**
```bash
# Upload multiple documents
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "X-API-Key: test-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.txt" \
  -F "files=@doc3.docx" \
  -F "metadata={\"batch\": \"training_materials\"}"

# Monitor batch progress
curl -X GET "http://localhost:8000/ingest/batch/batch_123456" \
  -H "X-API-Key: test-api-key"
```

---

**Next Steps**: 
- [Set up the Chatbot Service](../chatbot/README.md)
- [Configure Document Retrieval](../chatbot/retrievers.md)
- [Use the Complete System](../../tutorials/complete-walkthrough.md)
