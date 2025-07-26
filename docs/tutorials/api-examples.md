# API Examples

Comprehensive examples for using the RAG system APIs, including ingestion, chatbot, and management endpoints.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Ingestion API Examples](#ingestion-api-examples)
- [Chatbot API Examples](#chatbot-api-examples)
- [Management API Examples](#management-api-examples)
- [Batch Operations](#batch-operations)
- [Error Handling](#error-handling)
- [SDK Examples](#sdk-examples)
- [Performance Tips](#performance-tips)

## Overview

This guide provides practical examples for all RAG system APIs:

### Available APIs
- **Ingestion API** (Port 8000): Document upload and processing
- **Chatbot API** (Port 8001): Query processing and conversation management
- **Management API** (Port 8002): System administration and monitoring

### Example Formats
- cURL commands for quick testing
- Python requests for integration
- JavaScript/Node.js for web applications

## Authentication

### Environment Setup

```bash
# Set authentication headers
export SOEID="user123"
export API_KEY="your-api-key"  # If API key authentication is enabled
```

### Authentication Headers

```bash
# Required headers for all requests
-H "soeid: user123"
-H "Content-Type: application/json"  # For JSON requests
```

## Ingestion API Examples

### Single Document Upload

**cURL:**
```bash
# Upload PDF document
curl -X POST "http://localhost:8000/upload" \
  -H "soeid: user123" \
  -F "file=@documents/report.pdf" \
  -F "metadata={\"source\": \"reports\", \"department\": \"finance\", \"year\": 2024}"

# Upload text file
curl -X POST "http://localhost:8000/upload" \
  -H "soeid: user123" \
  -F "file=@documents/policy.txt" \
  -F "metadata={\"type\": \"policy\", \"version\": \"1.2\"}"
```

**Python:**
```python
import requests
import json

def upload_document(file_path, metadata=None):
    """Upload a single document to the RAG system."""
    url = "http://localhost:8000/upload"
    headers = {"soeid": "user123"}
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'metadata': json.dumps(metadata or {})}
        
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()

# Example usage
result = upload_document(
    "documents/financial_report.pdf",
    {
        "source": "finance_team",
        "quarter": "Q3_2024",
        "confidential": True
    }
)

print(f"Document ID: {result['document_id']}")
print(f"Pages processed: {result['pages_processed']}")
print(f"Processing time: {result['processing_time']}s")
```

**JavaScript/Node.js:**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function uploadDocument(filePath, metadata = {}) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('metadata', JSON.stringify(metadata));
    
    try {
        const response = await axios.post('http://localhost:8000/upload', form, {
            headers: {
                'soeid': 'user123',
                ...form.getHeaders()
            }
        });
        
        return response.data;
    } catch (error) {
        console.error('Upload failed:', error.response?.data || error.message);
        throw error;
    }
}

// Example usage
uploadDocument('documents/manual.pdf', {
    type: 'manual',
    version: '2.1',
    department: 'engineering'
}).then(result => {
    console.log('Upload successful:', result);
}).catch(error => {
    console.error('Upload failed:', error);
});
```

### Batch Document Upload

**cURL:**
```bash
# Upload multiple documents
curl -X POST "http://localhost:8000/batch-upload" \
  -H "soeid: user123" \
  -F "files=@documents/doc1.pdf" \
  -F "files=@documents/doc2.pdf" \
  -F "files=@documents/doc3.txt" \
  -F "metadata={\"batch_id\": \"finance_q3\", \"source\": \"finance_team\"}"
```

**Python:**
```python
def batch_upload_documents(file_paths, metadata=None):
    """Upload multiple documents in a single batch."""
    url = "http://localhost:8000/batch-upload"
    headers = {"soeid": "user123"}
    
    files = []
    for file_path in file_paths:
        files.append(('files', open(file_path, 'rb')))
    
    data = {'metadata': json.dumps(metadata or {})}
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()
    finally:
        # Close all file handles
        for _, file_handle in files:
            file_handle.close()

# Example usage
file_list = [
    "documents/report1.pdf",
    "documents/report2.pdf",
    "documents/summary.txt"
]

result = batch_upload_documents(file_list, {
    "batch_id": "quarterly_reports",
    "quarter": "Q3_2024",
    "department": "finance"
})

print(f"Batch ID: {result['batch_id']}")
print(f"Documents processed: {result['documents_processed']}")
```

### URL-Based Ingestion

**cURL:**
```bash
# Ingest document from URL
curl -X POST "http://localhost:8000/ingest-url" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "metadata": {
      "source": "external",
      "url_source": "company_website"
    }
  }'
```

**Python:**
```python
def ingest_from_url(url, metadata=None):
    """Ingest document from URL."""
    api_url = "http://localhost:8000/ingest-url"
    headers = {
        "soeid": "user123",
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url,
        "metadata": metadata or {}
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# Example usage
result = ingest_from_url(
    "https://company.com/policies/privacy-policy.pdf",
    {
        "type": "policy",
        "category": "privacy",
        "last_updated": "2024-01-15"
    }
)
```

### Check Processing Status

**cURL:**
```bash
# Check document processing status
curl "http://localhost:8000/status/doc_12345" \
  -H "soeid: user123"

# Check batch processing status
curl "http://localhost:8000/batch-status/batch_67890" \
  -H "soeid: user123"
```

**Python:**
```python
def check_document_status(document_id):
    """Check the processing status of a document."""
    url = f"http://localhost:8000/status/{document_id}"
    headers = {"soeid": "user123"}
    
    response = requests.get(url, headers=headers)
    return response.json()

def wait_for_processing(document_id, timeout=300):
    """Wait for document processing to complete."""
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = check_document_status(document_id)
        
        if status['status'] == 'completed':
            return status
        elif status['status'] == 'failed':
            raise Exception(f"Processing failed: {status.get('error')}")
        
        time.sleep(5)  # Check every 5 seconds
    
    raise TimeoutError("Processing timeout exceeded")

# Example usage
document_id = "doc_12345"
final_status = wait_for_processing(document_id)
print(f"Processing completed: {final_status}")
```

## Chatbot API Examples

### Basic Query

**cURL:**
```bash
# Simple query
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=What are the key points in the financial reports?"

# Query with session
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=Can you elaborate on the revenue trends?" \
  -F "session_id=finance_session_1"
```

**Python:**
```python
def send_chat_message(query, session_id=None, use_retrieval=True, use_history=True):
    """Send a message to the chatbot."""
    url = "http://localhost:8001/chat/message/json"
    headers = {
        "soeid": "user123",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "session_id": session_id,
        "use_retrieval": use_retrieval,
        "use_history": use_history
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Example usage
response = send_chat_message(
    "What are the main risks mentioned in the compliance documents?",
    session_id="compliance_review",
    use_retrieval=True,
    use_history=True
)

print(f"Response: {response['response']}")
print(f"Sources: {len(response['sources'])} documents")
print(f"Processing time: {response['processing_time']}s")
```

### Advanced Query Options

**cURL:**
```bash
# Query with advanced options
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze the financial performance trends",
    "session_id": "analysis_session",
    "use_retrieval": true,
    "use_history": true,
    "use_chat_history": true,
    "chat_history_days": 14,
    "metadata": {
      "top_k": 8,
      "score_threshold": 0.7,
      "filter": {
        "department": "finance",
        "year": 2024
      }
    }
  }'
```

**Python:**
```python
def advanced_chat_query(query, session_id, **options):
    """Send advanced chat query with custom options."""
    url = "http://localhost:8001/chat/message/json"
    headers = {
        "soeid": "user123",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "session_id": session_id,
        "use_retrieval": options.get('use_retrieval', True),
        "use_history": options.get('use_history', True),
        "use_chat_history": options.get('use_chat_history', False),
        "chat_history_days": options.get('chat_history_days', 7),
        "metadata": {
            "top_k": options.get('top_k', 5),
            "score_threshold": options.get('score_threshold', 0.5),
            "filter": options.get('filter', {})
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Example usage
response = advanced_chat_query(
    "What are the compliance requirements for data handling?",
    session_id="compliance_session",
    use_chat_history=True,
    chat_history_days=30,
    top_k=10,
    score_threshold=0.8,
    filter={"type": "policy", "category": "data_protection"}
)
```

### Streaming Responses

**cURL:**
```bash
# Stream response
curl -X POST "http://localhost:8001/chat/stream" \
  -H "soeid: user123" \
  -F "query=Explain the quarterly financial results in detail" \
  -F "session_id=finance_stream"
```

**Python:**
```python
def stream_chat_response(query, session_id=None):
    """Stream chat response for real-time updates."""
    url = "http://localhost:8001/chat/stream"
    headers = {"soeid": "user123"}
    
    data = {
        "query": query,
        "session_id": session_id or f"stream_{int(time.time())}"
    }
    
    response = requests.post(url, headers=headers, data=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            # Parse Server-Sent Events format
            if line.startswith(b'data: '):
                data = line[6:].decode('utf-8')
                if data != '[DONE]':
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

# Example usage
print("Streaming response:")
for chunk in stream_chat_response("Summarize the key business metrics"):
    if 'content' in chunk:
        print(chunk['content'], end='', flush=True)
    elif 'sources' in chunk:
        print(f"\n\nSources: {len(chunk['sources'])} documents")
```

### Session Management

**cURL:**
```bash
# List user sessions
curl "http://localhost:8001/sessions" \
  -H "soeid: user123"

# Get session history
curl "http://localhost:8001/sessions/finance_session/history" \
  -H "soeid: user123"

# Clear session
curl -X DELETE "http://localhost:8001/sessions/finance_session" \
  -H "soeid: user123"
```

**Python:**
```python
def list_sessions():
    """List all sessions for the user."""
    url = "http://localhost:8001/sessions"
    headers = {"soeid": "user123"}
    
    response = requests.get(url, headers=headers)
    return response.json()

def get_session_history(session_id):
    """Get conversation history for a session."""
    url = f"http://localhost:8001/sessions/{session_id}/history"
    headers = {"soeid": "user123"}
    
    response = requests.get(url, headers=headers)
    return response.json()

def clear_session(session_id):
    """Clear a specific session."""
    url = f"http://localhost:8001/sessions/{session_id}"
    headers = {"soeid": "user123"}
    
    response = requests.delete(url, headers=headers)
    return response.json()

# Example usage
sessions = list_sessions()
print(f"Active sessions: {len(sessions['sessions'])}")

for session in sessions['sessions']:
    history = get_session_history(session['session_id'])
    print(f"Session {session['session_id']}: {len(history['messages'])} messages")
```

## Management API Examples

### Document Management

**cURL:**
```bash
# List all documents
curl "http://localhost:8000/documents" \
  -H "soeid: user123"

# Get document details
curl "http://localhost:8000/documents/doc_12345" \
  -H "soeid: user123"

# Search documents
curl "http://localhost:8000/documents/search?query=financial&limit=10" \
  -H "soeid: user123"

# Delete document
curl -X DELETE "http://localhost:8000/documents/doc_12345" \
  -H "soeid: user123"
```

**Python:**
```python
def list_documents(limit=50, offset=0, filter_params=None):
    """List documents with pagination and filtering."""
    url = "http://localhost:8000/documents"
    headers = {"soeid": "user123"}
    
    params = {
        "limit": limit,
        "offset": offset
    }
    
    if filter_params:
        params.update(filter_params)
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def search_documents(query, limit=10):
    """Search documents by content."""
    url = "http://localhost:8000/documents/search"
    headers = {"soeid": "user123"}
    
    params = {
        "query": query,
        "limit": limit
    }
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def delete_document(document_id):
    """Delete a specific document."""
    url = f"http://localhost:8000/documents/{document_id}"
    headers = {"soeid": "user123"}
    
    response = requests.delete(url, headers=headers)
    return response.json()

# Example usage
documents = list_documents(limit=20, filter_params={"type": "policy"})
print(f"Found {len(documents['documents'])} policy documents")

search_results = search_documents("data privacy", limit=5)
print(f"Search found {len(search_results['documents'])} relevant documents")
```

### System Health and Monitoring

**cURL:**
```bash
# Basic health check
curl "http://localhost:8001/health"

# Detailed health check
curl "http://localhost:8001/health/detailed"

# System metrics
curl "http://localhost:8002/metrics"

# System information
curl "http://localhost:8000/system/info" \
  -H "soeid: user123"
```

**Python:**
```python
def check_system_health():
    """Check the health of all system components."""
    services = [
        ("Ingestion", "http://localhost:8000/health"),
        ("Chatbot", "http://localhost:8001/health"),
        ("Metrics", "http://localhost:8002/metrics")
    ]
    
    health_status = {}
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            health_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except requests.RequestException as e:
            health_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status

def get_system_info():
    """Get detailed system information."""
    url = "http://localhost:8000/system/info"
    headers = {"soeid": "user123"}
    
    response = requests.get(url, headers=headers)
    return response.json()

# Example usage
health = check_system_health()
for service, status in health.items():
    print(f"{service}: {status['status']}")

system_info = get_system_info()
print(f"Documents indexed: {system_info['document_count']}")
print(f"System version: {system_info['version']}")
```

## Batch Operations

### Bulk Document Processing

**Python:**
```python
import asyncio
import aiohttp
import json
from pathlib import Path

async def bulk_upload_directory(directory_path, metadata_template=None):
    """Upload all documents in a directory."""
    directory = Path(directory_path)
    files = list(directory.glob("*.pdf")) + list(directory.glob("*.txt"))
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for file_path in files:
            metadata = {
                **(metadata_template or {}),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size
            }
            
            task = upload_document_async(session, file_path, metadata)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def upload_document_async(session, file_path, metadata):
    """Upload a single document asynchronously."""
    url = "http://localhost:8000/upload"
    headers = {"soeid": "user123"}
    
    data = aiohttp.FormData()
    data.add_field('file', open(file_path, 'rb'), filename=file_path.name)
    data.add_field('metadata', json.dumps(metadata))
    
    try:
        async with session.post(url, headers=headers, data=data) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e), "file": str(file_path)}

# Example usage
async def main():
    results = await bulk_upload_directory(
        "documents/policies/",
        {"department": "legal", "type": "policy"}
    )
    
    successful = [r for r in results if not isinstance(r, Exception) and "error" not in r]
    failed = [r for r in results if isinstance(r, Exception) or "error" in r]
    
    print(f"Successfully uploaded: {len(successful)} documents")
    print(f"Failed uploads: {len(failed)} documents")

# Run with: asyncio.run(main())
```

## Error Handling

### Robust Error Handling

**Python:**
```python
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RAGAPIClient:
    """Robust API client with error handling and retries."""
    
    def __init__(self, base_url, soeid, timeout=30):
        self.base_url = base_url
        self.soeid = soeid
        self.timeout = timeout
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({"soeid": self.soeid})
    
    def upload_document(self, file_path, metadata=None, max_retries=3):
        """Upload document with error handling and retries."""
        url = f"{self.base_url}/upload"
        
        for attempt in range(max_retries):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    data = {'metadata': json.dumps(metadata or {})}
                    
                    response = self.session.post(
                        url, 
                        files=files, 
                        data=data, 
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    return response.json()
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Upload attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff
                time.sleep(2 ** attempt)

# Example usage
client = RAGAPIClient("http://localhost:8000", "user123")

try:
    result = client.upload_document("document.pdf", {"type": "report"})
    print(f"Upload successful: {result['document_id']}")
except Exception as e:
    print(f"Upload failed: {e}")
```

## SDK Examples

### Simple Python SDK

```python
class RAGSDK:
    """Simple SDK for RAG system."""
    
    def __init__(self, ingestion_url, chatbot_url, soeid):
        self.ingestion_url = ingestion_url
        self.chatbot_url = chatbot_url
        self.soeid = soeid
    
    def upload(self, file_path, **metadata):
        """Upload and wait for processing."""
        result = upload_document(file_path, metadata)
        return wait_for_processing(result['document_id'])
    
    def chat(self, query, session_id=None, **options):
        """Send a chat message."""
        return send_chat_message(query, session_id, **options)

# Example usage
sdk = RAGSDK("http://localhost:8000", "http://localhost:8001", "user123")

# Upload document
doc_status = sdk.upload("report.pdf", type="financial", quarter="Q3")
print(f"Document processed: {doc_status['chunks_created']} chunks")

# Chat with the system
response = sdk.chat("What are the key findings in the financial reports?")
print(f"Response: {response['response']}")
```

## Performance Tips

### Optimization Strategies

1. **Batch Operations**: Use batch upload for multiple documents
2. **Async Processing**: Use async/await for concurrent operations
3. **Caching**: Cache frequently accessed documents and responses
4. **Connection Pooling**: Reuse HTTP connections
5. **Rate Limiting**: Implement client-side rate limiting
6. **Compression**: Use gzip compression for large payloads
7. **Pagination**: Use pagination for large result sets

### Example Optimized Client

```python
class OptimizedRAGClient:
    """Optimized client with connection pooling and caching."""
    
    def __init__(self, base_url, soeid):
        self.session = requests.Session()
        self.session.headers.update({
            "soeid": soeid,
            "Accept-Encoding": "gzip, deflate"
        })
        
        # Connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Simple cache
        self.cache = {}
    
    def cached_request(self, method, url, cache_key=None, **kwargs):
        """Make request with optional caching."""
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        response = self.session.request(method, url, **kwargs)
        result = response.json()
        
        if cache_key:
            self.cache[cache_key] = result
        
        return result

# Usage
client = OptimizedRAGClient("http://localhost:8000", "user123")
```

This comprehensive API examples guide provides practical, production-ready code for integrating with the RAG system across different programming languages and use cases.
