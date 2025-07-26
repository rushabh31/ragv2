# Complete Walkthrough

A comprehensive end-to-end tutorial for setting up and using the RAG system, from installation to advanced features.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Configuration](#step-2-configuration)
- [Step 3: Database Setup](#step-3-database-setup)
- [Step 4: Document Ingestion](#step-4-document-ingestion)
- [Step 5: Chatbot Interaction](#step-5-chatbot-interaction)
- [Step 6: Advanced Features](#step-6-advanced-features)
- [Step 7: Monitoring & Debugging](#step-7-monitoring--debugging)
- [Step 8: Production Deployment](#step-8-production-deployment)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Overview

This tutorial will guide you through:
- Setting up the complete RAG system
- Configuring multi-provider AI models
- Ingesting your first documents
- Interacting with the chatbot
- Using advanced features like parallel processing
- Monitoring and debugging
- Deploying to production

**Time Required**: 30-60 minutes  
**Skill Level**: Beginner to Intermediate

## Prerequisites

### System Requirements
- Python 3.9 or higher
- PostgreSQL 12 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Required Accounts
- Google Cloud Platform account (for Vertex AI)
- Optional: OpenAI, Groq, or Azure OpenAI accounts

### Knowledge Requirements
- Basic Python programming
- Basic command line usage
- Basic understanding of APIs

## Step 1: Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/your-org/rag-system.git
cd rag-system
```

### 1.2 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 1.4 Verify Installation

```bash
# Run installation test
python test_installation.py
```

Expected output:
```
✅ All dependencies installed successfully
✅ Core modules importable
✅ Configuration files valid
✅ System ready for setup
```

## Step 2: Configuration

### 2.1 Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.template .env

# Edit with your credentials
nano .env
```

### 2.2 Configure Authentication

Add your credentials to `.env`:

```bash
# Universal Authentication (Required)
COIN_CONSUMER_ENDPOINT_URL=https://your-oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# Google Cloud (Required for Vertex AI)
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Optional API Keys
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 2.3 Configure Models

Edit `config.yaml` to select your preferred models:

```yaml
# config.yaml
vision:
  provider: vertex_ai  # Options: vertex_ai, groq
  config:
    model: gemini-1.5-pro-002
    region: us-central1

generation:
  provider: vertex  # Options: vertex, anthropic_vertex, openai, azure_openai, groq
  config:
    model_name: gemini-1.5-pro-002
    max_tokens: 8192
    temperature: 0.7

embedding:
  provider: vertex_ai  # Options: vertex_ai, openai_universal, azure_openai, sentence_transformer
  config:
    model: text-embedding-004
    batch_size: 100
```

### 2.4 Test Authentication

```bash
# Test model authentication
python examples/test_authentication.py
```

Expected output:
```
✅ Vertex AI authentication successful
✅ Token retrieved and valid
✅ Model access confirmed
```

## Step 3: Database Setup

### 3.1 Install PostgreSQL

**On macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**On Ubuntu:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 3.2 Create Database

```bash
# Connect to PostgreSQL
psql postgres

# Create database and user
CREATE DATABASE rag_system;
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_system TO rag_user;

# Install pgvector extension
\c rag_system
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### 3.3 Configure Database Connection

Update your `.env` file:

```bash
# PostgreSQL Configuration
POSTGRES_CONNECTION_STRING=postgresql://rag_user:your_password@localhost:5432/rag_system
```

### 3.4 Test Database Connection

```bash
# Test database connectivity
python examples/test_database.py
```

Expected output:
```
✅ Database connection successful
✅ Tables created successfully
✅ Memory system initialized
```

## Step 4: Document Ingestion

### 4.1 Prepare Sample Documents

Create a `documents` folder and add some sample files:

```bash
mkdir documents
cd documents

# Download sample PDFs (or add your own)
curl -o sample1.pdf "https://example.com/sample-document.pdf"
curl -o sample2.pdf "https://example.com/another-document.pdf"

# Or create a simple text file
echo "This is a sample document for testing the RAG system." > sample.txt
cd ..
```

### 4.2 Start Ingestion Service

```bash
# Start the ingestion API
python -m src.api.ingestion_api
```

The service will start on `http://localhost:8000`

### 4.3 Upload Your First Document

**Using curl:**
```bash
# Upload a single document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@documents/sample1.pdf" \
  -F "metadata={\"source\": \"tutorial\", \"type\": \"sample\"}"
```

**Using Python:**
```python
import requests

# Upload document
with open('documents/sample1.pdf', 'rb') as f:
    files = {'file': f}
    metadata = {'metadata': '{"source": "tutorial", "type": "sample"}'}
    response = requests.post('http://localhost:8000/upload', files=files, data=metadata)
    
print(f"Upload status: {response.status_code}")
print(f"Response: {response.json()}")
```

Expected output:
```json
{
  "status": "success",
  "document_id": "doc_12345",
  "pages_processed": 5,
  "chunks_created": 23,
  "processing_time": 12.5
}
```

### 4.4 Check Processing Status

```bash
# Check document status
curl "http://localhost:8000/status/doc_12345"
```

### 4.5 Batch Upload (Optional)

```bash
# Upload multiple documents
curl -X POST "http://localhost:8000/batch-upload" \
  -F "files=@documents/sample1.pdf" \
  -F "files=@documents/sample2.pdf" \
  -F "metadata={\"batch_id\": \"tutorial_batch\"}"
```

## Step 5: Chatbot Interaction

### 5.1 Start Chatbot Service

Open a new terminal and start the chatbot API:

```bash
# Activate virtual environment
source venv/bin/activate

# Start chatbot API
python -m src.api.chatbot_api
```

The service will start on `http://localhost:8001`

### 5.2 Your First Query

**Using curl:**
```bash
# Send a query
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=What information do you have about the uploaded documents?"
```

**Using Python:**
```python
import requests

# Send chat message
response = requests.post(
    'http://localhost:8001/chat/message/json',
    headers={'soeid': 'user123'},
    json={
        'query': 'What information do you have about the uploaded documents?',
        'session_id': 'tutorial_session',
        'use_retrieval': True,
        'use_history': True
    }
)

print(f"Response: {response.json()}")
```

Expected output:
```json
{
  "response": "Based on the uploaded documents, I can see information about...",
  "session_id": "tutorial_session",
  "sources": [
    {
      "document_id": "doc_12345",
      "page": 1,
      "relevance_score": 0.85
    }
  ],
  "processing_time": 2.3
}
```

### 5.3 Follow-up Questions

```bash
# Ask follow-up questions
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=Can you provide more details about the first document?" \
  -F "session_id=tutorial_session"
```

### 5.4 Test Different Features

```bash
# Query without retrieval (general knowledge)
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=What is artificial intelligence?" \
  -F "use_retrieval=false"

# Query with chat history from other sessions
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=What did we discuss yesterday?" \
  -F "use_chat_history=true" \
  -F "chat_history_days=7"
```

## Step 6: Advanced Features

### 6.1 Parallel Document Processing

Configure parallel processing for faster ingestion:

```yaml
# config.yaml
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 8  # Process 8 pages simultaneously
      max_pages: 100
```

Test with a large document:

```bash
# Upload large document with parallel processing
curl -X POST "http://localhost:8000/upload" \
  -F "file=@documents/large_document.pdf" \
  -F "metadata={\"parallel_processing\": true}"
```

### 6.2 Multi-Provider Model Usage

Switch between different AI providers:

```bash
# Test with different generation models
# Edit config.yaml to change provider:

# Option 1: Vertex AI Gemini
generation:
  provider: vertex
  config:
    model_name: gemini-1.5-pro-002

# Option 2: Groq Llama
generation:
  provider: groq
  config:
    model_name: llama-3.1-70b-versatile

# Option 3: Anthropic Claude
generation:
  provider: anthropic_vertex
  config:
    model_name: claude-3-5-sonnet@20240229
```

Restart the chatbot service and test:

```bash
curl -X POST "http://localhost:8001/chat/message" \
  -H "soeid: user123" \
  -F "query=Tell me about the documents using the new model"
```

### 6.3 Custom Embeddings

Test different embedding models:

```yaml
# config.yaml - Try local embeddings
embedding:
  provider: sentence_transformer
  config:
    model_name: all-mpnet-base-v2
    device: cpu
```

Re-ingest documents with new embeddings:

```bash
# Clear existing documents (optional)
curl -X DELETE "http://localhost:8000/documents/all"

# Re-upload with new embeddings
curl -X POST "http://localhost:8000/upload" \
  -F "file=@documents/sample1.pdf"
```

### 6.4 Advanced Query Features

```bash
# Semantic search with filtering
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find information about machine learning",
    "metadata": {
      "filter": {"source": "tutorial"},
      "top_k": 10,
      "score_threshold": 0.7
    }
  }'

# Streaming response
curl -X POST "http://localhost:8001/chat/stream" \
  -H "soeid: user123" \
  -F "query=Explain the main concepts in the documents"
```

## Step 7: Monitoring & Debugging

### 7.1 Check System Health

```bash
# Basic health check
curl "http://localhost:8001/health"

# Detailed health check
curl "http://localhost:8001/health/detailed"
```

### 7.2 View Metrics

```bash
# Start metrics server (if not already running)
python -c "
from src.utils.metrics import start_metrics_server
start_metrics_server(8002)
"

# View Prometheus metrics
curl "http://localhost:8002/metrics"
```

### 7.3 Debug Document Processing

```bash
# Check document status
curl "http://localhost:8000/status/doc_12345"

# List all documents
curl "http://localhost:8000/documents"

# Get document details
curl "http://localhost:8000/documents/doc_12345"
```

### 7.4 Debug Chat Sessions

```bash
# List active sessions
curl -H "soeid: user123" "http://localhost:8001/sessions"

# Get session history
curl -H "soeid: user123" "http://localhost:8001/sessions/tutorial_session/history"

# Clear session (if needed)
curl -X DELETE -H "soeid: user123" "http://localhost:8001/sessions/tutorial_session"
```

### 7.5 Performance Testing

```bash
# Run performance tests
python examples/test_parallel_performance.py

# Test system load
python examples/test_system_load.py
```

## Step 8: Production Deployment

### 8.1 Docker Deployment

```bash
# Build Docker image
docker build -t rag-system:latest .

# Run with Docker Compose
docker-compose up -d
```

### 8.2 Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get pods -n rag-system
```

### 8.3 Environment-Specific Configuration

```yaml
# config.production.yaml
logging:
  level: INFO
  format: json

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

ingestion:
  parsing:
    vision:
      max_concurrent_pages: 12  # Higher for production

memory:
  type: langgraph_checkpoint
  store_type: postgres
  postgres:
    connection_string: "${POSTGRES_CONNECTION_STRING}"
    pool_size: 20
```

### 8.4 Load Balancer Setup

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-system-ingress
  namespace: rag-system
spec:
  rules:
  - host: rag-api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-system-service
            port:
              number: 80
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures

**Problem**: `Authentication failed` errors

**Solution**:
```bash
# Check environment variables
echo $COIN_CONSUMER_CLIENT_ID
echo $PROJECT_ID

# Test authentication manually
python -c "
from src.utils.auth_manager import UniversalAuthManager
auth = UniversalAuthManager('vertex_ai')
token = auth.get_token()
print(f'Token: {token[:50]}...')
"
```

#### 2. Database Connection Issues

**Problem**: `Database connection failed`

**Solution**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection manually
psql $POSTGRES_CONNECTION_STRING -c "SELECT version();"

# Check pgvector extension
psql $POSTGRES_CONNECTION_STRING -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

#### 3. Document Processing Failures

**Problem**: Documents fail to process

**Solution**:
```bash
# Check ingestion logs
curl "http://localhost:8000/logs"

# Test with smaller document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@small_test.txt"

# Check vision model health
python -c "
from src.models.vision import VisionModelFactory
model = VisionModelFactory.create_model('vertex_ai')
print(model.get_auth_health_status())
"
```

#### 4. Slow Response Times

**Problem**: Chatbot responses are slow

**Solution**:
```bash
# Check performance metrics
curl "http://localhost:8002/metrics" | grep rag_request_duration

# Test different models
# Edit config.yaml to use faster model:
generation:
  provider: groq  # Often faster than Vertex AI
  config:
    model_name: llama-3.1-70b-versatile

# Optimize retrieval
curl -X POST "http://localhost:8001/chat/message/json" \
  -d '{"query": "test", "metadata": {"top_k": 3}}'  # Reduce retrieved documents
```

#### 5. Memory Issues

**Problem**: High memory usage or out of memory errors

**Solution**:
```bash
# Monitor memory usage
docker stats rag-system

# Reduce batch sizes in config.yaml:
embedding:
  config:
    batch_size: 50  # Reduce from 100

ingestion:
  parsing:
    vision:
      max_concurrent_pages: 3  # Reduce from 8
```

### Debug Commands

```bash
# Check all services
curl "http://localhost:8000/health" && echo
curl "http://localhost:8001/health" && echo

# View application logs
tail -f logs/application.log

# Check Docker containers
docker ps
docker logs rag-system-ingestion
docker logs rag-system-chatbot

# Database debugging
psql $POSTGRES_CONNECTION_STRING -c "
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public';
"
```

## Next Steps

Congratulations! You've successfully set up and tested the complete RAG system. Here are some next steps:

### 1. Explore Advanced Features
- Try different AI models and compare performance
- Experiment with custom prompt templates
- Set up monitoring dashboards with Grafana

### 2. Customize for Your Use Case
- Add custom document parsers for specific formats
- Implement custom retrieval strategies
- Create domain-specific prompt templates

### 3. Scale for Production
- Set up proper monitoring and alerting
- Implement rate limiting and authentication
- Configure auto-scaling based on load

### 4. Learn More
- Read the [Custom Pipeline Tutorial](custom-pipeline.md)
- Explore [API Examples](api-examples.md)
- Check out [Integration Examples](integrations.md)

### 5. Join the Community
- Report issues on GitHub
- Contribute improvements
- Share your use cases and configurations

## Summary

You've completed a comprehensive walkthrough of the RAG system, including:

✅ **Environment Setup**: Installed dependencies and configured authentication  
✅ **Database Configuration**: Set up PostgreSQL with pgvector  
✅ **Document Ingestion**: Uploaded and processed documents with vision parsing  
✅ **Chatbot Interaction**: Tested queries with retrieval and chat history  
✅ **Advanced Features**: Explored parallel processing and multi-provider models  
✅ **Monitoring**: Set up health checks and performance monitoring  
✅ **Production Deployment**: Configured Docker and Kubernetes deployment  

The system is now ready for your specific use case. Customize the configuration, add your documents, and start building intelligent applications with RAG!

For additional help, refer to the other documentation files or reach out to the community.
