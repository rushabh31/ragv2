# Getting Started Guide

## 🎯 Welcome to RAG System

This guide will walk you through setting up and running your first RAG (Retrieval-Augmented Generation) system. By the end of this tutorial, you'll have a fully functional AI assistant that can answer questions based on your own documents.

## 📋 Prerequisites

### **System Requirements**
- **Python 3.8+** (Python 3.9+ recommended)
- **PostgreSQL 12+** (for persistent memory)
- **4GB+ RAM** (8GB+ recommended for large documents)
- **2GB+ disk space** (for models and vector storage)

### **Required Accounts**
- **Google Cloud Platform** account with Vertex AI API enabled
- **PostgreSQL database** (local or cloud)

## 🚀 Quick Start (5 Minutes)

### **Step 1: Clone and Install**

```bash
# Clone the repository
git clone <repository-url>
cd controlsgenai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Environment Setup**

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google Cloud / Vertex AI Configuration
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Authentication (if using OAuth)
COIN_CONSUMER_ENDPOINT_URL=https://your-oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# PostgreSQL (for persistent memory)
POSTGRES_CONNECTION_STRING=postgresql://username:password@localhost:5432/ragdb
```

### **Step 3: Database Setup**

```bash
# Create PostgreSQL database
createdb ragdb

# The system will automatically create required tables on first run
```

### **Step 4: Test Installation**

```bash
# Run the installation test
python test_installation.py
```

You should see output like:
```
✅ All core modules imported successfully
✅ Configuration files are valid
✅ Environment variables are set
✅ Basic functionality test passed
🎉 Installation test completed successfully!
```

## 📄 Your First Document Processing

### **Step 1: Prepare Sample Documents**

Create a `documents/` folder and add some sample files:

```bash
mkdir documents
# Add your PDF, Word, or text files to this folder
```

### **Step 2: Start the Ingestion Service**

```bash
# Navigate to ingestion example
cd examples/rag/ingestion

# Start the ingestion API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The ingestion service will start at `http://localhost:8000`

### **Step 3: Upload Your First Document**

Using curl:
```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@documents/your-document.pdf" \
  -F "metadata={\"source\": \"user_upload\", \"category\": \"general\"}"
```

Using Python:
```python
import requests

# Upload a document
with open('documents/your-document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ingest/file',
        headers={'X-API-Key': 'test-api-key'},
        files={'file': f},
        data={'metadata': '{"source": "user_upload"}'}
    )

print(f"Upload status: {response.status_code}")
print(f"Response: {response.json()}")
```

### **Step 4: Start the Chatbot Service**

Open a new terminal:

```bash
# Navigate to chatbot example
cd examples/rag/chatbot

# Start the chatbot API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

The chatbot service will start at `http://localhost:8001`

### **Step 5: Ask Your First Question**

Using curl:
```bash
curl -X POST "http://localhost:8001/chat/message" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: test-user-123" \
  -d "query=What is this document about?&use_retrieval=true"
```

Using Python:
```python
import requests

response = requests.post(
    'http://localhost:8001/chat/message/json',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': 'test-api-key',
        'soeid': 'test-user-123'
    },
    json={
        'query': 'What is this document about?',
        'use_retrieval': True
    }
)

print(f"AI Response: {response.json()['response']}")
```

## 🎉 Congratulations!

You now have a working RAG system! The AI can answer questions based on your uploaded documents.

## 🔧 Configuration Customization

### **Using Different AI Providers**

The system supports multiple AI providers. Edit `config.yaml`:

```yaml
# For Vertex AI (default, recommended)
chatbot:
  generation:
    provider: "vertex"
    config:
      model_name: "gemini-1.5-pro-002"
      max_tokens: 1000

# For OpenAI (requires OPENAI_API_KEY)
chatbot:
  generation:
    provider: "openai"
    config:
      model: "gpt-4"
      max_tokens: 1000
```

### **Adjusting Performance Settings**

For better performance with large documents:

```yaml
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 8  # Process more pages in parallel
  chunking:
    semantic:
      chunk_size: 1000
      overlap: 200
  embedding:
    vertex:
      batch_size: 100  # Process more texts at once
```

## 🚨 Troubleshooting Common Issues

### **Authentication Errors**

```bash
# Error: "Authentication failed"
# Solution: Check your .env file and GCP credentials

# Test authentication
python -c "
from src.models.generation import VertexGenAI
model = VertexGenAI()
print('Auth status:', model.get_auth_health_status())
"
```

### **Database Connection Issues**

```bash
# Error: "could not connect to server"
# Solution: Check PostgreSQL is running

# Test database connection
psql $POSTGRES_CONNECTION_STRING -c "SELECT 1;"
```

### **Import Errors**

```bash
# Error: "ModuleNotFoundError"
# Solution: Ensure you're in the project root and virtual environment is activated

pwd  # Should show /path/to/controlsgenai
which python  # Should show virtual environment path
```

### **Memory Issues**

```bash
# Error: "Out of memory"
# Solution: Reduce batch sizes and concurrency

# Edit config.yaml
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 2  # Reduce from default 5
  embedding:
    vertex:
      batch_size: 10  # Reduce from default 100
```

## 📚 Next Steps

### **Learn More**
1. **[Configuration Guide](./configuration.md)** - Detailed configuration options
2. **[Complete Walkthrough](./tutorials/complete-walkthrough.md)** - Advanced features
3. **[API Examples](./tutorials/api-examples.md)** - Integration patterns

### **Customize Your System**
1. **[Ingestion Pipeline](./rag/ingestion/README.md)** - Document processing options
2. **[Chatbot Service](./rag/chatbot/README.md)** - Response generation customization
3. **[Memory System](./memory/README.md)** - Conversation management

### **Production Deployment**
1. **[Deployment Guide](./deployment/README.md)** - Production setup
2. **[Monitoring Guide](./deployment/monitoring.md)** - System monitoring
3. **[Testing Guide](./development/testing.md)** - Quality assurance

## 🎯 Quick Reference

### **Common Commands**
```bash
# Start ingestion service
cd examples/rag/ingestion && python -m uvicorn api.main:app --port 8000

# Start chatbot service  
cd examples/rag/chatbot && python -m uvicorn api.main:app --port 8001

# Run tests
python run_tests.py --all

# Check system health
python test_installation.py
```

### **Key URLs**
- **Ingestion API**: http://localhost:8000/docs
- **Chatbot API**: http://localhost:8001/docs
- **Health Check**: http://localhost:8000/health

### **Configuration Files**
- **Main Config**: `config.yaml`
- **Ingestion Config**: `examples/rag/ingestion/config.yaml`
- **Chatbot Config**: `examples/rag/chatbot/config.yaml`
- **Environment**: `.env`

---

**Need Help?** Check the troubleshooting section above or refer to the [Configuration Guide](./configuration.md) for detailed setup options.
