# 🚀 Enterprise RAG System

**A production-grade Retrieval-Augmented Generation (RAG) system with multi-provider AI support, parallel processing, and enterprise-ready features.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [📖 Usage Guide](#-usage-guide)
- [🧪 Testing](#-testing)
- [📚 Examples](#-examples)
- [🔧 Development](#-development)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

## ✨ Features

### 🤖 Multi-Provider AI Support
- **Generation Models**: Vertex AI (Gemini), Groq (Llama), OpenAI (GPT), Azure OpenAI, Anthropic (Claude)
- **Vision Models**: Vertex AI (Gemini Vision), Groq (Llama Vision), OpenAI (GPT-4V)
- **Embedding Models**: Vertex AI, OpenAI, Azure OpenAI, Sentence Transformers (local)
- **Runtime Provider Switching**: Change providers via configuration without code changes

### ⚡ High-Performance Processing
- **Parallel Document Processing**: Process multiple PDF pages simultaneously (2-5x speedup)
- **Configurable Concurrency**: Tune parallel processing based on your resources
- **Async Architecture**: Non-blocking operations throughout the system
- **Efficient Caching**: Local and Redis caching with TTL and LRU eviction

### 🏢 Enterprise-Ready Features
- **LangGraph Memory System**: Advanced conversation memory with PostgreSQL support
- **SOEID User Tracking**: Enterprise user identification across sessions
- **Universal Authentication**: Unified auth system across all providers
- **Rate Limiting**: Configurable API rate limiting and quotas
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

### 🔧 Production Architecture
- **Factory Pattern**: Consistent model instantiation across providers
- **Middleware Support**: CORS, authentication, rate limiting, request logging
- **Error Handling**: Multi-level fallback mechanisms and graceful degradation
- **Health Monitoring**: Authentication validation and system health checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion API (8000)           │  Chatbot API (8001)           │
│  • Document Upload              │  • Chat Processing            │
│  • Batch Processing             │  • Session Management         │
│  • Parallel Parsing             │  • Memory Operations          │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion Service              │  Chatbot Service              │
│  • Document Processing         │  • Retrieval & Generation     │
│  • Embedding Generation        │  • Memory Management          │
│  • Vector Storage              │  • Response Assembly          │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Components                            │
├─────────────────────────────────────────────────────────────────┤
│  Parsers                       │  Models                        │
│  • VisionParser (Vertex AI)    │  • Generation Factory         │
│  • GroqVisionParser (Groq)     │  • Embedding Factory          │
│  • OpenAIVisionParser (OpenAI) │  • Vision Factory             │
│  • SimpleTextParser            │  • Universal Auth Manager     │
├─────────────────────────────────────────────────────────────────┤
│  Storage & Memory              │  Processing                    │
│  • Vector Store (FAISS)        │  • Parallel Processing        │
│  • LangGraph Memory            │  • Chunking Strategies        │
│  • PostgreSQL Support          │  • Reranking                  │
│  • Cache Manager               │  • Retrieval                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Document Ingestion
```
PDF/Document → Parallel Vision Parsing → Chunking → Embedding → Vector Store
```

#### Query Processing
```
User Query → Embedding → Vector Retrieval → Reranking → Context Assembly → Generation → Response
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **API Keys** for your chosen providers:
  - Vertex AI: Google Cloud credentials
  - Groq: `GROQ_API_KEY`
  - OpenAI: OpenAI API key
  - Azure OpenAI: Azure credentials
- **Optional**: PostgreSQL for production memory storage
- **Optional**: Redis for distributed caching

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/controlsgenai.git
   cd controlsgenai
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Configure the system**:
   ```bash
   cp config_sample.yaml config.yaml
   # Edit config.yaml with your preferred settings
   ```

### Quick Test

```bash
# Test the installation
python test_installation.py

# Run comprehensive system tests
python tests/test_complete_system.py
```

### Start the Services

```bash
# Terminal 1: Start Ingestion API
python examples/rag/ingestion/run_ingestion.py

# Terminal 2: Start Chatbot API  
python examples/rag/chatbot/run_chatbot.py
```

Your RAG system is now running! 🎉

- **Ingestion API**: http://localhost:8000
- **Chatbot API**: http://localhost:8001

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your API keys:

```bash
# Required for Vertex AI
COIN_CONSUMER_ENDPOINT_URL=https://your-oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform
PROJECT_ID=your-gcp-project-id

# Required for Groq
GROQ_API_KEY=your-groq-api-key

# Optional: For OpenAI direct usage
OPENAI_API_KEY=your-openai-api-key

# Optional: For PostgreSQL memory
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb

# Optional: For Redis caching
REDIS_URL=redis://localhost:6379
```

### Main Configuration (`config.yaml`)

```yaml
# System settings
system:
  log_level: INFO
  environment: development

# Vision model configuration (for document parsing)
vision:
  provider: vertex_ai  # Options: vertex_ai, groq
  config:
    model: gemini-1.5-pro-002
    region: us-central1

# Generation model configuration (for chat responses)
generation:
  provider: groq  # Options: vertex, groq, openai, azure_openai, anthropic_vertex
  config:
    model_name: meta-llama/llama-4-scout-17b-16e-instruct
    temperature: 0.1
    max_tokens: 2048

# Embedding model configuration
embedding:
  provider: vertex_ai  # Options: vertex_ai, openai_universal, azure_openai, sentence_transformer
  config:
    model: text-embedding-004
    batch_size: 100

# Document processing settings
ingestion:
  parsing:
    default_parser: groq_vision_parser  # Options: vision_parser, groq_vision_parser, openai_vision_parser
    vision:
      max_pages: 100
      max_concurrent_pages: 5  # Parallel processing setting
  chunking:
    strategy: fixed_size
    chunk_size: 1000
    chunk_overlap: 200
  vector_store:
    type: faiss
    path: ./data/vector_store

# Chatbot settings
chatbot:
  memory:
    type: langgraph_checkpoint  # Options: simple, langgraph_checkpoint
    store_type: in_memory  # Options: in_memory, postgres
    postgres:
      connection_string: "${DATABASE_URL}"
  retrieval:
    top_k: 5
    min_score: 0.7
  reranking:
    enabled: true
    top_k: 3

# Security settings
security:
  api_keys:
    - name: default
      key: "${API_KEY}"
      roles: [user]
    - name: admin
      key: "${ADMIN_API_KEY}"
      roles: [user, admin]
  rate_limiting:
    enabled: true
    requests: 100
    period_seconds: 60

# Cache settings
cache:
  enabled: true
  provider: local  # Options: local, redis
  default_ttl_seconds: 3600
  max_size: 10000
```

### Provider-Specific Configurations

#### Groq Configuration
```yaml
generation:
  provider: groq
  config:
    model_name: meta-llama/llama-4-scout-17b-16e-instruct
    temperature: 0.1
    max_tokens: 2048

parser:
  provider: groq_vision_parser
  config:
    model_name: llama-3.2-11b-vision-preview
    prompt_template: "Extract and structure the text content from this document."
    max_concurrent_pages: 5
```

#### Vertex AI Configuration
```yaml
generation:
  provider: vertex
  config:
    model_name: gemini-1.5-pro-002
    temperature: 0.1
    max_tokens: 2048

embedding:
  provider: vertex_ai
  config:
    model: text-embedding-004
    project_id: "${PROJECT_ID}"
    location: us-central1
```

#### Sentence Transformers (Local)
```yaml
embedding:
  provider: sentence_transformer
  config:
    model: all-mpnet-base-v2
    device: cpu  # or cuda
    normalize_embeddings: true
```

## 📖 Usage Guide

### Document Upload

```bash
# Upload a single document
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "metadata={\"title\": \"My Document\"}"

# Upload multiple documents
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "X-API-Key: your-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf"
```

### Chat Interaction

```bash
# Send a chat message
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "soeid: user123" \
  -d '{
    "query": "What is machine learning?",
    "use_retrieval": true,
    "session_id": "session456"
  }'

# Get user chat history
curl -X GET "http://localhost:8001/chat/history/user123" \
  -H "X-API-Key: your-api-key"

# Get memory statistics
curl -X GET "http://localhost:8001/chat/memory/stats" \
  -H "X-API-Key: your-api-key"
```

### Python SDK Usage

```python
import asyncio
from src.rag.ingestion.parsers.groq_vision_parser import GroqVisionParser
from src.rag.chatbot.generators.groq_generator import GroqGenerator

async def main():
    # Parse a document with parallel processing
    parser = GroqVisionParser({
        "model_name": "llama-3.2-11b-vision-preview",
        "max_concurrent_pages": 8
    })
    documents = await parser._parse_file("document.pdf", {})
    
    # Generate a response
    generator = GroqGenerator({
        "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
    })
    response = await generator.generate_response(
        query="Summarize the document",
        documents=documents,
        conversation_history=[]
    )
    
    print(response)

asyncio.run(main())
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_parsers.py
│   ├── test_generators.py
│   └── test_embedders.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_memory_system.py
│   └── test_parallel_processing.py
├── performance/             # Performance tests
│   ├── test_parallel_performance.py
│   └── test_memory_performance.py
└── system/                  # System tests
    ├── test_complete_system.py
    └── test_authentication.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/

# Run system tests
python tests/system/test_complete_system.py

# Test parallel processing
python tests/performance/test_parallel_performance.py

# Test memory system
python tests/integration/test_memory_system.py

# Test authentication
python tests/system/test_authentication.py
```

### Performance Testing

```bash
# Test all parallel parsers
python tests/performance/test_parallel_performance.py

# Test memory performance
python tests/performance/test_memory_performance.py

# Test API performance
python tests/performance/test_api_performance.py
```

## 📚 Examples

### Basic Examples

- **[Ingestion Example](examples/rag/ingestion/)**: Document processing and embedding
- **[Chatbot Example](examples/rag/chatbot/)**: Conversational AI with memory
- **[Configuration Examples](examples/configs/)**: Various provider configurations

### Advanced Examples

- **[Parallel Processing](examples/advanced/parallel_processing.py)**: High-performance document processing
- **[Multi-Provider Setup](examples/advanced/multi_provider.py)**: Using multiple AI providers
- **[Memory Management](examples/advanced/memory_management.py)**: Advanced conversation memory
- **[Custom Parsers](examples/advanced/custom_parsers.py)**: Creating custom document parsers

### Use Case Examples

- **[Enterprise Chatbot](examples/use_cases/enterprise_chatbot.py)**: Full-featured enterprise chatbot
- **[Document Analysis](examples/use_cases/document_analysis.py)**: Automated document analysis
- **[Knowledge Base](examples/use_cases/knowledge_base.py)**: Building a knowledge base system

## 🔧 Development

### Project Structure

```
controlsgenai/
├── src/
│   ├── models/                 # AI model implementations
│   │   ├── generation/         # Generation models
│   │   ├── embedding/          # Embedding models
│   │   └── vision/             # Vision models
│   ├── rag/
│   │   ├── ingestion/          # Document processing
│   │   │   ├── parsers/        # Document parsers
│   │   │   ├── embedders/      # Embedding generators
│   │   │   └── chunkers/       # Text chunking
│   │   ├── chatbot/            # Chatbot functionality
│   │   │   ├── generators/     # Response generators
│   │   │   ├── retrievers/     # Document retrievers
│   │   │   ├── memory/         # Conversation memory
│   │   │   └── workflow/       # LangGraph workflows
│   │   └── shared/             # Shared utilities
│   └── utils/                  # Authentication & utilities
├── examples/                   # Usage examples
├── tests/                      # Test suite
├── docs/                       # Documentation
└── config/                     # Configuration files
```

### Adding New Providers

1. **Create Model Implementation**:
   ```python
   # src/models/generation/new_provider.py
   from .base import BaseGenerationModel
   
   class NewProviderGenAI(BaseGenerationModel):
       async def generate_content(self, prompt: str) -> str:
           # Implementation
           pass
   ```

2. **Update Factory**:
   ```python
   # src/models/generation/generation_factory.py
   from .new_provider import NewProviderGenAI
   
   class GenerationProvider(Enum):
       NEW_PROVIDER = "new_provider"
   
   MODEL_REGISTRY = {
       GenerationProvider.NEW_PROVIDER: NewProviderGenAI,
   }
   ```

3. **Create RAG Integration**:
   ```python
   # src/rag/chatbot/generators/new_provider_generator.py
   from ..base_generator import BaseGenerator
   
   class NewProviderGenerator(BaseGenerator):
       # Implementation
       pass
   ```

4. **Update Configuration**:
   ```yaml
   generation:
     provider: new_provider
     config:
       model_name: new-model-name
   ```

### Code Style

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

### Documentation

```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/

# Update README
# Edit this file and relevant example READMEs
```

## 🐛 Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Check environment variables
python -c "import os; print(os.getenv('GROQ_API_KEY'))"

# Test authentication
python tests/system/test_authentication.py
```

#### Memory Issues
```bash
# Check memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# Reduce concurrency
# Edit config.yaml: max_concurrent_pages: 2
```

#### Performance Issues
```bash
# Test parallel processing
python tests/performance/test_parallel_performance.py

# Monitor API calls
tail -f logs/api.log
```

#### Model Availability
```bash
# Test model access
python tests/system/test_model_availability.py

# Check quotas
# Review your provider's usage dashboard
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python examples/rag/chatbot/run_chatbot.py --verbose

# Check system health
curl http://localhost:8001/health
```

### Performance Tuning

#### Parallel Processing
- **Small documents (1-5 pages)**: `max_concurrent_pages: 2-3`
- **Medium documents (6-20 pages)**: `max_concurrent_pages: 5-8`
- **Large documents (20+ pages)**: `max_concurrent_pages: 8-12`

#### Memory Settings
- **Development**: `store_type: in_memory`
- **Production**: `store_type: postgres`
- **High traffic**: Enable Redis caching

#### API Rate Limits
- **Groq**: 30 requests/minute (free tier)
- **Vertex AI**: Varies by model and project
- **OpenAI**: Varies by tier

### Getting Help

1. **Check the logs**: `tail -f logs/system.log`
2. **Run diagnostics**: `python tests/system/test_diagnostics.py`
3. **Check documentation**: `docs/`
4. **Search issues**: GitHub Issues
5. **Ask for help**: Create a new issue with:
   - Error message
   - Configuration
   - Steps to reproduce
   - System information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests**: Ensure your changes are tested
5. **Run the test suite**: `python -m pytest`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/controlsgenai.git
cd controlsgenai

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for advanced conversation memory
- **Vertex AI** for powerful language models
- **Groq** for high-performance inference
- **Sentence Transformers** for local embeddings
- **FAISS** for efficient vector search
- **FastAPI** for the API framework

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/controlsgenai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/controlsgenai/discussions)

---

**Made with ❤️ by the ControlsGenAI Team**
