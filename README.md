# ControlsGenAI - Production-Grade RAG System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system with multi-provider AI model support, enterprise-grade authentication, and advanced document processing capabilities.

## 🚀 Features

- **Multi-Provider AI Support**: Vertex AI, OpenAI, Azure OpenAI, Anthropic, Groq
- **Universal Authentication**: Enterprise OAuth2 token management
- **Advanced Document Processing**: Vision-based PDF parsing with parallel processing
- **Flexible Memory Systems**: PostgreSQL, LangGraph checkpoints, in-memory
- **Production-Ready**: Comprehensive monitoring, logging, and error handling
- **Scalable Architecture**: Factory-based model selection and configuration-driven setup

## 📋 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd controlsgenai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Basic Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure your environment variables:
```bash
# Authentication
COIN_CONSUMER_ENDPOINT_URL=https://your-oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# Google Cloud
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Database
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname
```

### Usage Example

```python
from src.rag.ingestion.api.main import IngestionAPI
from src.rag.chatbot.api.main import ChatbotAPI

# Initialize services
ingestion = IngestionAPI()
chatbot = ChatbotAPI()

# Ingest documents
await ingestion.ingest_document("path/to/document.pdf")

# Query the system
response = await chatbot.query("What is the main topic of the document?")
print(response)
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Getting Started Guide](docs/getting-started.md)
- [Configuration Guide](docs/configuration.md)
- [Development Guide](docs/development/README.md)
- [Deployment Guide](docs/deployment/README.md)
- [API Examples](docs/tutorials/api-examples.md)
- [Complete Walkthrough](docs/tutorials/complete-walkthrough.md)

## 🏗️ Architecture

The system is built with a modular, factory-based architecture:

```
src/
├── models/              # Multi-provider AI models
│   ├── generation/      # Text generation models
│   ├── embedding/       # Embedding models
│   └── vision/          # Vision models
├── rag/
│   ├── ingestion/       # Document processing pipeline
│   ├── chatbot/         # Query and response generation
│   └── memory/          # Conversation memory systems
└── shared/              # Common utilities and authentication
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --category unit
python run_tests.py --category integration
python run_tests.py --category performance

# Check environment setup
python run_tests.py --check-env
```

## 🚀 Deployment

The system supports multiple deployment options:

- **Local Development**: Direct Python execution
- **Docker**: Containerized deployment
- **Kubernetes**: Scalable cloud deployment
- **Cloud Functions**: Serverless ingestion automation

See the [Deployment Guide](docs/deployment/README.md) for detailed instructions.

## 🔧 Configuration

The system uses YAML configuration files for flexible setup:

```yaml
# config.yaml
generation:
  provider: vertex_ai
  config:
    model_name: gemini-1.5-pro-002
    max_tokens: 8192

embedding:
  provider: vertex_ai
  config:
    model: text-embedding-004
    batch_size: 100

vision:
  provider: vertex_ai
  config:
    model: gemini-1.5-pro-002
    max_concurrent_pages: 5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python run_tests.py --all`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the [troubleshooting guide](docs/tutorials/complete-walkthrough.md#troubleshooting)
- Review [common issues](docs/development/testing.md#common-issues)
- See [API examples](docs/tutorials/api-examples.md) for usage patterns

## 🏆 Features Highlights

- **Parallel Processing**: 2-5x speedup for multi-page document processing
- **Universal Authentication**: Single OAuth2 system across all providers
- **Enterprise Ready**: Comprehensive monitoring, logging, and error handling
- **Flexible Memory**: Multiple storage backends with session management
- **Multi-Modal**: Text, vision, and structured data processing
- **Production Monitoring**: Prometheus metrics, health checks, and alerting
