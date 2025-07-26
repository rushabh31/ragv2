# Development Guide

This guide provides comprehensive information for developers working on the RAG system, including setup, development workflows, testing strategies, debugging techniques, and contribution guidelines.

## Overview

The RAG system is built with modern Python development practices, emphasizing:

- **Modular Architecture**: Clean separation of concerns with factory patterns
- **Type Safety**: Full type hints and static analysis
- **Testing**: Comprehensive test coverage with multiple testing strategies
- **Documentation**: Extensive documentation with examples
- **Performance**: Optimized for production workloads
- **Maintainability**: Clean code practices and consistent patterns

## Development Environment Setup

### Prerequisites

**Required Software:**
- Python 3.8+ (recommended: Python 3.11)
- Git
- PostgreSQL 12+ (for memory backend)
- Docker (optional, for containerized development)

**System Dependencies:**
```bash
# macOS
brew install python@3.11 postgresql git

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv postgresql git

# Install additional tools
pip install poetry  # For dependency management
```

### Project Setup

**1. Clone Repository:**
```bash
git clone <repository-url>
cd controlsgenai
```

**2. Create Virtual Environment:**
```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using poetry (recommended)
poetry install
poetry shell
```

**3. Install Dependencies:**
```bash
# Using pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Using poetry
poetry install --with dev
```

**4. Environment Configuration:**
```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**
```bash
# Universal Authentication
COIN_CONSUMER_ENDPOINT_URL=https://oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# Google Cloud
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Optional API Keys
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key

# PostgreSQL (for memory backend)
POSTGRES_CONNECTION_STRING=postgresql://username:password@localhost:5432/langgraph_db
```

**5. Database Setup:**
```bash
# Create PostgreSQL database
createdb langgraph_db

# Install pgvector extension (if using vector storage)
psql langgraph_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run database migrations (if any)
python scripts/setup_database.py
```

## Project Structure

```
controlsgenai/
├── src/                          # Source code
│   ├── models/                   # Model factories and implementations
│   │   ├── generation/           # Generation models
│   │   ├── embedding/            # Embedding models
│   │   └── vision/               # Vision models
│   ├── rag/                      # RAG system components
│   │   ├── ingestion/            # Document ingestion pipeline
│   │   │   ├── parsers/          # Document parsers
│   │   │   ├── chunkers/         # Text chunkers
│   │   │   ├── embedders/        # Embedding generators
│   │   │   └── vector_stores/    # Vector storage
│   │   └── chatbot/              # Chatbot service
│   │       ├── retrievers/       # Document retrievers
│   │       ├── rerankers/        # Result rerankers
│   │       ├── generators/       # Response generators
│   │       └── memory/           # Memory systems
│   └── shared/                   # Shared utilities
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── examples/                     # Example applications
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── config/                       # Configuration files
└── requirements.txt              # Dependencies
```

## Development Workflow

### Code Style and Standards

**1. Code Formatting:**
```bash
# Install pre-commit hooks
pre-commit install

# Format code with black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Lint with flake8
flake8 src/ tests/ examples/
```

**2. Type Checking:**
```bash
# Run mypy for type checking
mypy src/

# Check specific module
mypy src/models/generation/
```

**3. Code Quality:**
```bash
# Run all quality checks
make lint  # or
./scripts/lint.sh

# Fix common issues automatically
make format  # or
./scripts/format.sh
```

### Git Workflow

**1. Branch Naming:**
```bash
# Feature branches
git checkout -b feature/add-new-embedder
git checkout -b feature/improve-memory-performance

# Bug fixes
git checkout -b fix/authentication-error
git checkout -b fix/memory-leak-in-parser

# Documentation
git checkout -b docs/update-api-reference
```

**2. Commit Messages:**
```bash
# Good commit message format
git commit -m "feat: add support for Azure OpenAI embeddings

- Implement AzureOpenAIEmbeddingAI class
- Add factory registration
- Include configuration examples
- Add comprehensive tests

Closes #123"

# Types: feat, fix, docs, style, refactor, test, chore
```

### Testing Strategy

**1. Unit Tests:**
```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/models/test_generation.py

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

**2. Integration Tests:**
```bash
# Run integration tests
pytest tests/integration/

# Test specific component
pytest tests/integration/test_rag_pipeline.py

# Run with real APIs (requires credentials)
pytest tests/integration/ --real-apis
```

**3. End-to-End Tests:**
```bash
# Run E2E tests
pytest tests/e2e/

# Test complete workflows
pytest tests/e2e/test_document_ingestion_workflow.py
```

### Local Development Servers

**1. Start Ingestion Service:**
```bash
# Development mode with auto-reload
cd examples/rag/ingestion
python -m uvicorn api.main:app --reload --port 8000

# Or using the script
./scripts/start_ingestion_dev.sh
```

**2. Start Chatbot Service:**
```bash
# Development mode
cd examples/rag/chatbot
python -m uvicorn api.main:app --reload --port 8001

# Or using the script
./scripts/start_chatbot_dev.sh
```

**3. Start Both Services:**
```bash
# Using docker-compose for full stack
docker-compose -f docker-compose.dev.yml up

# Or using the development script
./scripts/start_dev_stack.sh
```

## Testing Guidelines

### Writing Unit Tests

**1. Test Structure:**
```python
# tests/unit/models/test_generation.py
import pytest
from unittest.mock import AsyncMock, patch
from src.models.generation import VertexGenAI

class TestVertexGenAI:
    @pytest.fixture
    def model_config(self):
        return {
            "model_name": "gemini-1.5-pro-002",
            "max_tokens": 4096,
            "temperature": 0.7
        }
    
    @pytest.fixture
    def vertex_model(self, model_config):
        return VertexGenAI(**model_config)
    
    @pytest.mark.asyncio
    async def test_generate_content_success(self, vertex_model):
        # Arrange
        prompt = "Test prompt"
        expected_response = "Test response"
        
        with patch.object(vertex_model, '_make_api_call') as mock_call:
            mock_call.return_value = expected_response
            
            # Act
            result = await vertex_model.generate_content(prompt)
            
            # Assert
            assert result == expected_response
            mock_call.assert_called_once_with(prompt)
```

**2. Mocking External APIs:**
```python
import pytest
from unittest.mock import patch, AsyncMock

class TestVertexGenAI:
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_api_call_success(self, mock_post, vertex_model):
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Response"}]}}]
        }
        mock_post.return_value = mock_response
        
        result = await vertex_model.generate_content("Test prompt")
        
        assert result == "Response"
        mock_post.assert_called_once()
```

### Integration Testing

**1. Component Integration:**
```python
# tests/integration/test_rag_pipeline.py
import pytest
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder

class TestRAGPipeline:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_ingestion_pipeline(self, sample_pdf):
        # Test complete ingestion pipeline
        parser = VisionParser({"model": "gemini-1.5-pro-002"})
        embedder = VertexEmbedder({"model": "text-embedding-004"})
        
        # Parse document
        documents = await parser.parse_file(sample_pdf, {"source": "test"})
        assert len(documents) > 0
        
        # Generate embeddings
        embedded_docs = await embedder.embed_documents(documents)
        assert len(embedded_docs) == len(documents)
```

## Debugging

### Logging Configuration

**1. Development Logging:**
```python
# config/logging_dev.yaml
version: 1
disable_existing_loggers: false

formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: detailed
    stream: ext://sys.stdout

loggers:
  src.models:
    level: DEBUG
    handlers: [console]
    propagate: false
  
  src.rag:
    level: DEBUG
    handlers: [console]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

**2. Debugging Tools:**
```python
# Use pdb for debugging
import pdb; pdb.set_trace()

# Use breakpoint() (Python 3.7+)
breakpoint()

# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Common Debugging Scenarios

**1. Authentication Issues:**
```python
async def debug_authentication():
    from src.models.generation import VertexGenAI
    
    model = VertexGenAI()
    
    try:
        # Test token generation
        token = await model.get_coin_token()
        print(f"Token obtained: {token[:50]}...")
        
        # Test health status
        health = await model.get_auth_health_status()
        print(f"Auth health: {health}")
        
    except Exception as e:
        print(f"Authentication failed: {e}")
        import traceback
        traceback.print_exc()
```

**2. Memory Issues:**
```python
import psutil
import gc

def debug_memory_usage():
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"Memory percent: {process.memory_percent():.2f}%")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Garbage collected: {collected} objects")
```

## Best Practices

### Code Organization

**1. Import Organization:**
```python
# Standard library imports
import asyncio
import logging
from typing import Dict, List, Optional, Any

# Third-party imports
import httpx
import yaml
from pydantic import BaseModel

# Local imports
from src.models.base import BaseModel
from src.models.exceptions import AuthenticationError
```

**2. Error Handling:**
```python
# Custom exceptions
class RAGException(Exception):
    """Base exception for RAG system."""
    pass

class AuthenticationError(RAGException):
    """Authentication failed."""
    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.provider = provider

# Good error handling pattern
async def robust_api_call(self, prompt: str) -> str:
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return await self._make_api_call(prompt)
            
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            delay = e.retry_after or (base_delay * (2 ** attempt))
            await asyncio.sleep(delay)
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
```

**3. Performance Optimization:**
```python
# Good async patterns
import asyncio

async def process_documents_efficiently(documents: List[str]) -> List[str]:
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(10)
    
    async def process_single_document(doc: str) -> str:
        async with semaphore:
            return await some_async_operation(doc)
    
    # Process all documents concurrently
    tasks = [process_single_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    return [result for result in results if not isinstance(result, Exception)]
```

## Contributing

### Contribution Guidelines

**1. Development Process:**
```bash
# 1. Fork and clone
git clone https://github.com/your-username/controlsgenai.git
cd controlsgenai

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and test
make test
make lint

# 4. Commit and push
git commit -m "feat: add your feature"
git push origin feature/your-feature-name

# 5. Create pull request
```

**2. Code Review Checklist:**
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Release Process

**1. Version Bumping:**
```bash
# Update version in setup.py, __init__.py
# Follow semantic versioning: MAJOR.MINOR.PATCH

# Create release branch
git checkout -b release/v1.2.0

# Update CHANGELOG.md
# Tag release
git tag v1.2.0
git push origin v1.2.0
```

**2. Release Checklist:**
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Security review completed
- [ ] Performance benchmarks run

## Related Documentation

- [Testing Guide](./testing.md) - Detailed testing strategies
- [Configuration Guide](../configuration.md) - Configuration reference
- [API Documentation](../rag/chatbot/api.md) - API reference
- [Deployment Guide](../deployment/README.md) - Deployment instructions

## Useful Commands

```bash
# Development commands
make install          # Install dependencies
make test            # Run all tests
make lint            # Run linting
make format          # Format code
make docs            # Build documentation
make clean           # Clean build artifacts

# Testing commands
pytest tests/unit/                    # Unit tests
pytest tests/integration/            # Integration tests
pytest tests/e2e/                    # End-to-end tests
pytest --cov=src --cov-report=html   # Coverage report

# Development servers
./scripts/start_ingestion_dev.sh     # Start ingestion service
./scripts/start_chatbot_dev.sh       # Start chatbot service
./scripts/start_dev_stack.sh         # Start full development stack
```
