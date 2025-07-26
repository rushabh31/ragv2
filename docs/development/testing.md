# Testing Guide

This guide provides comprehensive testing strategies, best practices, and implementation details for the RAG system. It covers unit testing, integration testing, end-to-end testing, performance testing, and testing automation.

## Overview

The RAG system employs a multi-layered testing strategy to ensure reliability, performance, and maintainability:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **End-to-End Tests**: Test complete user scenarios
- **Performance Tests**: Benchmark and load testing
- **Contract Tests**: API contract validation
- **Security Tests**: Authentication and authorization testing

## Testing Framework and Tools

### Core Testing Stack

**Primary Framework:**
- `pytest` - Main testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Enhanced mocking capabilities

**Additional Tools:**
- `httpx` - HTTP client for API testing
- `factory_boy` - Test data factories
- `freezegun` - Time mocking
- `responses` - HTTP request mocking
- `memory_profiler` - Memory usage testing

### Installation

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install httpx factory-boy freezegun responses memory-profiler

# Or using poetry
poetry install --with test
```

## Test Structure and Organization

### Directory Structure

```
tests/
├── conftest.py                   # Shared fixtures and configuration
├── unit/                         # Unit tests
│   ├── models/
│   │   ├── test_generation.py
│   │   ├── test_embedding.py
│   │   └── test_vision.py
│   ├── rag/
│   │   ├── ingestion/
│   │   └── chatbot/
│   └── shared/
├── integration/                  # Integration tests
│   ├── test_ingestion_pipeline.py
│   ├── test_chatbot_pipeline.py
│   └── test_api_integration.py
├── e2e/                         # End-to-end tests
│   ├── test_document_workflow.py
│   └── test_chat_workflow.py
├── performance/                 # Performance tests
│   ├── test_embedding_performance.py
│   └── test_generation_performance.py
└── fixtures/                    # Test data and fixtures
    ├── documents/
    ├── images/
    └── configs/
```

### Test Configuration

**conftest.py:**
```python
# tests/conftest.py
import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any
from unittest.mock import AsyncMock

# Configure pytest for async testing
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Configuration fixtures
@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        "generation": {
            "provider": "vertex",
            "config": {
                "model_name": "gemini-1.5-pro-002",
                "max_tokens": 4096,
                "temperature": 0.7
            }
        },
        "embedding": {
            "provider": "vertex_ai",
            "config": {
                "model": "text-embedding-004",
                "dimensions": 768
            }
        }
    }

@pytest.fixture
def mock_generation_model():
    """Mock generation model for testing."""
    mock_model = AsyncMock()
    mock_model.generate_content.return_value = "Mock generated response"
    mock_model.chat_completion.return_value = "Mock chat response"
    mock_model.get_coin_token.return_value = "mock_token_12345"
    mock_model.validate_authentication.return_value = True
    return mock_model

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"id": "doc1", "source": "test"}
        },
        {
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"id": "doc2", "source": "test"}
        }
    ]
```

## Unit Testing

### Testing Model Components

**1. Generation Model Tests:**
```python
# tests/unit/models/test_generation.py
import pytest
from unittest.mock import AsyncMock, patch
from src.models.generation import VertexGenAI, GenerationModelFactory
from src.models.exceptions import AuthenticationError, APIError

class TestVertexGenAI:
    @pytest.fixture
    def vertex_config(self):
        return {
            "model_name": "gemini-1.5-pro-002",
            "max_tokens": 4096,
            "temperature": 0.7
        }
    
    @pytest.fixture
    def vertex_model(self, vertex_config):
        return VertexGenAI(**vertex_config)
    
    @pytest.mark.asyncio
    async def test_generate_content_success(self, vertex_model):
        """Test successful content generation."""
        prompt = "What is machine learning?"
        expected_response = "Machine learning is a subset of AI..."
        
        with patch.object(vertex_model, '_make_api_call') as mock_call:
            mock_call.return_value = expected_response
            
            result = await vertex_model.generate_content(prompt)
            
            assert result == expected_response
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, vertex_model):
        """Test authentication failure handling."""
        with patch.object(vertex_model, 'get_coin_token') as mock_token:
            mock_token.side_effect = AuthenticationError("Invalid credentials")
            
            with pytest.raises(AuthenticationError):
                await vertex_model.generate_content("test")
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, vertex_model):
        """Test rate limit handling."""
        from src.models.exceptions import RateLimitError
        
        with patch.object(vertex_model, '_make_api_call') as mock_call:
            mock_call.side_effect = RateLimitError("Rate limited", retry_after=60)
            
            with pytest.raises(RateLimitError) as exc_info:
                await vertex_model.generate_content("test")
            
            assert exc_info.value.retry_after == 60

class TestGenerationModelFactory:
    def test_create_vertex_model(self):
        """Test creating Vertex AI model via factory."""
        model = GenerationModelFactory.create_model(
            provider="vertex",
            model_name="gemini-1.5-pro-002"
        )
        
        assert isinstance(model, VertexGenAI)
        assert model.model_name == "gemini-1.5-pro-002"
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        with pytest.raises(ValueError) as exc_info:
            GenerationModelFactory.create_model(
                provider="unsupported_provider",
                model_name="test-model"
            )
        
        assert "Unsupported provider" in str(exc_info.value)
```

**2. RAG Component Tests:**
```python
# tests/unit/rag/ingestion/test_parsers.py
import pytest
from unittest.mock import AsyncMock, patch
from src.rag.ingestion.parsers.vision_parser import VisionParser

class TestVisionParser:
    @pytest.fixture
    def parser_config(self):
        return {
            "model": "gemini-1.5-pro-002",
            "max_pages": 10,
            "max_concurrent_pages": 3
        }
    
    @pytest.fixture
    def vision_parser(self, parser_config):
        with patch('src.models.vision.VisionModelFactory.create_model') as mock_factory:
            mock_model = AsyncMock()
            mock_factory.return_value = mock_model
            
            parser = VisionParser(parser_config)
            parser.vision_model = mock_model
            return parser
    
    @pytest.mark.asyncio
    async def test_parse_single_page_pdf(self, vision_parser):
        """Test parsing single page PDF."""
        expected_text = "Sample PDF content extracted by vision model"
        vision_parser.vision_model.parse_text_from_image.return_value = expected_text
        
        with patch('src.rag.ingestion.parsers.vision_parser.extract_pdf_pages') as mock_extract:
            mock_extract.return_value = ["base64_page_data"]
            
            documents = await vision_parser.parse_file(
                "test.pdf", 
                {"source": "test"}
            )
            
            assert len(documents) == 1
            assert documents[0].content == expected_text
            assert documents[0].metadata["source"] == "test"
```

## Integration Testing

### Pipeline Integration Tests

**1. Ingestion Pipeline Test:**
```python
# tests/integration/test_ingestion_pipeline.py
import pytest
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder
from src.rag.ingestion.vector_stores.faiss_store import FAISSVectorStore

class TestIngestionPipeline:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_ingestion_workflow(self, sample_pdf_path):
        """Test complete document ingestion workflow."""
        # Initialize components
        parser = VisionParser({"model": "gemini-1.5-pro-002"})
        embedder = VertexEmbedder({"model": "text-embedding-004"})
        vector_store = FAISSVectorStore({"dimension": 768})
        
        # Parse document
        documents = await parser.parse_file(sample_pdf_path, {"source": "test"})
        assert len(documents) > 0
        
        # Generate embeddings
        embedded_docs = await embedder.embed_documents(documents)
        assert len(embedded_docs) == len(documents)
        
        # Store in vector database
        for i, doc in enumerate(embedded_docs):
            await vector_store.add_document(
                document_id=f"doc_{i}",
                embedding=doc.embedding,
                metadata=doc.metadata
            )
        
        # Test retrieval
        query_embedding = await embedder.get_embedding("test query")
        results = await vector_store.search(query_embedding, top_k=3)
        assert len(results) > 0
```

**2. API Integration Tests:**
```python
# tests/integration/test_api_integration.py
import pytest
from fastapi.testclient import TestClient

class TestChatbotAPI:
    @pytest.fixture
    def client(self):
        from examples.rag.chatbot.api.main import app
        return TestClient(app)
    
    def test_chat_message_endpoint(self, client):
        response = client.post(
            "/chat/message",
            data={
                "query": "What is machine learning?",
                "use_retrieval": True
            },
            headers={"soeid": "test-user"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
    
    def test_chat_history_endpoint(self, client):
        # Send a message first
        client.post(
            "/chat/message",
            data={"query": "Hello"},
            headers={"soeid": "test-user"}
        )
        
        # Retrieve history
        response = client.get("/chat/history/test-user")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
```

## End-to-End Testing

### Complete Workflow Tests

```python
# tests/e2e/test_document_workflow.py
import pytest
from fastapi.testclient import TestClient

class TestDocumentWorkflow:
    @pytest.mark.e2e
    def test_complete_document_workflow(self, sample_pdf_path):
        """Test complete document processing and querying workflow."""
        from examples.rag.ingestion.api.main import ingestion_app
        from examples.rag.chatbot.api.main import chatbot_app
        
        ingestion_client = TestClient(ingestion_app)
        chatbot_client = TestClient(chatbot_app)
        
        # Upload document
        with open(sample_pdf_path, "rb") as f:
            upload_response = ingestion_client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"metadata": '{"source": "test"}'}
            )
        
        assert upload_response.status_code == 200
        
        # Query the document
        chat_response = chatbot_client.post(
            "/chat/message",
            data={
                "query": "What is this document about?",
                "use_retrieval": True
            },
            headers={"soeid": "test-user"}
        )
        
        assert chat_response.status_code == 200
        data = chat_response.json()
        assert "response" in data
        assert len(data["response"]) > 0
```

## Performance Testing

### Benchmark Tests

```python
# tests/performance/test_embedding_performance.py
import pytest
import time
import asyncio
from src.models.embedding import EmbeddingModelFactory

class TestEmbeddingPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_embedding_throughput(self):
        """Test embedding model throughput."""
        model = EmbeddingModelFactory.create_model(
            provider="vertex_ai",
            model="text-embedding-004"
        )
        
        # Test data
        texts = [f"Sample text {i}" for i in range(100)]
        batch_sizes = [1, 10, 50, 100]
        
        results = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                await model.get_embeddings(batch)
            
            total_time = time.time() - start_time
            texts_per_second = len(texts) / total_time
            
            results[batch_size] = texts_per_second
            
        # Assert performance thresholds
        assert results[100] > results[1]  # Batch processing should be faster
        assert results[100] > 10  # Should process at least 10 texts/second
```

### Load Testing

```python
# tests/performance/test_load.py
import pytest
import asyncio
import aiohttp
import time

class TestLoadPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_chat_endpoint_load(self):
        """Load test the chat endpoint."""
        url = "http://localhost:8001/chat/message/json"
        headers = {"Content-Type": "application/json", "soeid": "load-test"}
        
        async def send_request(session, request_id):
            payload = {"query": f"Test query {request_id}", "use_retrieval": True}
            start_time = time.time()
            async with session.post(url, json=payload, headers=headers) as response:
                await response.json()
                return time.time() - start_time
        
        # Run concurrent requests
        concurrent_requests = 50
        async with aiohttp.ClientSession() as session:
            tasks = [send_request(session, i) for i in range(concurrent_requests)]
            response_times = await asyncio.gather(*tasks)
        
        # Analyze results
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Assert performance requirements
        assert avg_response_time < 5.0  # Average < 5 seconds
        assert max_response_time < 10.0  # Max < 10 seconds
        assert len([t for t in response_times if t < 3.0]) > len(response_times) * 0.8  # 80% under 3s
```

## Test Automation

### CI/CD Pipeline Configuration

**GitHub Actions:**
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        POSTGRES_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Commands and Scripts

**Makefile:**
```makefile
# Makefile
.PHONY: test test-unit test-integration test-e2e test-performance

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=html

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

test-performance:
	pytest tests/performance/ -v

test-coverage:
	pytest tests/unit/ tests/integration/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
```

## Best Practices

### Test Organization

1. **Clear test names**: Use descriptive test method names
2. **Arrange-Act-Assert**: Structure tests clearly
3. **One assertion per test**: Focus on single behaviors
4. **Use fixtures**: Share common setup code
5. **Mock external dependencies**: Isolate units under test

### Mocking Strategies

```python
# Good mocking example
@pytest.mark.asyncio
async def test_generation_with_mock(self):
    with patch('src.models.generation.httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test"}
        mock_post.return_value = mock_response
        
        model = VertexGenAI()
        result = await model.generate_content("test")
        
        assert result == "test"
        mock_post.assert_called_once()
```

### Performance Testing Guidelines

1. **Set realistic thresholds**: Based on production requirements
2. **Test different loads**: Various concurrency levels
3. **Monitor resources**: Memory, CPU, network usage
4. **Baseline comparisons**: Track performance over time
5. **Environment consistency**: Use similar test environments

## Related Documentation

- [Development Guide](./README.md) - Development setup and workflows
- [Configuration Guide](../configuration.md) - Configuration reference
- [API Documentation](../rag/chatbot/api.md) - API testing reference
- [Deployment Guide](../deployment/README.md) - Production testing

## Useful Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m performance

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run tests and stop on first failure
pytest -x

# Run tests matching pattern
pytest -k "test_generation"
```
