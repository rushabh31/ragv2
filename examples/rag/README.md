# 📚 RAG System Examples

This directory contains comprehensive examples demonstrating how to use the RAG system components and features.

## 🗂️ Directory Structure

```
examples/
├── rag/
│   ├── ingestion/          # Document processing examples
│   ├── chatbot/            # Chatbot implementation examples
│   └── configs/            # Configuration examples
├── advanced/               # Advanced usage patterns
├── use_cases/              # Real-world use case examples
└── README.md              # This file
```

## 🚀 Quick Start Examples

### 1. Document Ingestion

Process documents and create embeddings:

```bash
cd examples/rag/ingestion
python run_ingestion.py
```

**Features demonstrated:**
- Parallel vision parsing (2-5x speedup)
- Multi-provider support (Vertex AI, Groq, OpenAI)
- Configurable chunking strategies
- Vector store creation and management

### 2. Chatbot Interface

Interactive chatbot with memory:

```bash
cd examples/rag/chatbot
python run_chatbot.py
```

**Features demonstrated:**
- Multi-provider generation models
- LangGraph conversation memory
- Document retrieval and reranking
- Session management with SOEID tracking

### 3. Configuration Examples

Various provider configurations:

```bash
# Groq-based setup (cost-effective)
cp examples/configs/groq_config.yaml config.yaml

# Vertex AI setup (Google Cloud)
cp examples/configs/vertex_config.yaml config.yaml

# Mixed provider setup
cp examples/configs/mixed_providers_config.yaml config.yaml
```

## 📖 Detailed Examples

### RAG Components

#### Document Ingestion (`examples/rag/ingestion/`)

**Files:**
- `run_ingestion.py` - Main ingestion server
- `config.yaml` - Ingestion configuration
- `test_ingestion.py` - Ingestion testing

**Usage:**
```bash
# Start ingestion API
python run_ingestion.py

# Upload documents
curl -X POST "http://localhost:8000/ingest/upload" \
  -H "X-API-Key: test-api-key" \
  -F "file=@document.pdf"

# Check ingestion status
curl -X GET "http://localhost:8000/health"
```

**Key Features:**
- **Parallel Processing**: Configure `max_concurrent_pages` for optimal performance
- **Multiple Parsers**: Vision parsers for PDFs, text parsers for documents
- **Flexible Embedding**: Support for multiple embedding providers
- **Vector Storage**: FAISS-based vector store with persistence

#### Chatbot System (`examples/rag/chatbot/`)

**Files:**
- `run_chatbot.py` - Main chatbot server
- `config.yaml` - Chatbot configuration
- `api/service.py` - Chatbot service implementation
- `test_chatbot.py` - Chatbot testing

**Usage:**
```bash
# Start chatbot API
python run_chatbot.py

# Send chat message
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user123" \
  -d '{"query": "What is machine learning?", "use_retrieval": true}'

# Get chat history
curl -X GET "http://localhost:8001/chat/history/user123"
```

**Key Features:**
- **Multi-Provider Generation**: Groq, Vertex AI, OpenAI, Azure OpenAI
- **Advanced Memory**: LangGraph-based conversation memory
- **Document Retrieval**: Semantic search with reranking
- **Session Management**: SOEID-based user tracking

### Configuration Examples (`examples/configs/`)

#### Groq Configuration (Cost-Effective)
```yaml
# examples/configs/groq_config.yaml
generation:
  provider: groq
  config:
    model_name: meta-llama/llama-4-scout-17b-16e-instruct
    temperature: 0.1

embedding:
  provider: sentence_transformer
  config:
    model: all-mpnet-base-v2
    device: cpu

parser:
  provider: groq_vision_parser
  config:
    model_name: llama-3.2-11b-vision-preview
    max_concurrent_pages: 5
```

#### Vertex AI Configuration (Google Cloud)
```yaml
# examples/configs/vertex_config.yaml
generation:
  provider: vertex
  config:
    model_name: gemini-1.5-pro-002
    temperature: 0.1

embedding:
  provider: vertex_ai
  config:
    model: text-embedding-004
    project_id: "${PROJECT_ID}"

vision:
  provider: vertex_ai
  config:
    model: gemini-1.5-pro-002
    region: us-central1
```

#### Mixed Providers Configuration
```yaml
# examples/configs/mixed_providers_config.yaml
generation:
  provider: groq  # Fast and cost-effective
  
embedding:
  provider: vertex_ai  # High-quality embeddings
  
vision:
  provider: vertex_ai  # Advanced vision capabilities
```

## 🔧 Advanced Examples

### Custom Parser Implementation

```python
# examples/advanced/custom_parser.py
from src.rag.ingestion.parsers.base_parser import BaseParser

class CustomParser(BaseParser):
    async def _parse_file(self, file_path: str, config: dict):
        # Custom parsing logic
        pass
```

### Multi-Provider Setup

```python
# examples/advanced/multi_provider.py
from src.models.generation import GenerationModelFactory
from src.models.embedding import EmbeddingModelFactory

# Use different providers for different tasks
fast_generator = GenerationModelFactory.create_model("groq")
quality_embedder = EmbeddingModelFactory.create_model("vertex_ai")
```

### Performance Optimization

```python
# examples/advanced/performance_optimization.py
# Optimize parallel processing based on document size
config = {
    "max_concurrent_pages": 8 if doc_pages > 20 else 5,
    "batch_size": 50 if doc_count > 100 else 20
}
```

## 🧪 Testing Examples

All examples include comprehensive testing:

```bash
# Test ingestion system
cd examples/rag/ingestion
python test_ingestion.py

# Test chatbot system
cd examples/rag/chatbot
python test_chatbot.py

# Test specific configurations
python test_config.py --config groq_config.yaml
```

## 🎯 Use Case Examples

### Enterprise Document Processing

```python
# examples/use_cases/enterprise_processing.py
# High-volume document processing with parallel parsing
parser = GroqVisionParser({
    "max_concurrent_pages": 12,
    "batch_processing": True
})
```

### Customer Support Chatbot

```python
# examples/use_cases/customer_support.py
# Chatbot with conversation memory and document retrieval
chatbot = ChatbotService({
    "memory_type": "langgraph_checkpoint",
    "retrieval_enabled": True,
    "reranking_enabled": True
})
```

### Knowledge Base System

```python
# examples/use_cases/knowledge_base.py
# Comprehensive knowledge base with multi-modal search
kb = KnowledgeBase({
    "vision_parsing": True,
    "semantic_search": True,
    "multi_provider": True
})
```

## 🔍 Troubleshooting Examples

### Authentication Issues

```bash
# Test authentication for all providers
python examples/test_auth.py

# Check environment variables
python examples/check_env.py
```

### Performance Issues

```bash
# Benchmark parallel processing
python examples/benchmark_parallel.py

# Memory usage analysis
python examples/analyze_memory.py
```

### Configuration Issues

```bash
# Validate configuration
python examples/validate_config.py --config config.yaml

# Test provider connectivity
python examples/test_providers.py
```

## 📝 Example Output

### Successful Ingestion
```
✅ Document uploaded successfully
📄 Pages processed: 25
⚡ Processing time: 12.3s (with parallel processing)
🔢 Embeddings generated: 156 chunks
💾 Stored in vector database
```

### Chatbot Response
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {"document": "ml_guide.pdf", "page": 3, "score": 0.95},
    {"document": "ai_basics.pdf", "page": 7, "score": 0.87}
  ],
  "session_id": "session_123",
  "processing_time": 2.1
}
```

## 🚀 Next Steps

1. **Start with Basic Examples**: Try the ingestion and chatbot examples
2. **Explore Configurations**: Test different provider combinations
3. **Performance Tuning**: Adjust parallel processing settings
4. **Custom Implementation**: Build your own parsers and generators
5. **Production Deployment**: Use the enterprise examples for scaling

## 📞 Support

- **Documentation**: [Main README](../README.md)
- **API Reference**: [docs/api/](../docs/api/)
- **Troubleshooting**: [docs/troubleshooting.md](../docs/troubleshooting.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/controlsgenai/issues)

---

**Happy coding! 🎉**
