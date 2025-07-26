# Embedding Models

Embedding models convert text into high-dimensional vector representations that capture semantic meaning. These embeddings are essential for the RAG system's document retrieval, similarity search, and semantic understanding capabilities.

## Overview

The embedding model system provides:

- **Multi-provider support**: Vertex AI, OpenAI, Azure OpenAI, Sentence Transformers
- **Universal authentication**: Consistent auth across cloud providers
- **Factory pattern**: Easy model instantiation and switching
- **Batch processing**: Efficient processing of multiple texts
- **Dimension flexibility**: Support for different embedding dimensions
- **Local and cloud options**: Both API-based and local embedding models

## Supported Providers

### Vertex AI (Google Cloud)

**Provider**: `vertex_ai`  
**Models**: text-embedding-004, text-embedding-gecko, textembedding-gecko-multilingual  
**Dimensions**: 768 (text-embedding-004), 768 (gecko models)  
**Authentication**: Universal auth with OAuth2 tokens

```python
from src.models.embedding import VertexEmbeddingAI

# Create model instance
model = VertexEmbeddingAI(
    model="text-embedding-004",
    dimensions=768
)

# Get single embedding
text = "Machine learning is a subset of artificial intelligence"
embedding = await model.get_embedding(text)
print(f"Embedding dimension: {len(embedding)}")

# Get multiple embeddings
texts = [
    "What is machine learning?",
    "How does neural networks work?",
    "Explain deep learning concepts"
]
embeddings = await model.get_embeddings(texts)
print(f"Generated {len(embeddings)} embeddings")
```

**Configuration**:
```yaml
embedding:
  provider: "vertex_ai"
  config:
    model: "text-embedding-004"
    dimensions: 768
    batch_size: 100
    region: "us-central1"
```

### OpenAI Universal

**Provider**: `openai_universal`  
**Models**: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002  
**Dimensions**: 3072 (large), 1536 (small), 1536 (ada-002)  
**Authentication**: Universal auth with API key headers

```python
from src.models.embedding import OpenAIEmbeddingAI

# Create model instance
model = OpenAIEmbeddingAI(
    model="text-embedding-3-large",
    dimensions=3072
)

# Get embedding with custom dimensions
embedding = await model.get_embedding(
    text="Natural language processing",
    dimensions=1024  # Reduce dimensions for efficiency
)

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await model.get_embeddings(texts, batch_size=50)
```

**Configuration**:
```yaml
embedding:
  provider: "openai_universal"
  config:
    model: "text-embedding-3-large"
    dimensions: 3072
    batch_size: 50
    api_base: "https://api.openai.com/v1"
```

### Azure OpenAI

**Provider**: `azure_openai`  
**Models**: text-embedding-ada-002, text-embedding-3-large  
**Dimensions**: 1536 (ada-002), 3072 (3-large)  
**Authentication**: Universal auth with Azure AD tokens

```python
from src.models.embedding import AzureOpenAIEmbeddingAI

# Create model instance
model = AzureOpenAIEmbeddingAI(
    model="text-embedding-ada-002",
    dimensions=1536
)

# Get embeddings
texts = ["Azure OpenAI embedding", "Enterprise AI solutions"]
embeddings = await model.get_embeddings(texts)
```

**Configuration**:
```yaml
embedding:
  provider: "azure_openai"
  config:
    model: "text-embedding-ada-002"
    dimensions: 1536
    batch_size: 100
    api_version: "2024-02-15-preview"
    azure_endpoint: "https://your-resource.openai.azure.com/"
```

### Sentence Transformers (Local)

**Provider**: `sentence_transformer`  
**Models**: all-mpnet-base-v2, all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1  
**Dimensions**: 768 (mpnet), 384 (MiniLM), 768 (multi-qa)  
**Authentication**: None (local model)

```python
from src.models.embedding import SentenceTransformerEmbedding

# Create model instance (downloads model on first use)
model = SentenceTransformerEmbedding(
    model_name="all-mpnet-base-v2",
    dimensions=768
)

# Get embeddings (runs locally)
texts = ["Local embedding processing", "No API calls required"]
embeddings = await model.get_embeddings(texts)

# Batch processing with custom batch size
large_texts = ["Text " + str(i) for i in range(1000)]
embeddings = await model.get_embeddings(large_texts, batch_size=32)
```

**Configuration**:
```yaml
embedding:
  provider: "sentence_transformer"
  config:
    model_name: "all-mpnet-base-v2"
    dimensions: 768
    batch_size: 32
    device: "cpu"  # or "cuda" for GPU
```

## Factory Usage

### Creating Models

Use the `EmbeddingModelFactory` to create model instances:

```python
from src.models.embedding import EmbeddingModelFactory

# Create model using factory
model = EmbeddingModelFactory.create_model(
    provider="vertex_ai",
    model="text-embedding-004",
    dimensions=768
)

# List available providers
providers = EmbeddingModelFactory.list_providers()
print(f"Available providers: {providers}")

# Check if provider is supported
is_supported = EmbeddingModelFactory.is_provider_supported("vertex_ai")
```

### Configuration-Based Creation

Create models from configuration files:

```python
import yaml
from src.models.embedding import EmbeddingModelFactory

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create model from config
embedding_config = config["embedding"]
model = EmbeddingModelFactory.create_model(
    provider=embedding_config["provider"],
    **embedding_config["config"]
)
```

## API Reference

### Common Methods

All embedding models implement these methods:

#### `get_embedding(text: str) -> List[float]`

Get embedding for a single text:

```python
text = "What is artificial intelligence?"
embedding = await model.get_embedding(text)
print(f"Embedding: {embedding[:5]}...")  # Show first 5 dimensions
```

#### `get_embeddings(texts: List[str], batch_size: int = None) -> List[List[float]]`

Get embeddings for multiple texts:

```python
texts = [
    "Machine learning algorithms",
    "Deep neural networks",
    "Natural language processing"
]
embeddings = await model.get_embeddings(texts, batch_size=10)
print(f"Generated {len(embeddings)} embeddings")
```

#### `get_coin_token() -> str` (Cloud providers only)

Get authentication token:

```python
# Only for cloud providers (Vertex AI, OpenAI, Azure OpenAI)
token = await model.get_coin_token()
```

#### `validate_authentication() -> bool` (Cloud providers only)

Validate authentication status:

```python
is_valid = await model.validate_authentication()
```

#### `get_auth_health_status() -> Dict` (Cloud providers only)

Get authentication health status:

```python
health = await model.get_auth_health_status()
print(f"Auth status: {health['status']}")
```

### Advanced Methods

#### Similarity Calculation

```python
import numpy as np
from typing import List

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    return dot_product / (norm1 * norm2)

# Example usage
text1 = "Machine learning is powerful"
text2 = "AI algorithms are effective"

embedding1 = await model.get_embedding(text1)
embedding2 = await model.get_embedding(text2)

similarity = cosine_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")
```

#### Batch Processing with Progress

```python
import asyncio
from tqdm import tqdm

async def process_large_dataset(model, texts: List[str], batch_size: int = 100):
    """Process large datasets with progress tracking."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = await model.get_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
        
        # Optional: Add delay to respect rate limits
        await asyncio.sleep(0.1)
    
    return all_embeddings
```

## RAG Integration

### Embedder Components

Embedding models are used in RAG embedder components:

```python
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder

# Create RAG embedder
embedder = VertexEmbedder({
    "model": "text-embedding-004",
    "dimensions": 768,
    "batch_size": 100
})

# Embed documents
documents = [
    {"content": "Document 1 content", "metadata": {"id": "doc1"}},
    {"content": "Document 2 content", "metadata": {"id": "doc2"}}
]

embedded_docs = await embedder.embed_documents(documents)
```

### Vector Store Integration

Embeddings are stored in vector databases:

```python
from src.rag.ingestion.vector_stores.faiss_store import FAISSVectorStore

# Create vector store
vector_store = FAISSVectorStore({
    "dimension": 768,
    "index_type": "HNSW",
    "metric": "cosine"
})

# Add embeddings to vector store
for doc, embedding in zip(documents, embeddings):
    await vector_store.add_document(
        document_id=doc["metadata"]["id"],
        embedding=embedding,
        metadata=doc["metadata"]
    )

# Search similar documents
query_embedding = await model.get_embedding("search query")
similar_docs = await vector_store.search(
    query_embedding=query_embedding,
    top_k=5,
    score_threshold=0.7
)
```

### Retrieval Integration

Embeddings enable semantic retrieval:

```python
from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever

# Create retriever
retriever = VectorRetriever({
    "embedder": {
        "provider": "vertex_ai",
        "config": {
            "model": "text-embedding-004",
            "dimensions": 768
        }
    },
    "vector_store": {
        "type": "faiss",
        "config": {"dimension": 768}
    }
})

# Retrieve relevant documents
query = "What is machine learning?"
relevant_docs = await retriever.retrieve(
    query=query,
    top_k=5,
    score_threshold=0.75
)
```

## Performance Optimization

### Batch Processing

Optimize batch sizes for different providers:

```python
# Provider-specific optimal batch sizes
OPTIMAL_BATCH_SIZES = {
    "vertex_ai": 100,
    "openai_universal": 50,
    "azure_openai": 100,
    "sentence_transformer": 32
}

async def optimized_embedding(model, texts: List[str], provider: str):
    batch_size = OPTIMAL_BATCH_SIZES.get(provider, 50)
    return await model.get_embeddings(texts, batch_size=batch_size)
```

### Caching

Implement embedding caching to avoid recomputation:

```python
import hashlib
import json
from typing import Dict, List, Optional

class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._get_cache_key(text, model_name)
        return self.cache.get(key)
    
    def set(self, text: str, model_name: str, embedding: List[float]):
        """Cache embedding."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_cache_key(text, model_name)
        self.cache[key] = embedding

# Usage with caching
cache = EmbeddingCache()

async def get_embedding_with_cache(model, text: str, model_name: str):
    # Check cache first
    cached_embedding = cache.get(text, model_name)
    if cached_embedding:
        return cached_embedding
    
    # Generate new embedding
    embedding = await model.get_embedding(text)
    
    # Cache result
    cache.set(text, model_name, embedding)
    
    return embedding
```

### Parallel Processing

Process multiple texts in parallel:

```python
import asyncio
from typing import List

async def parallel_embedding(model, texts: List[str], max_concurrent: int = 10):
    """Process embeddings with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_text(text: str):
        async with semaphore:
            return await model.get_embedding(text)
    
    tasks = [process_text(text) for text in texts]
    embeddings = await asyncio.gather(*tasks)
    
    return embeddings
```

### Memory Optimization

Optimize memory usage for large datasets:

```python
import gc
from typing import Iterator, List

def chunk_texts(texts: List[str], chunk_size: int) -> Iterator[List[str]]:
    """Yield chunks of texts to process."""
    for i in range(0, len(texts), chunk_size):
        yield texts[i:i + chunk_size]

async def memory_efficient_embedding(model, texts: List[str], chunk_size: int = 1000):
    """Process large datasets with memory optimization."""
    all_embeddings = []
    
    for chunk in chunk_texts(texts, chunk_size):
        chunk_embeddings = await model.get_embeddings(chunk)
        all_embeddings.extend(chunk_embeddings)
        
        # Force garbage collection
        gc.collect()
    
    return all_embeddings
```

## Error Handling

### Authentication Errors

```python
from src.models.exceptions import AuthenticationError

try:
    embeddings = await model.get_embeddings(texts)
except AuthenticationError as e:
    logger.error(f"Authentication failed: {e}")
    # Handle authentication failure
    # - Check credentials
    # - Retry with backoff
    # - Switch to local model
```

### API Errors

```python
from src.models.exceptions import APIError, RateLimitError

try:
    embeddings = await model.get_embeddings(texts)
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    # Wait and retry
    await asyncio.sleep(e.retry_after or 60)
    embeddings = await model.get_embeddings(texts)
except APIError as e:
    logger.error(f"API error: {e}")
    # Handle API failure
    # - Check API status
    # - Switch to fallback provider
    # - Use cached embeddings if available
```

### Dimension Mismatch

```python
def validate_embedding_dimensions(embeddings: List[List[float]], expected_dim: int):
    """Validate embedding dimensions."""
    for i, embedding in enumerate(embeddings):
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding {i} has dimension {len(embedding)}, "
                f"expected {expected_dim}"
            )

# Usage
try:
    embeddings = await model.get_embeddings(texts)
    validate_embedding_dimensions(embeddings, 768)
except ValueError as e:
    logger.error(f"Dimension validation failed: {e}")
```

### Fallback Strategies

```python
async def embedding_with_fallback(texts: List[str], providers: List[str]):
    """Try multiple providers with fallback."""
    for provider in providers:
        try:
            model = EmbeddingModelFactory.create_model(provider, "default-model")
            embeddings = await model.get_embeddings(texts)
            logger.info(f"Successfully generated embeddings using {provider}")
            return embeddings
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue
    
    # All providers failed
    raise Exception("All embedding providers failed")
```

## Monitoring and Analytics

### Performance Tracking

```python
import time
from typing import Dict, Any

class EmbeddingMetrics:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_texts": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0,
            "average_latency": 0,
            "texts_per_second": 0
        }
    
    async def track_embedding(self, model, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        self.metrics["total_requests"] += 1
        self.metrics["total_texts"] += len(texts)
        
        try:
            embeddings = await model.get_embeddings(texts)
            latency = time.time() - start_time
            
            self.metrics["successful_requests"] += 1
            self.metrics["total_latency"] += latency
            self.metrics["average_latency"] = (
                self.metrics["total_latency"] / self.metrics["successful_requests"]
            )
            self.metrics["texts_per_second"] = len(texts) / latency
            
            logger.info(f"Embedded {len(texts)} texts in {latency:.2f}s")
            return embeddings
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Embedding failed: {e}")
            raise
```

### Usage Analytics

```python
from collections import defaultdict
import json

class EmbeddingUsageAnalytics:
    def __init__(self):
        self.usage_stats = defaultdict(lambda: {
            "requests": 0,
            "texts_processed": 0,
            "total_dimensions": 0,
            "average_batch_size": 0,
            "error_rate": 0
        })
    
    def record_usage(self, provider: str, texts_count: int, dimensions: int, success: bool):
        stats = self.usage_stats[provider]
        stats["requests"] += 1
        
        if success:
            stats["texts_processed"] += texts_count
            stats["total_dimensions"] += dimensions * texts_count
            # Update average batch size
            stats["average_batch_size"] = stats["texts_processed"] / stats["requests"]
        else:
            # Update error rate
            error_count = stats["requests"] * stats["error_rate"] + 1
            stats["error_rate"] = error_count / stats["requests"]
    
    def get_usage_report(self) -> str:
        return json.dumps(dict(self.usage_stats), indent=2)
```

### Health Monitoring

```python
async def check_embedding_health():
    """Check health of all embedding providers."""
    providers = ["vertex_ai", "openai_universal", "azure_openai", "sentence_transformer"]
    health_status = {}
    
    for provider in providers:
        try:
            model = EmbeddingModelFactory.create_model(provider, "default-model")
            
            # Check authentication (cloud providers only)
            if hasattr(model, 'get_auth_health_status'):
                auth_status = await model.get_auth_health_status()
            else:
                auth_status = {"status": "not_applicable"}
            
            # Test embedding generation
            start_time = time.time()
            test_embedding = await model.get_embedding("Health check test")
            latency = time.time() - start_time
            
            health_status[provider] = {
                "status": "healthy",
                "auth_status": auth_status,
                "latency": latency,
                "embedding_dimension": len(test_embedding)
            }
            
        except Exception as e:
            health_status[provider] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status
```

## Best Practices

### Model Selection

1. **Vertex AI**: Best for Google Cloud environments, high quality embeddings
2. **OpenAI**: High-dimensional embeddings, good for complex tasks
3. **Azure OpenAI**: Enterprise Azure integration
4. **Sentence Transformers**: Local processing, no API costs, good for privacy

### Dimension Selection

1. **768 dimensions**: Good balance of quality and performance
2. **1536+ dimensions**: Better for complex semantic tasks
3. **384 dimensions**: Faster processing, lower storage requirements
4. **3072 dimensions**: Highest quality for critical applications

### Batch Size Optimization

```python
# Recommended batch sizes by provider
BATCH_SIZE_RECOMMENDATIONS = {
    "vertex_ai": {
        "small_texts": 100,    # < 100 characters
        "medium_texts": 50,    # 100-500 characters
        "large_texts": 20      # > 500 characters
    },
    "openai_universal": {
        "small_texts": 50,
        "medium_texts": 25,
        "large_texts": 10
    },
    "sentence_transformer": {
        "small_texts": 64,
        "medium_texts": 32,
        "large_texts": 16
    }
}
```

### Text Preprocessing

```python
import re

def preprocess_text_for_embedding(text: str) -> str:
    """Preprocess text for better embedding quality."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (optional)
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    # Truncate if too long (model-specific limits)
    max_length = 8000  # Adjust based on model
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()
```

### Caching Strategy

```python
# Cache configuration
CACHE_CONFIG = {
    "enable_caching": True,
    "cache_size": 10000,
    "cache_ttl": 86400,  # 24 hours
    "cache_key_include_model": True
}
```

## Troubleshooting

### Common Issues

**Authentication Failures**:
```bash
# Check environment variables
echo $COIN_CONSUMER_CLIENT_ID
echo $PROJECT_ID

# Test authentication
python -c "
from src.models.embedding import VertexEmbeddingAI
import asyncio
async def test():
    model = VertexEmbeddingAI()
    token = await model.get_coin_token()
    print(f'Token: {token[:50]}...')
asyncio.run(test())
"
```

**Dimension Mismatches**:
```python
# Check model dimensions
model_info = {
    "text-embedding-004": 768,
    "text-embedding-3-large": 3072,
    "all-mpnet-base-v2": 768
}

# Validate configuration
def validate_embedding_config(config):
    provider = config["provider"]
    model_name = config["config"]["model"]
    expected_dim = config["config"]["dimensions"]
    
    if model_name in model_info:
        actual_dim = model_info[model_name]
        if actual_dim != expected_dim:
            print(f"Warning: {model_name} produces {actual_dim}D embeddings, "
                  f"but config specifies {expected_dim}D")
```

**Performance Issues**:
```python
# Monitor embedding performance
import time

async def debug_embedding_performance(model, texts):
    print(f"Processing {len(texts)} texts...")
    
    start_time = time.time()
    embeddings = await model.get_embeddings(texts)
    total_time = time.time() - start_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Texts per second: {len(texts) / total_time:.2f}")
    print(f"Average time per text: {total_time / len(texts):.4f}s")
    print(f"Embedding dimensions: {len(embeddings[0])}")
```

**Memory Issues**:
```python
import psutil

def check_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    print(f"Available memory: {memory.available / 1024**3:.2f} GB")
    
    if memory.percent > 80:
        print("Warning: High memory usage detected")
        print("Consider reducing batch size or using chunked processing")
```

## Related Documentation

- [Model Factories Overview](./README.md) - Factory system overview
- [Generation Models](./generation.md) - Generation model documentation
- [Vision Models](./vision.md) - Vision model capabilities
- [Vector Stores](../rag/ingestion/vector-stores.md) - Vector storage systems
- [Retrievers](../rag/chatbot/retrievers.md) - Document retrieval components
- [Configuration Guide](../configuration.md) - Configuration reference

## Examples

For complete examples and test scripts, see:

- `examples/test_embedding_models.py` - Embedding model testing
- `examples/test_factory_system.py` - Factory system testing
- `examples/rag/ingestion/` - Ingestion pipeline examples
- `src/rag/ingestion/embedders/` - Embedder implementations
