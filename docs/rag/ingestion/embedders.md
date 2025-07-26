# Embedding Models

## 🎯 Overview

Embedding models convert text chunks into numerical vectors (embeddings) that capture semantic meaning. These vectors enable similarity search, allowing the RAG system to find relevant documents based on meaning rather than just keyword matching.

## 🧮 What are Embeddings?

### **The Concept**
Embeddings are high-dimensional vectors (typically 768-1536 dimensions) that represent text meaning in mathematical space. Similar texts have similar vectors, enabling semantic search.

```python
# Example: Text to vector conversion
"machine learning" → [0.1, -0.3, 0.8, ..., 0.2]  # 768 numbers
"artificial intelligence" → [0.2, -0.2, 0.7, ..., 0.3]  # Similar vector
"cooking recipes" → [-0.5, 0.9, -0.1, ..., -0.8]  # Different vector
```

### **Why Embeddings Matter**
- **Semantic Search**: Find documents by meaning, not just keywords
- **Multilingual Support**: Similar concepts in different languages have similar vectors
- **Context Understanding**: Capture nuanced meaning and context
- **Efficient Retrieval**: Fast vector similarity search

## 🏭 Available Embedding Providers

### **1. Vertex AI Embeddings (Recommended)**
Google's latest text embedding models with high quality and performance.

**Features:**
- Latest text-embedding-004 model
- 768-dimensional vectors
- Multilingual support
- High accuracy and performance
- Enterprise-grade reliability

### **2. OpenAI Embeddings**
OpenAI's text embedding models with excellent semantic understanding.

**Features:**
- text-embedding-3-large model
- 1536-dimensional vectors
- Strong semantic understanding
- Good multilingual support
- Consistent performance

## 📋 Vertex AI Embedding Configuration

### **Basic Setup**
```yaml
ingestion:
  embedding:
    provider: "vertex"
    vertex:
      model: "text-embedding-004"      # Latest Vertex AI model
      batch_size: 100                  # Texts per API call
      region: "us-central1"            # GCP region
      dimension: 768                   # Output vector dimension
```

### **Advanced Configuration**
```yaml
ingestion:
  embedding:
    provider: "vertex"
    vertex:
      model: "text-embedding-004"
      batch_size: 100
      region: "us-central1"
      dimension: 768
      
      # Performance settings
      max_retries: 3                   # Retry failed requests
      timeout_seconds: 60              # Request timeout
      rate_limit_rpm: 1000             # Requests per minute
      
      # Processing options
      normalize_embeddings: true       # Normalize vector length
      truncate_input: true             # Truncate long texts
      max_input_length: 8192           # Maximum input tokens
      
      # Batch processing
      batch_timeout: 30                # Batch processing timeout
      concurrent_batches: 5            # Parallel batch processing
```

### **Performance Tuning**
```yaml
# High-throughput setup
vertex:
  batch_size: 200                     # Larger batches
  concurrent_batches: 10              # More parallelism
  rate_limit_rpm: 2000               # Higher rate limit

# Memory-optimized setup
vertex:
  batch_size: 50                      # Smaller batches
  concurrent_batches: 2               # Less parallelism
  rate_limit_rpm: 500                # Conservative rate limit

# Balanced setup (recommended)
vertex:
  batch_size: 100                     # Good balance
  concurrent_batches: 5               # Moderate parallelism
  rate_limit_rpm: 1000               # Standard rate limit
```

## 🔧 OpenAI Embedding Configuration

### **Basic Setup**
```yaml
ingestion:
  embedding:
    provider: "openai_universal"
    openai_universal:
      model: "text-embedding-3-large"  # Latest OpenAI model
      batch_size: 100                  # Texts per API call
      api_base: "https://api.openai.com/v1"
      dimension: 1536                  # Output vector dimension
```

### **Advanced Configuration**
```yaml
ingestion:
  embedding:
    provider: "openai_universal"
    openai_universal:
      model: "text-embedding-3-large"
      batch_size: 100
      api_base: "https://api.openai.com/v1"
      dimension: 1536
      
      # Authentication
      api_key_env: "OPENAI_API_KEY"    # Environment variable
      
      # Performance settings
      max_retries: 3
      timeout_seconds: 60
      rate_limit_rpm: 3000
      
      # Processing options
      encoding_format: "float"         # Vector format
      truncate_input: true
      max_input_length: 8191           # OpenAI token limit
```

## 🛠️ Embedding Implementation

### **Using Vertex AI Embedder**
```python
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder

# Initialize embedder
embedder = VertexEmbedder({
    "model": "text-embedding-004",
    "batch_size": 100,
    "region": "us-central1"
})

# Generate embeddings for chunks
chunks = [
    Document(content="Machine learning is a subset of AI"),
    Document(content="Deep learning uses neural networks"),
    Document(content="Natural language processing handles text")
]

embedded_chunks = await embedder.embed_documents(chunks)

print(f"Generated embeddings for {len(embedded_chunks)} chunks")
for chunk in embedded_chunks:
    print(f"Chunk: {chunk.content[:50]}...")
    print(f"Embedding dimension: {len(chunk.embedding)}")
    print(f"Embedding sample: {chunk.embedding[:5]}")
```

### **Using OpenAI Embedder**
```python
from src.rag.ingestion.embedders.openai_universal_embedder import OpenAIUniversalEmbedder

# Initialize embedder
embedder = OpenAIUniversalEmbedder({
    "model": "text-embedding-3-large",
    "batch_size": 100,
    "dimension": 1536
})

# Generate embeddings
embedded_chunks = await embedder.embed_documents(chunks)

print(f"OpenAI embeddings generated: {len(embedded_chunks)} chunks")
for chunk in embedded_chunks:
    print(f"Embedding dimension: {len(chunk.embedding)}")
```

## 📊 Embedding Quality and Performance

### **Vertex AI text-embedding-004**
- **Dimension**: 768
- **Max Input**: 8,192 tokens
- **Languages**: 100+ languages
- **Performance**: 95%+ accuracy on semantic similarity tasks
- **Speed**: 1000+ texts per minute
- **Cost**: Optimized for enterprise use

### **OpenAI text-embedding-3-large**
- **Dimension**: 1536 (configurable)
- **Max Input**: 8,191 tokens
- **Languages**: 100+ languages
- **Performance**: 96%+ accuracy on semantic similarity tasks
- **Speed**: 3000+ texts per minute
- **Cost**: Pay-per-token pricing

### **Performance Comparison**
```python
# Benchmark results (approximate)
Provider          | Dimension | Speed (texts/min) | Accuracy | Cost
------------------|-----------|-------------------|----------|------
Vertex AI         | 768       | 1000+            | 95%+     | Low
OpenAI            | 1536      | 3000+            | 96%+     | Medium
```

## 🚀 Batch Processing

### **How Batch Processing Works**
```python
# Example: Processing 1000 text chunks with batch_size=100
# Batch 1: chunks 1-100   → API call 1
# Batch 2: chunks 101-200 → API call 2
# ...
# Batch 10: chunks 901-1000 → API call 10
# Total: 10 API calls instead of 1000
```

### **Batch Size Guidelines**
- **Small batches (20-50)**: Lower memory, more API calls
- **Medium batches (50-100)**: Balanced performance (recommended)
- **Large batches (100-200)**: Higher memory, fewer API calls

### **Concurrent Batch Processing**
```yaml
# Process multiple batches simultaneously
embedding:
  vertex:
    batch_size: 100
    concurrent_batches: 5           # Process 5 batches in parallel
    # Total throughput: 500 texts per API round trip
```

## 📈 Performance Optimization

### **Memory Management**
```python
# Memory usage estimates
- Vertex AI (768-dim): ~3KB per embedding
- OpenAI (1536-dim): ~6KB per embedding
- 10,000 embeddings: 30-60MB memory
- Batch processing: Memory × batch_size
```

### **API Rate Limiting**
```yaml
# Respect provider rate limits
vertex:
  rate_limit_rpm: 1000              # Vertex AI limit
  batch_size: 100                   # Optimize for rate limit

openai_universal:
  rate_limit_rpm: 3000              # OpenAI limit
  batch_size: 100                   # Balance speed and limits
```

### **Error Handling and Retries**
```python
# Automatic retry logic
1. API call fails → Wait and retry (up to max_retries)
2. Rate limit hit → Exponential backoff
3. Batch fails → Split batch and retry smaller pieces
4. Individual text fails → Skip and log error
```

## 🚨 Common Issues and Solutions

### **Authentication Errors**
```bash
# Vertex AI authentication
python -c "
from src.models.embedding import VertexEmbeddingAI
model = VertexEmbeddingAI()
print('Auth status:', model.get_auth_health_status())
"

# OpenAI authentication
python -c "
import os
print('OpenAI API Key set:', bool(os.getenv('OPENAI_API_KEY')))
"
```

### **Memory Issues**
```yaml
# Reduce memory usage
embedding:
  vertex:
    batch_size: 50                  # Reduce from 100
    concurrent_batches: 2           # Reduce from 5
```

### **Rate Limit Errors**
```yaml
# Reduce API call rate
embedding:
  vertex:
    rate_limit_rpm: 500             # Reduce from 1000
    batch_size: 50                  # Smaller batches
```

### **Input Too Long Errors**
```yaml
# Handle long texts
embedding:
  vertex:
    truncate_input: true            # Truncate long texts
    max_input_length: 8000          # Conservative limit
```

## 🔍 Embedding Analysis

### **Vector Similarity**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compare embedding similarity
def compare_embeddings(text1, text2, embedder):
    # Get embeddings
    emb1 = await embedder.embed_single(text1)
    emb2 = await embedder.embed_single(text2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    return similarity

# Example usage
similarity = compare_embeddings(
    "machine learning algorithms",
    "artificial intelligence methods",
    embedder
)
print(f"Similarity: {similarity:.3f}")  # Expected: 0.7-0.9
```

### **Embedding Quality Metrics**
```python
# Evaluate embedding quality
def evaluate_embeddings(test_pairs, embedder):
    similarities = []
    
    for text1, text2, expected_sim in test_pairs:
        actual_sim = compare_embeddings(text1, text2, embedder)
        similarities.append((actual_sim, expected_sim))
    
    # Calculate correlation
    actual = [s[0] for s in similarities]
    expected = [s[1] for s in similarities]
    correlation = np.corrcoef(actual, expected)[0][1]
    
    return correlation

# Test with known similar/dissimilar pairs
test_pairs = [
    ("dog", "puppy", 0.8),           # High similarity
    ("car", "automobile", 0.9),      # High similarity
    ("apple", "computer", 0.1),      # Low similarity
    ("book", "reading", 0.6),        # Medium similarity
]

quality_score = evaluate_embeddings(test_pairs, embedder)
print(f"Embedding quality score: {quality_score:.3f}")
```

## 🎯 Best Practices

### **Model Selection**
1. **Vertex AI**: Best for enterprise, GCP integration, cost-effectiveness
2. **OpenAI**: Best for highest accuracy, fast processing, broad compatibility
3. **Consider**: Your existing cloud infrastructure and budget

### **Batch Size Optimization**
1. **Start with 100**: Good balance for most use cases
2. **Monitor Memory**: Increase if memory allows
3. **Watch Rate Limits**: Adjust based on provider limits
4. **Test Performance**: Measure actual throughput

### **Quality Assurance**
1. **Test Similarity**: Verify embeddings capture semantic meaning
2. **Monitor Failures**: Track and investigate embedding failures
3. **Validate Dimensions**: Ensure consistent vector dimensions
4. **Check Normalization**: Verify vector normalization if required

### **Production Deployment**
1. **Use Batch Processing**: Always use batching for efficiency
2. **Implement Retries**: Handle transient failures gracefully
3. **Monitor Costs**: Track API usage and costs
4. **Cache Embeddings**: Avoid recomputing identical texts

## 🔧 Custom Embedder Development

### **Creating a Custom Embedder**
```python
from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.shared.models.document import Document

class CustomEmbedder(BaseEmbedder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "custom-model")
        self.dimension = config.get("dimension", 768)
    
    async def embed_documents(self, documents: List[Document]) -> List[Document]:
        # Process documents in batches
        embedded_docs = []
        
        for batch in self._create_batches(documents):
            batch_embeddings = await self._embed_batch(batch)
            
            for doc, embedding in zip(batch, batch_embeddings):
                doc.embedding = embedding
                embedded_docs.append(doc)
        
        return embedded_docs
    
    async def _embed_batch(self, documents: List[Document]) -> List[List[float]]:
        # Implement custom embedding logic
        texts = [doc.content for doc in documents]
        embeddings = self._generate_embeddings(texts)
        return embeddings
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Custom embedding generation
        pass
```

### **Registering Custom Embedder**
```python
# In embedder factory
from src.rag.ingestion.embedders.embedder_factory import EmbedderFactory

EmbedderFactory.register_embedder("custom", CustomEmbedder)

# Use in configuration
ingestion:
  embedding:
    provider: "custom"
    custom:
      model: "custom-model-v1"
      dimension: 768
```

## 📚 Related Documentation

- **[Text Chunkers](./chunkers.md)** - Previous step: split documents into chunks
- **[Vector Stores](./vector-stores.md)** - Next step: store embeddings for search
- **[Model Providers](../../models/embedding.md)** - Detailed embedding model information
- **[Configuration Guide](../../configuration.md)** - Complete configuration reference

## 🚀 Quick Examples

### **Generate Embeddings for Text**
```python
# Quick embedding generation
from src.rag.ingestion.embedders import VertexEmbedder

embedder = VertexEmbedder({"model": "text-embedding-004"})

# Single text
embedding = await embedder.embed_single("Hello world")
print(f"Embedding dimension: {len(embedding)}")

# Multiple texts
texts = ["Hello", "World", "AI"]
embeddings = await embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### **Compare Text Similarity**
```bash
# API endpoint for similarity comparison
curl -X POST "http://localhost:8000/embed/similarity" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "machine learning",
    "text2": "artificial intelligence"
  }'
```

---

**Next Steps**: 
- [Configure Vector Storage](./vector-stores.md)
- [Use the Ingestion API](./api.md)
- [Set up the Chatbot Service](../chatbot/README.md)
