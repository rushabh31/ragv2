# Vector Stores

## 🎯 Overview

Vector stores are specialized databases that efficiently store and search high-dimensional vectors (embeddings). They enable fast similarity search, which is the core of RAG's retrieval mechanism. When a user asks a question, the vector store finds the most semantically similar document chunks.

## 🔍 How Vector Search Works

### **The Process**
1. **Query Embedding**: Convert user question to vector
2. **Similarity Search**: Find vectors closest to query vector
3. **Document Retrieval**: Return corresponding text chunks
4. **Ranking**: Order results by similarity score

```python
# Example similarity search
Query: "How does machine learning work?"
Query Vector: [0.1, -0.3, 0.8, ..., 0.2]

# Search finds similar document vectors:
Doc 1: [0.2, -0.2, 0.7, ..., 0.3] → Similarity: 0.92
Doc 2: [0.1, -0.4, 0.6, ..., 0.1] → Similarity: 0.87
Doc 3: [0.0, -0.1, 0.9, ..., 0.4] → Similarity: 0.85
```

### **Similarity Metrics**
- **Cosine Similarity**: Measures angle between vectors (most common)
- **Euclidean Distance**: Measures straight-line distance
- **Dot Product**: Measures vector alignment

## 🏗️ Available Vector Stores

### **1. FAISS (Facebook AI Similarity Search)**
High-performance vector search library optimized for large-scale similarity search.

**Features:**
- Extremely fast search performance
- Multiple index algorithms (HNSW, IVF, Flat)
- Memory and disk-based storage
- GPU acceleration support
- Scalable to billions of vectors

**Best for:**
- Production deployments
- Large document collections
- High-performance requirements
- Local/on-premise deployment

## 📋 FAISS Configuration

### **Basic Setup**
```yaml
ingestion:
  vector_store:
    provider: "faiss"
    faiss:
      index_type: "HNSW"              # Index algorithm
      dimension: 768                  # Embedding dimension
      metric: "cosine"                # Distance metric
      storage_path: "./vector_storage" # Storage directory
```

### **Advanced Configuration**
```yaml
ingestion:
  vector_store:
    provider: "faiss"
    faiss:
      index_type: "HNSW"
      dimension: 768
      metric: "cosine"
      storage_path: "./vector_storage"
      
      # HNSW-specific parameters
      ef_construction: 200            # Build-time search depth
      ef_search: 100                  # Query-time search depth
      max_connections: 16             # Graph connectivity (M parameter)
      
      # Performance settings
      num_threads: 4                  # Parallel search threads
      batch_size: 1000               # Batch insertion size
      
      # Storage options
      persist_on_disk: true          # Save index to disk
      memory_map: false              # Memory-map large indices
      compression: false             # Compress stored vectors
      
      # Metadata storage
      store_metadata: true           # Store document metadata
      metadata_index: "btree"        # Metadata index type
```

### **Index Types**
```yaml
# HNSW (Hierarchical Navigable Small World) - Recommended
faiss:
  index_type: "HNSW"
  ef_construction: 200              # Higher = better quality, slower build
  ef_search: 100                    # Higher = better recall, slower search
  max_connections: 16               # Higher = better connectivity, more memory

# IVF (Inverted File) - Good for very large datasets
faiss:
  index_type: "IVF"
  nlist: 1024                       # Number of clusters
  nprobe: 64                        # Number of clusters to search

# Flat - Exact search, good for small datasets
faiss:
  index_type: "Flat"
  # No additional parameters needed
```

## 🛠️ Vector Store Implementation

### **Using FAISS Vector Store**
```python
from src.rag.ingestion.vector_stores.faiss_store import FAISSVectorStore

# Initialize vector store
vector_store = FAISSVectorStore({
    "index_type": "HNSW",
    "dimension": 768,
    "metric": "cosine",
    "storage_path": "./vector_storage"
})

# Add documents with embeddings
documents = [
    Document(
        content="Machine learning is a subset of AI",
        embedding=[0.1, -0.3, 0.8, ...],  # 768-dimensional vector
        metadata={"source": "textbook", "page": 1}
    ),
    Document(
        content="Deep learning uses neural networks",
        embedding=[0.2, -0.2, 0.7, ...],  # 768-dimensional vector
        metadata={"source": "textbook", "page": 2}
    )
]

# Store documents
await vector_store.add_documents(documents)

# Search for similar documents
query_embedding = [0.15, -0.25, 0.75, ...]  # Query vector
results = await vector_store.similarity_search(
    query_embedding=query_embedding,
    top_k=5,
    score_threshold=0.7
)

print(f"Found {len(results)} similar documents")
for doc, score in results:
    print(f"Score: {score:.3f} - {doc.content[:100]}...")
```

### **Batch Operations**
```python
# Batch insertion for better performance
large_document_batch = [...]  # 1000+ documents

# Add in batches
batch_size = 100
for i in range(0, len(large_document_batch), batch_size):
    batch = large_document_batch[i:i + batch_size]
    await vector_store.add_documents(batch)
    print(f"Added batch {i//batch_size + 1}")

# Batch search
query_embeddings = [...]  # Multiple query vectors
all_results = await vector_store.batch_similarity_search(
    query_embeddings=query_embeddings,
    top_k=5
)
```

## 📊 Performance Characteristics

### **FAISS Performance**
- **Index Building**: 1000-10000 vectors/second
- **Search Speed**: 1-10ms per query (depending on dataset size)
- **Memory Usage**: ~4-8 bytes per dimension per vector
- **Disk Storage**: ~4 bytes per dimension per vector

### **Scalability**
```python
# Performance by dataset size (approximate)
Dataset Size    | Build Time | Search Time | Memory Usage
----------------|------------|-------------|-------------
1K documents    | 1 second   | <1ms        | 10MB
10K documents   | 10 seconds | 1-2ms       | 100MB
100K documents  | 2 minutes  | 2-5ms       | 1GB
1M documents    | 20 minutes | 5-10ms      | 10GB
10M documents   | 3 hours    | 10-20ms     | 100GB
```

### **Index Algorithm Comparison**
```python
Algorithm | Build Speed | Search Speed | Memory | Accuracy
----------|-------------|--------------|--------|----------
Flat      | Fast        | Slow*        | Low    | 100%
HNSW      | Medium      | Fast         | Medium | 95-99%
IVF       | Slow        | Fast         | Low    | 90-95%

* Slow for large datasets, fast for small datasets
```

## ⚡ Performance Optimization

### **HNSW Tuning**
```yaml
# High-performance setup (more memory, better quality)
faiss:
  index_type: "HNSW"
  ef_construction: 400              # Better build quality
  ef_search: 200                    # Better search quality
  max_connections: 32               # More connectivity

# Memory-optimized setup
faiss:
  index_type: "HNSW"
  ef_construction: 100              # Faster build
  ef_search: 50                     # Faster search
  max_connections: 8                # Less memory

# Balanced setup (recommended)
faiss:
  index_type: "HNSW"
  ef_construction: 200              # Good balance
  ef_search: 100                    # Good balance
  max_connections: 16               # Good balance
```

### **Search Optimization**
```python
# Optimize search parameters
search_params = {
    "top_k": 10,                    # Number of results
    "score_threshold": 0.7,         # Minimum similarity
    "ef_search": 100,               # Search depth (HNSW only)
    "nprobe": 64                    # Clusters to search (IVF only)
}

# Use appropriate top_k
# - Too small: May miss relevant documents
# - Too large: Slower search, more irrelevant results
# - Recommended: 5-20 for most use cases
```

### **Memory Management**
```yaml
# For large datasets, consider memory mapping
faiss:
  memory_map: true                  # Memory-map index files
  persist_on_disk: true            # Store on disk
  
# For memory-constrained environments
faiss:
  compression: true                 # Compress vectors (slight quality loss)
  index_type: "IVF"                # More memory-efficient than HNSW
```

## 🔍 Search Strategies

### **Basic Similarity Search**
```python
# Standard vector similarity search
results = await vector_store.similarity_search(
    query_embedding=query_vector,
    top_k=10,
    score_threshold=0.7
)
```

### **Filtered Search**
```python
# Search with metadata filters
results = await vector_store.similarity_search(
    query_embedding=query_vector,
    top_k=10,
    filter_metadata={
        "source": "manual",          # Only from manuals
        "category": "technical",     # Only technical content
        "date": {"$gte": "2024-01-01"}  # Only recent content
    }
)
```

### **Hybrid Search**
```python
# Combine vector similarity with keyword search
results = await vector_store.hybrid_search(
    query_embedding=query_vector,
    query_text="machine learning",
    top_k=10,
    vector_weight=0.7,              # 70% vector similarity
    keyword_weight=0.3              # 30% keyword matching
)
```

## 🚨 Common Issues and Solutions

### **Index Building Errors**
```python
# Error: "Dimension mismatch"
# Solution: Ensure all embeddings have same dimension
def validate_embeddings(documents):
    dimensions = [len(doc.embedding) for doc in documents]
    if len(set(dimensions)) > 1:
        raise ValueError(f"Inconsistent dimensions: {set(dimensions)}")
```

### **Search Performance Issues**
```yaml
# Issue: Slow search performance
# Solution: Tune search parameters
faiss:
  ef_search: 50                     # Reduce from 100
  top_k: 5                          # Reduce from 10

# Issue: Poor search quality
# Solution: Increase search parameters
faiss:
  ef_search: 200                    # Increase from 100
  ef_construction: 400              # Rebuild with higher quality
```

### **Memory Issues**
```yaml
# Issue: Out of memory during index building
# Solution: Use batch insertion
faiss:
  batch_size: 100                   # Reduce from 1000

# Issue: Index too large for memory
# Solution: Use memory mapping
faiss:
  memory_map: true
  persist_on_disk: true
```

### **Storage Issues**
```python
# Issue: Index files not persisting
# Solution: Ensure proper storage configuration
vector_store_config = {
    "storage_path": "./vector_storage",
    "persist_on_disk": True,
    "auto_save": True,
    "save_interval": 1000           # Save every 1000 additions
}
```

## 📈 Monitoring and Maintenance

### **Index Statistics**
```python
# Get index information
stats = await vector_store.get_stats()
print(f"Total vectors: {stats.total_vectors}")
print(f"Index size: {stats.index_size_mb:.1f} MB")
print(f"Build time: {stats.build_time:.1f} seconds")
print(f"Average search time: {stats.avg_search_time:.1f} ms")
```

### **Performance Monitoring**
```python
# Monitor search performance
import time

def monitor_search_performance(vector_store, test_queries):
    search_times = []
    
    for query in test_queries:
        start_time = time.time()
        results = await vector_store.similarity_search(query)
        search_time = (time.time() - start_time) * 1000
        search_times.append(search_time)
    
    avg_time = sum(search_times) / len(search_times)
    print(f"Average search time: {avg_time:.1f} ms")
    print(f"95th percentile: {sorted(search_times)[int(0.95 * len(search_times))]:.1f} ms")
```

### **Index Maintenance**
```python
# Rebuild index for better performance
await vector_store.rebuild_index()

# Optimize index (remove deleted documents, compress)
await vector_store.optimize_index()

# Backup index
await vector_store.backup_index("./backups/index_backup_20240115")

# Restore from backup
await vector_store.restore_index("./backups/index_backup_20240115")
```

## 🎯 Best Practices

### **Index Selection**
1. **Small datasets (<10K docs)**: Use Flat index for exact search
2. **Medium datasets (10K-1M docs)**: Use HNSW for balanced performance
3. **Large datasets (>1M docs)**: Consider IVF or distributed solutions

### **Parameter Tuning**
1. **Start with defaults**: Use recommended settings initially
2. **Measure performance**: Benchmark with your actual data
3. **Tune gradually**: Adjust one parameter at a time
4. **Monitor quality**: Ensure search quality doesn't degrade

### **Production Deployment**
1. **Persistent storage**: Always enable disk persistence
2. **Regular backups**: Backup indices regularly
3. **Monitor performance**: Track search times and quality
4. **Plan for growth**: Consider scalability requirements

### **Data Management**
1. **Consistent embeddings**: Ensure all vectors have same dimension
2. **Metadata indexing**: Index frequently filtered metadata fields
3. **Regular cleanup**: Remove outdated or duplicate documents
4. **Version control**: Track index versions and changes

## 🔧 Custom Vector Store Development

### **Creating a Custom Vector Store**
```python
from src.rag.ingestion.vector_stores.base_vector_store import BaseVectorStore

class CustomVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_param = config.get("custom_param", "default")
        self._initialize_store()
    
    def _initialize_store(self):
        # Initialize custom vector store
        pass
    
    async def add_documents(self, documents: List[Document]) -> None:
        # Implement document addition
        for doc in documents:
            await self._add_single_document(doc)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        # Implement similarity search
        pass
    
    async def _add_single_document(self, document: Document) -> None:
        # Custom document addition logic
        pass
```

### **Registering Custom Vector Store**
```python
# In vector store factory
from src.rag.ingestion.vector_stores.vector_store_factory import VectorStoreFactory

VectorStoreFactory.register_store("custom", CustomVectorStore)

# Use in configuration
ingestion:
  vector_store:
    provider: "custom"
    custom:
      custom_param: "value"
```

## 📚 Related Documentation

- **[Embedding Models](./embedders.md)** - Previous step: generate embeddings
- **[Ingestion API](./api.md)** - Use vector stores via API
- **[Chatbot Retrievers](../chatbot/retrievers.md)** - Search stored vectors
- **[Configuration Guide](../../configuration.md)** - Complete configuration reference

## 🚀 Quick Examples

### **Build and Search Index**
```python
# Quick index building and searching
from src.rag.ingestion.vector_stores import FAISSVectorStore

# Initialize
store = FAISSVectorStore({
    "index_type": "HNSW",
    "dimension": 768,
    "storage_path": "./test_index"
})

# Add documents
documents = [...]  # Your embedded documents
await store.add_documents(documents)

# Search
query_vector = [...]  # Your query embedding
results = await store.similarity_search(query_vector, top_k=5)

for doc, score in results:
    print(f"Score: {score:.3f} - {doc.content[:100]}...")
```

### **Performance Testing**
```bash
# API endpoint for performance testing
curl -X POST "http://localhost:8000/vector_store/benchmark" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "num_queries": 100,
    "top_k": 10,
    "measure_latency": true
  }'
```

---

**Next Steps**: 
- [Use the Ingestion API](./api.md)
- [Set up Document Retrieval](../chatbot/retrievers.md)
- [Configure the Chatbot Service](../chatbot/README.md)
