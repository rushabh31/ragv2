# Document Retrievers

## 🎯 Overview

Document retrievers are responsible for finding relevant documents from the vector store based on user queries. They convert user questions into embeddings and perform similarity search to identify the most relevant content. The quality of retrieval directly impacts the accuracy of AI responses.

## 🔍 How Retrieval Works

### **The Process**
1. **Query Embedding**: Convert user question to vector using the same embedding model used for documents
2. **Similarity Search**: Find document chunks with vectors most similar to the query vector
3. **Filtering**: Apply metadata filters and score thresholds
4. **Ranking**: Order results by similarity score
5. **Return**: Provide top-k most relevant documents

```python
# Example retrieval process
User Query: "How does machine learning work?"
Query Vector: [0.1, -0.3, 0.8, ..., 0.2]

# Vector store search finds:
Doc 1: "Machine learning algorithms..." → Similarity: 0.92
Doc 2: "Deep learning is a subset..." → Similarity: 0.87  
Doc 3: "AI models learn from data..." → Similarity: 0.85
Doc 4: "Neural networks process..." → Similarity: 0.82
Doc 5: "Training data is essential..." → Similarity: 0.79
```

## 🔧 Available Retrievers

### **1. Vector Retriever (Primary)**
Pure semantic similarity search using vector embeddings.

**Best for:**
- Semantic understanding
- Conceptual queries
- Cross-language search
- Finding related concepts

**Features:**
- Fast similarity search
- Configurable result count
- Score thresholds
- Metadata filtering

### **2. Hybrid Retriever (Future)**
Combines vector similarity with keyword matching for improved accuracy.

**Best for:**
- Exact term matching
- Technical documentation
- Mixed semantic/keyword queries
- Comprehensive search coverage

## 📋 Vector Retriever Configuration

### **Basic Setup**
```yaml
chatbot:
  retrieval:
    provider: "vector"
    vector:
      top_k: 10                        # Number of documents to retrieve
      score_threshold: 0.7             # Minimum similarity score (0.0-1.0)
      vector_store_path: "../ingestion/vector_storage"  # Path to vector store
      embedding_model: "text-embedding-004"  # Must match ingestion model
```

### **Advanced Configuration**
```yaml
chatbot:
  retrieval:
    provider: "vector"
    vector:
      top_k: 10
      score_threshold: 0.7
      vector_store_path: "../ingestion/vector_storage"
      embedding_model: "text-embedding-004"
      
      # Search parameters
      search_type: "similarity"        # Search algorithm
      distance_metric: "cosine"        # Distance calculation
      ef_search: 100                   # FAISS search parameter
      
      # Filtering options
      filter_metadata: {}              # Metadata filters
      exclude_metadata: {}             # Metadata exclusions
      
      # Performance settings
      batch_search: false              # Batch multiple queries
      cache_embeddings: true           # Cache query embeddings
      timeout_seconds: 30              # Search timeout
      
      # Quality settings
      diversity_threshold: 0.8         # Avoid duplicate results
      min_content_length: 50           # Minimum chunk size
      max_content_length: 2000         # Maximum chunk size
```

### **Performance Tuning**
```yaml
# High-recall setup (find more documents)
vector:
  top_k: 20                          # More documents
  score_threshold: 0.6               # Lower threshold
  ef_search: 200                     # More thorough search

# High-precision setup (find best documents)
vector:
  top_k: 5                           # Fewer documents
  score_threshold: 0.8               # Higher threshold
  diversity_threshold: 0.9           # More diverse results

# Balanced setup (recommended)
vector:
  top_k: 10                          # Good balance
  score_threshold: 0.7               # Reasonable threshold
  ef_search: 100                     # Standard search depth
```

## 🛠️ Retriever Implementation

### **Using Vector Retriever**
```python
from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever

# Initialize retriever
retriever = VectorRetriever({
    "top_k": 10,
    "score_threshold": 0.7,
    "vector_store_path": "./vector_storage",
    "embedding_model": "text-embedding-004"
})

# Retrieve documents for a query
query = "What is machine learning?"
documents = await retriever.retrieve(
    query=query,
    filters={"category": "technical"},
    top_k=10
)

print(f"Retrieved {len(documents)} documents")
for doc in documents:
    print(f"Score: {doc.score:.3f} - {doc.content[:100]}...")
    print(f"Source: {doc.metadata.get('filename', 'unknown')}")
    print(f"Page: {doc.metadata.get('page_number', 'N/A')}")
```

### **Advanced Retrieval**
```python
# Retrieve with complex filters
documents = await retriever.retrieve(
    query="machine learning algorithms",
    filters={
        "source": {"$in": ["textbook", "research_paper"]},
        "date": {"$gte": "2023-01-01"},
        "category": "technical",
        "confidentiality": {"$ne": "restricted"}
    },
    top_k=15,
    score_threshold=0.75
)

# Retrieve with exclusions
documents = await retriever.retrieve(
    query="deep learning",
    exclude_filters={
        "category": "marketing",
        "type": "advertisement"
    },
    top_k=10
)
```

## 📊 Retrieval Quality Metrics

### **Similarity Scores**
- **0.9-1.0**: Highly relevant, near-exact semantic match
- **0.8-0.9**: Very relevant, strong semantic similarity
- **0.7-0.8**: Relevant, good semantic match
- **0.6-0.7**: Moderately relevant, some semantic similarity
- **0.5-0.6**: Weakly relevant, limited semantic match
- **<0.5**: Likely irrelevant

### **Performance Characteristics**
```python
# Typical performance metrics
Metric                | Small DB | Medium DB | Large DB
                     | (<10K)   | (10K-100K)| (>100K)
---------------------|----------|-----------|----------
Search Time          | <10ms    | 10-50ms   | 50-200ms
Memory Usage         | 50MB     | 500MB     | 5GB
Accuracy (top-5)     | 95%      | 90%       | 85%
Accuracy (top-10)    | 98%      | 95%       | 90%
```

### **Quality Assessment**
```python
# Evaluate retrieval quality
def evaluate_retrieval_quality(test_queries, retriever):
    results = []
    
    for query_data in test_queries:
        query = query_data["query"]
        expected_docs = query_data["expected_documents"]
        
        # Retrieve documents
        retrieved = await retriever.retrieve(query, top_k=10)
        retrieved_ids = [doc.metadata.get("document_id") for doc in retrieved]
        
        # Calculate metrics
        relevant_found = len(set(retrieved_ids) & set(expected_docs))
        precision_at_5 = relevant_found / min(5, len(retrieved))
        recall = relevant_found / len(expected_docs)
        
        results.append({
            "query": query,
            "precision@5": precision_at_5,
            "recall": recall,
            "avg_score": sum(doc.score for doc in retrieved[:5]) / 5
        })
    
    return results
```

## 🔍 Metadata Filtering

### **Basic Filters**
```python
# Simple equality filters
filters = {
    "source": "manual",              # Exact match
    "category": "technical",         # Exact match
    "language": "english"            # Exact match
}

# Multiple values (OR logic)
filters = {
    "source": {"$in": ["manual", "guide", "documentation"]},
    "category": {"$in": ["technical", "tutorial"]}
}
```

### **Advanced Filters**
```python
# Comparison operators
filters = {
    "date": {"$gte": "2023-01-01"},          # Greater than or equal
    "page_count": {"$lt": 100},              # Less than
    "confidence": {"$gte": 0.8},             # Minimum confidence
    "word_count": {"$range": [100, 2000]}    # Range filter
}

# Negation filters
filters = {
    "category": {"$ne": "marketing"},        # Not equal
    "status": {"$nin": ["draft", "archived"]} # Not in list
}

# Text filters
filters = {
    "title": {"$contains": "machine learning"}, # Contains text
    "author": {"$startswith": "Dr."},           # Starts with
    "filename": {"$endswith": ".pdf"}           # Ends with
}
```

### **Dynamic Filtering**
```python
# User-based filtering
def get_user_filters(user_role, user_department):
    base_filters = {}
    
    # Role-based access
    if user_role == "employee":
        base_filters["confidentiality"] = {"$in": ["public", "internal"]}
    elif user_role == "contractor":
        base_filters["confidentiality"] = "public"
    
    # Department-based filtering
    if user_department:
        base_filters["relevant_departments"] = {"$contains": user_department}
    
    return base_filters

# Time-based filtering
def get_temporal_filters(time_preference="recent"):
    if time_preference == "recent":
        return {"date": {"$gte": "2023-01-01"}}
    elif time_preference == "historical":
        return {"date": {"$lt": "2020-01-01"}}
    else:
        return {}
```

## ⚡ Performance Optimization

### **Search Parameter Tuning**
```yaml
# For speed (sacrifice some accuracy)
vector:
  top_k: 5                           # Fewer results
  ef_search: 50                      # Faster search
  score_threshold: 0.75              # Higher threshold

# For accuracy (sacrifice some speed)
vector:
  top_k: 20                          # More results
  ef_search: 200                     # Thorough search
  score_threshold: 0.6               # Lower threshold
```

### **Caching Strategies**
```python
# Query embedding caching
class CachedRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.embedding_cache = {}
        self.result_cache = {}
    
    async def retrieve(self, query, **kwargs):
        # Check result cache first
        cache_key = self._get_cache_key(query, kwargs)
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # Check embedding cache
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            query_embedding = await self._embed_query(query)
            self.embedding_cache[query] = query_embedding
        
        # Perform search
        results = await self.base_retriever.search_by_embedding(
            query_embedding, **kwargs
        )
        
        # Cache results
        self.result_cache[cache_key] = results
        return results
```

### **Batch Retrieval**
```python
# Process multiple queries efficiently
async def batch_retrieve(queries, retriever):
    # Embed all queries in batch
    query_embeddings = await retriever.embed_batch(queries)
    
    # Perform batch search
    all_results = await retriever.batch_search(
        query_embeddings,
        top_k=10,
        score_threshold=0.7
    )
    
    return dict(zip(queries, all_results))
```

## 🚨 Common Issues and Solutions

### **Poor Retrieval Quality**
```yaml
# Issue: Irrelevant documents returned
# Solution: Increase score threshold
retrieval:
  vector:
    score_threshold: 0.8             # Increase from 0.7
    top_k: 5                         # Reduce to get best results

# Issue: Missing relevant documents  
# Solution: Lower threshold, increase top_k
retrieval:
  vector:
    score_threshold: 0.6             # Decrease from 0.7
    top_k: 15                        # Increase from 10
```

### **Slow Retrieval Performance**
```yaml
# Issue: Search taking too long
# Solution: Optimize search parameters
retrieval:
  vector:
    ef_search: 50                    # Reduce from 100
    top_k: 5                         # Reduce from 10
    cache_embeddings: true           # Enable caching
```

### **Embedding Model Mismatch**
```python
# Issue: "Dimension mismatch" or poor results
# Solution: Ensure same embedding model as ingestion
def validate_embedding_compatibility(retriever, vector_store):
    retriever_dim = retriever.embedding_model.dimension
    store_dim = vector_store.dimension
    
    if retriever_dim != store_dim:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"retriever={retriever_dim}, store={store_dim}"
        )
```

### **Memory Issues**
```yaml
# Issue: High memory usage
# Solution: Reduce vector store memory footprint
retrieval:
  vector:
    memory_map: true                 # Memory-map vector store
    cache_embeddings: false          # Disable embedding cache
```

## 🎯 Best Practices

### **Query Preprocessing**
```python
def preprocess_query(query):
    # Clean and normalize query
    query = query.strip().lower()
    
    # Expand abbreviations
    abbreviations = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing"
    }
    
    for abbr, full in abbreviations.items():
        query = query.replace(abbr, full)
    
    # Remove stop words for better matching
    stop_words = {"the", "a", "an", "and", "or", "but"}
    words = query.split()
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)
```

### **Result Validation**
```python
def validate_retrieval_results(results, query):
    validated_results = []
    
    for doc in results:
        # Check minimum content length
        if len(doc.content) < 50:
            continue
        
        # Check for query relevance
        if not has_query_overlap(doc.content, query):
            continue
        
        # Check metadata completeness
        if not doc.metadata.get("source"):
            continue
        
        validated_results.append(doc)
    
    return validated_results

def has_query_overlap(content, query):
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    overlap = len(query_words & content_words)
    return overlap >= min(2, len(query_words) * 0.3)
```

### **Adaptive Retrieval**
```python
def adaptive_retrieve(query, retriever, min_results=3):
    # Start with high threshold
    threshold = 0.8
    top_k = 5
    
    while threshold > 0.5:
        results = await retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=threshold
        )
        
        if len(results) >= min_results:
            return results
        
        # Relax constraints
        threshold -= 0.1
        top_k = min(top_k + 5, 20)
    
    # Final attempt with minimal constraints
    return await retriever.retrieve(
        query=query,
        top_k=20,
        score_threshold=0.5
    )
```

## 🔧 Custom Retriever Development

### **Creating a Custom Retriever**
```python
from src.rag.chatbot.retrievers.base_retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_param = config.get("custom_param", "default")
        self._initialize_custom_components()
    
    def _initialize_custom_components(self):
        # Initialize custom retrieval components
        pass
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[Document]:
        # Implement custom retrieval logic
        
        # 1. Process query
        processed_query = self._preprocess_query(query)
        
        # 2. Generate embedding
        query_embedding = await self._embed_query(processed_query)
        
        # 3. Search vector store
        candidates = await self._search_vectors(
            query_embedding, top_k * 2, filters
        )
        
        # 4. Apply custom scoring
        scored_docs = self._custom_scoring(candidates, query)
        
        # 5. Filter and rank
        final_results = self._filter_and_rank(
            scored_docs, top_k, score_threshold
        )
        
        return final_results
    
    def _custom_scoring(self, documents, query):
        # Implement custom scoring logic
        pass
```

### **Registering Custom Retriever**
```python
# In retriever factory
from src.rag.chatbot.retrievers.retriever_factory import RetrieverFactory

RetrieverFactory.register_retriever("custom", CustomRetriever)

# Use in configuration
chatbot:
  retrieval:
    provider: "custom"
    custom:
      custom_param: "value"
```

## 📚 Related Documentation

- **[Result Rerankers](./rerankers.md)** - Next step: improve result relevance
- **[Response Generators](./generators.md)** - Use retrieved documents for generation
- **[Vector Stores](../ingestion/vector-stores.md)** - Configure the underlying search index
- **[Embedding Models](../ingestion/embedders.md)** - Configure embedding generation

## 🚀 Quick Examples

### **Basic Retrieval**
```python
# Simple document retrieval
retriever = VectorRetriever(config)

documents = await retriever.retrieve(
    query="machine learning algorithms",
    top_k=5,
    score_threshold=0.8
)

for doc in documents:
    print(f"Score: {doc.score:.3f}")
    print(f"Content: {doc.content[:200]}...")
    print(f"Source: {doc.metadata['filename']}")
    print("---")
```

### **Filtered Retrieval**
```bash
# API endpoint with filters
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "troubleshooting network issues",
    "use_retrieval": true,
    "retrieval_filters": {
      "category": "technical",
      "source": "manual",
      "date": {"$gte": "2023-01-01"}
    }
  }'
```

---

**Next Steps**: 
- [Configure Result Reranking](./rerankers.md)
- [Set up Response Generation](./generators.md)
- [Configure Memory Systems](./memory.md)
