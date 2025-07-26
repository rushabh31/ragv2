# Result Rerankers

## 🎯 Overview

Result rerankers improve the relevance of retrieved documents by reordering them based on additional scoring criteria. While retrievers find potentially relevant documents using vector similarity, rerankers apply more sophisticated analysis to identify the most useful documents for answering the specific user query.

## 🔄 How Reranking Works

### **The Process**
1. **Input**: Receive documents from retriever (typically 10-20 documents)
2. **Analysis**: Apply additional scoring methods beyond vector similarity
3. **Scoring**: Calculate relevance scores using multiple criteria
4. **Ranking**: Reorder documents by combined relevance score
5. **Filtering**: Return top-k most relevant documents (typically 3-8)

```python
# Example reranking process
Retriever Output (10 docs):
Doc A: Vector Score 0.92, Content: "Machine learning algorithms..."
Doc B: Vector Score 0.87, Content: "Deep learning networks..."
Doc C: Vector Score 0.85, Content: "Data preprocessing steps..."

Reranker Analysis:
Doc A: Vector 0.92 + Query Match 0.8 + Recency 0.6 = Final Score 0.77
Doc B: Vector 0.87 + Query Match 0.9 + Recency 0.8 = Final Score 0.86
Doc C: Vector 0.85 + Query Match 0.7 + Recency 0.9 = Final Score 0.82

Reranked Output (top 3):
1. Doc B: Final Score 0.86
2. Doc C: Final Score 0.82  
3. Doc A: Final Score 0.77
```

## 🔧 Available Rerankers

### **1. Custom Reranker (Default)**
Text-based relevance scoring using multiple heuristics.

**Features:**
- Query keyword matching
- Content quality assessment
- Metadata-based boosting
- Recency scoring
- Length normalization

**Best for:**
- General-purpose reranking
- Fast processing
- Customizable scoring criteria
- No additional AI model required

### **2. Cross-Encoder Reranker (Future)**
Deep learning model that analyzes query-document pairs.

**Features:**
- Deep semantic understanding
- Query-document interaction modeling
- High accuracy scoring
- Learned relevance patterns

**Best for:**
- Maximum accuracy
- Complex queries
- Domain-specific optimization
- When computational cost is acceptable

## 📋 Custom Reranker Configuration

### **Basic Setup**
```yaml
chatbot:
  reranking:
    provider: "custom"
    custom:
      top_k: 5                         # Final number of documents
      boost_recent: true               # Boost recent documents
      boost_keywords: []               # Keywords to boost
      penalize_keywords: []            # Keywords to penalize
      min_content_length: 50           # Minimum content length
      max_content_length: 2000         # Maximum content length
```

### **Advanced Configuration**
```yaml
chatbot:
  reranking:
    provider: "custom"
    custom:
      top_k: 5
      
      # Scoring weights (must sum to 1.0)
      weights:
        vector_similarity: 0.4         # Original retrieval score
        query_match: 0.3               # Query keyword matching
        content_quality: 0.15          # Content quality metrics
        metadata_boost: 0.1            # Metadata-based scoring
        recency: 0.05                  # Document recency
      
      # Query matching settings
      query_matching:
        exact_match_boost: 2.0         # Boost for exact phrase matches
        partial_match_boost: 1.5       # Boost for partial matches
        stemming: true                 # Use word stemming
        synonyms: true                 # Use synonym matching
        case_sensitive: false          # Case sensitivity
      
      # Content quality settings
      content_quality:
        min_words: 20                  # Minimum word count
        max_words: 500                 # Maximum word count
        penalize_repetition: true      # Penalize repetitive content
        boost_structured: true         # Boost structured content
      
      # Metadata boosting
      metadata_boosts:
        source_manual: 1.2             # Boost manual pages
        category_technical: 1.1        # Boost technical content
        priority_high: 1.3             # Boost high-priority docs
        author_expert: 1.15            # Boost expert authors
      
      # Recency settings
      recency:
        enabled: true
        decay_days: 365                # Days for full decay
        boost_factor: 1.2              # Maximum recency boost
      
      # Diversity settings
      diversity:
        enabled: true
        similarity_threshold: 0.9      # Threshold for similar docs
        max_similar_docs: 2            # Max docs from same source
```

### **Domain-Specific Configuration**
```yaml
# Technical documentation reranking
custom:
  top_k: 8
  boost_keywords: ["error", "troubleshooting", "configuration", "setup"]
  weights:
    vector_similarity: 0.3
    query_match: 0.4                 # Higher weight for exact matches
    content_quality: 0.2
    metadata_boost: 0.1
  metadata_boosts:
    category_troubleshooting: 1.5
    source_official_docs: 1.3

# Customer support reranking
custom:
  top_k: 5
  boost_keywords: ["problem", "issue", "help", "solution"]
  penalize_keywords: ["deprecated", "obsolete", "draft"]
  weights:
    vector_similarity: 0.35
    query_match: 0.35
    content_quality: 0.15
    recency: 0.15                    # Higher weight for recent docs
```

## 🛠️ Reranker Implementation

### **Using Custom Reranker**
```python
from src.rag.chatbot.rerankers.custom_reranker import CustomReranker

# Initialize reranker
reranker = CustomReranker({
    "top_k": 5,
    "boost_recent": True,
    "boost_keywords": ["machine learning", "algorithm"],
    "weights": {
        "vector_similarity": 0.4,
        "query_match": 0.3,
        "content_quality": 0.2,
        "metadata_boost": 0.1
    }
})

# Rerank retrieved documents
query = "machine learning algorithms"
retrieved_docs = [...]  # Documents from retriever

reranked_docs = await reranker.rerank(
    query=query,
    documents=retrieved_docs,
    top_k=5
)

print(f"Reranked to {len(reranked_docs)} documents")
for i, doc in enumerate(reranked_docs):
    print(f"{i+1}. Score: {doc.rerank_score:.3f}")
    print(f"   Content: {doc.content[:100]}...")
    print(f"   Source: {doc.metadata.get('filename')}")
```

### **Custom Scoring Logic**
```python
class AdvancedCustomReranker(CustomReranker):
    def _calculate_custom_score(self, document, query, query_words):
        scores = {}
        
        # Base vector similarity score
        scores['vector'] = document.score
        
        # Query matching score
        scores['query_match'] = self._calculate_query_match(
            document.content, query_words
        )
        
        # Content quality score
        scores['quality'] = self._calculate_content_quality(document)
        
        # Metadata boost score
        scores['metadata'] = self._calculate_metadata_boost(document)
        
        # Recency score
        scores['recency'] = self._calculate_recency_score(document)
        
        # Domain-specific scoring
        scores['domain'] = self._calculate_domain_score(document, query)
        
        # Combine scores with weights
        final_score = sum(
            scores[key] * self.weights.get(key, 0.0)
            for key in scores
        )
        
        return final_score, scores
    
    def _calculate_domain_score(self, document, query):
        # Custom domain-specific scoring logic
        domain_score = 0.0
        
        # Boost technical content for technical queries
        if any(term in query.lower() for term in ['api', 'code', 'function']):
            if document.metadata.get('category') == 'technical':
                domain_score += 0.2
        
        # Boost troubleshooting content for problem queries
        if any(term in query.lower() for term in ['error', 'problem', 'issue']):
            if 'troubleshooting' in document.content.lower():
                domain_score += 0.3
        
        return min(domain_score, 1.0)
```

## 📊 Scoring Components

### **Vector Similarity Score**
- **Source**: Original retrieval score from vector search
- **Range**: 0.0 - 1.0
- **Purpose**: Captures semantic similarity between query and document
- **Weight**: Typically 30-50% of final score

### **Query Match Score**
```python
def calculate_query_match_score(content, query_words):
    content_words = content.lower().split()
    content_set = set(content_words)
    
    # Exact matches
    exact_matches = sum(1 for word in query_words if word in content_set)
    exact_score = exact_matches / len(query_words)
    
    # Phrase matches
    query_phrases = extract_phrases(query_words)
    phrase_matches = sum(1 for phrase in query_phrases if phrase in content.lower())
    phrase_score = phrase_matches / max(len(query_phrases), 1)
    
    # Combine scores
    return (exact_score * 0.7) + (phrase_score * 0.3)
```

### **Content Quality Score**
```python
def calculate_content_quality_score(document):
    content = document.content
    
    # Length score (prefer medium-length content)
    word_count = len(content.split())
    if 50 <= word_count <= 300:
        length_score = 1.0
    elif word_count < 50:
        length_score = word_count / 50
    else:
        length_score = max(0.5, 1.0 - (word_count - 300) / 1000)
    
    # Structure score (prefer structured content)
    structure_indicators = [
        '\n-', '\n*', '\n1.', '\n2.', '##', '###',
        'Step 1', 'Step 2', 'First,', 'Second,'
    ]
    structure_score = min(1.0, sum(
        0.1 for indicator in structure_indicators
        if indicator in content
    ))
    
    # Readability score (prefer clear, readable content)
    sentences = content.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    readability_score = max(0.0, 1.0 - abs(avg_sentence_length - 15) / 15)
    
    return (length_score * 0.4) + (structure_score * 0.3) + (readability_score * 0.3)
```

### **Metadata Boost Score**
```python
def calculate_metadata_boost_score(document, boost_config):
    boost_score = 0.0
    metadata = document.metadata
    
    for field, boost_value in boost_config.items():
        if field.startswith('source_'):
            source_value = field.replace('source_', '')
            if metadata.get('source') == source_value:
                boost_score += boost_value - 1.0
        
        elif field.startswith('category_'):
            category_value = field.replace('category_', '')
            if metadata.get('category') == category_value:
                boost_score += boost_value - 1.0
        
        elif field.startswith('priority_'):
            priority_value = field.replace('priority_', '')
            if metadata.get('priority') == priority_value:
                boost_score += boost_value - 1.0
    
    return min(boost_score, 1.0)
```

### **Recency Score**
```python
def calculate_recency_score(document, decay_days=365):
    from datetime import datetime, timedelta
    
    # Get document date
    doc_date_str = document.metadata.get('date')
    if not doc_date_str:
        return 0.5  # Neutral score for undated documents
    
    try:
        doc_date = datetime.fromisoformat(doc_date_str)
        now = datetime.now()
        days_old = (now - doc_date).days
        
        # Calculate decay score
        if days_old <= 0:
            return 1.0
        elif days_old >= decay_days:
            return 0.0
        else:
            return 1.0 - (days_old / decay_days)
    
    except ValueError:
        return 0.5  # Neutral score for invalid dates
```

## ⚡ Performance Optimization

### **Efficient Scoring**
```python
class OptimizedReranker:
    def __init__(self, config):
        self.config = config
        # Pre-compile regex patterns
        self.keyword_patterns = {
            keyword: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in config.get('boost_keywords', [])
        }
    
    def _fast_query_match(self, content, query_words):
        # Use pre-compiled patterns for faster matching
        content_lower = content.lower()
        matches = 0
        
        for word in query_words:
            if word in content_lower:
                matches += 1
        
        return matches / len(query_words)
    
    def _batch_rerank(self, query, documents):
        # Process documents in batches for better performance
        batch_size = 50
        reranked_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = self._score_batch(query, batch)
            reranked_docs.extend(batch_scores)
        
        return sorted(reranked_docs, key=lambda x: x.rerank_score, reverse=True)
```

### **Caching Strategies**
```python
class CachedReranker:
    def __init__(self, base_reranker):
        self.base_reranker = base_reranker
        self.score_cache = {}
        self.query_cache = {}
    
    async def rerank(self, query, documents, top_k):
        # Cache query analysis
        if query not in self.query_cache:
            self.query_cache[query] = self._analyze_query(query)
        
        query_analysis = self.query_cache[query]
        
        # Score documents with caching
        scored_docs = []
        for doc in documents:
            cache_key = self._get_doc_cache_key(doc, query)
            
            if cache_key in self.score_cache:
                score = self.score_cache[cache_key]
            else:
                score = await self._calculate_score(doc, query_analysis)
                self.score_cache[cache_key] = score
            
            doc.rerank_score = score
            scored_docs.append(doc)
        
        # Sort and return top_k
        scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored_docs[:top_k]
```

## 🚨 Common Issues and Solutions

### **Poor Reranking Quality**
```yaml
# Issue: Reranking not improving results
# Solution: Adjust scoring weights
reranking:
  custom:
    weights:
      vector_similarity: 0.3         # Reduce if vector search is poor
      query_match: 0.5               # Increase for better keyword matching
      content_quality: 0.2           # Adjust based on content quality

# Issue: Missing relevant documents
# Solution: Increase top_k and adjust thresholds
reranking:
  custom:
    top_k: 8                         # Increase from 5
    min_content_length: 20           # Reduce from 50
```

### **Slow Reranking Performance**
```yaml
# Issue: Reranking taking too long
# Solution: Simplify scoring and reduce features
reranking:
  custom:
    weights:
      vector_similarity: 0.6         # Rely more on vector scores
      query_match: 0.4               # Simplify to just query matching
    query_matching:
      stemming: false                # Disable expensive features
      synonyms: false
```

### **Inconsistent Results**
```python
# Issue: Reranking results vary significantly
# Solution: Normalize scores and add stability
def normalize_scores(documents):
    scores = [doc.rerank_score for doc in documents]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range > 0:
        for doc in documents:
            doc.rerank_score = (doc.rerank_score - min_score) / score_range
    
    return documents
```

## 🎯 Best Practices

### **Weight Tuning**
1. **Start balanced**: Use equal weights initially
2. **Measure impact**: Test each component's contribution
3. **Domain-specific**: Adjust weights for your use case
4. **Iterative tuning**: Make small adjustments and test

### **Quality Assurance**
```python
def evaluate_reranking_quality(test_queries, reranker):
    improvements = []
    
    for query_data in test_queries:
        query = query_data["query"]
        documents = query_data["documents"]
        expected_order = query_data["expected_order"]
        
        # Get original order (by vector score)
        original_order = sorted(documents, key=lambda x: x.score, reverse=True)
        
        # Get reranked order
        reranked_docs = await reranker.rerank(query, documents, len(documents))
        
        # Calculate ranking improvements
        original_ndcg = calculate_ndcg(original_order, expected_order)
        reranked_ndcg = calculate_ndcg(reranked_docs, expected_order)
        
        improvement = reranked_ndcg - original_ndcg
        improvements.append(improvement)
    
    avg_improvement = sum(improvements) / len(improvements)
    return avg_improvement
```

### **A/B Testing**
```python
class ABTestReranker:
    def __init__(self, reranker_a, reranker_b, split_ratio=0.5):
        self.reranker_a = reranker_a
        self.reranker_b = reranker_b
        self.split_ratio = split_ratio
        self.results_a = []
        self.results_b = []
    
    async def rerank(self, query, documents, top_k):
        # Randomly assign to A or B group
        use_a = random.random() < self.split_ratio
        
        if use_a:
            results = await self.reranker_a.rerank(query, documents, top_k)
            self.results_a.append({
                "query": query,
                "results": results,
                "timestamp": datetime.now()
            })
        else:
            results = await self.reranker_b.rerank(query, documents, top_k)
            self.results_b.append({
                "query": query,
                "results": results,
                "timestamp": datetime.now()
            })
        
        return results
    
    def get_performance_comparison(self):
        # Compare performance metrics between A and B
        pass
```

## 🔧 Custom Reranker Development

### **Creating a Custom Reranker**
```python
from src.rag.chatbot.rerankers.base_reranker import BaseReranker

class DomainSpecificReranker(BaseReranker):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_keywords = config.get("domain_keywords", {})
        self.domain_weights = config.get("domain_weights", {})
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        # Analyze query for domain
        query_domain = self._detect_query_domain(query)
        
        # Score documents based on domain
        for doc in documents:
            base_score = doc.score
            domain_score = self._calculate_domain_score(doc, query_domain)
            doc.rerank_score = (base_score * 0.7) + (domain_score * 0.3)
        
        # Sort and return top_k
        documents.sort(key=lambda x: x.rerank_score, reverse=True)
        return documents[:top_k]
    
    def _detect_query_domain(self, query):
        # Detect domain from query keywords
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                return domain
        return "general"
    
    def _calculate_domain_score(self, document, query_domain):
        # Calculate domain-specific relevance score
        doc_content = document.content.lower()
        domain_keywords = self.domain_keywords.get(query_domain, [])
        
        keyword_matches = sum(1 for kw in domain_keywords if kw in doc_content)
        return min(1.0, keyword_matches / max(len(domain_keywords), 1))
```

### **Registering Custom Reranker**
```python
# In reranker factory
from src.rag.chatbot.rerankers.reranker_factory import RerankerFactory

RerankerFactory.register_reranker("domain_specific", DomainSpecificReranker)

# Use in configuration
chatbot:
  reranking:
    provider: "domain_specific"
    domain_specific:
      domain_keywords:
        technical: ["api", "function", "code", "algorithm"]
        support: ["error", "problem", "troubleshoot", "fix"]
        business: ["process", "workflow", "policy", "procedure"]
```

## 📚 Related Documentation

- **[Document Retrievers](./retrievers.md)** - Previous step: retrieve candidate documents
- **[Response Generators](./generators.md)** - Next step: generate responses with reranked documents
- **[Memory Systems](./memory.md)** - Integrate conversation context
- **[Chatbot API](./api.md)** - Use reranking via API

## 🚀 Quick Examples

### **Test Reranking Impact**
```python
# Compare retrieval vs reranked results
retriever = VectorRetriever(config)
reranker = CustomReranker(config)

query = "machine learning algorithms"

# Get retrieval results
retrieved_docs = await retriever.retrieve(query, top_k=10)
print("Top 3 Retrieved Documents:")
for i, doc in enumerate(retrieved_docs[:3]):
    print(f"{i+1}. Score: {doc.score:.3f} - {doc.content[:100]}...")

# Get reranked results
reranked_docs = await reranker.rerank(query, retrieved_docs, top_k=5)
print("\nTop 3 Reranked Documents:")
for i, doc in enumerate(reranked_docs[:3]):
    print(f"{i+1}. Score: {doc.rerank_score:.3f} - {doc.content[:100]}...")
```

### **API Usage with Reranking**
```bash
# Chatbot API automatically uses configured reranker
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "troubleshooting network connectivity issues",
    "use_retrieval": true
  }'
```

---

**Next Steps**: 
- [Configure Response Generation](./generators.md)
- [Set up Memory Systems](./memory.md)
- [Use the Chatbot API](./api.md)
