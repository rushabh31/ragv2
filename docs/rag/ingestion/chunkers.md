# Text Chunkers

## 🎯 Overview

Text chunkers split large documents into smaller, manageable pieces that can be effectively processed by AI models. Proper chunking is crucial for RAG performance as it determines how information is retrieved and presented to users.

## 🧩 Why Chunking Matters

### **The Challenge**
- AI models have token limits (typically 4K-32K tokens)
- Large documents exceed these limits
- Need to maintain context and meaning
- Balance between detail and relevance

### **The Solution**
Intelligent chunking strategies that:
- Preserve semantic meaning
- Maintain context boundaries
- Optimize for retrieval accuracy
- Enable efficient vector search

## 🔧 Available Chunkers

### **1. Semantic Chunker (Recommended)**
Uses AI to detect natural content boundaries and create semantically coherent chunks.

**Best for:**
- Complex documents with varied structure
- Maintaining context and meaning
- High-quality retrieval results
- Production deployments

**Features:**
- AI-powered boundary detection
- Preserves semantic coherence
- Adaptive chunk sizing
- Context-aware splitting

### **2. Sliding Window Chunker**
Creates overlapping chunks with fixed sizes for consistent processing.

**Best for:**
- Simple document structures
- Fast processing requirements
- Development and testing
- Predictable chunk sizes

**Features:**
- Fixed chunk sizes
- Configurable overlap
- Fast processing
- Predictable output

## 📋 Semantic Chunker Configuration

### **Basic Setup**
```yaml
ingestion:
  chunking:
    provider: "semantic"
    semantic:
      chunk_size: 1000                # Target chunk size (characters)
      overlap: 200                    # Overlap between chunks
      use_llm_boundary: true          # Use AI for boundary detection
      min_chunk_size: 100             # Minimum chunk size
      max_chunk_size: 2000            # Maximum chunk size
```

### **Advanced Configuration**
```yaml
ingestion:
  chunking:
    provider: "semantic"
    semantic:
      chunk_size: 1000
      overlap: 200
      use_llm_boundary: true
      min_chunk_size: 100
      max_chunk_size: 2000
      
      # AI model for boundary detection
      llm_model: "gemini-1.5-pro-002"
      llm_region: "us-central1"
      
      # Boundary detection prompt
      boundary_prompt: |
        Analyze this text and identify natural breaking points.
        Look for:
        - Topic changes
        - Section boundaries
        - Logical breaks
        - Paragraph endings
        
      # Semantic similarity threshold
      similarity_threshold: 0.8       # Higher = more similar chunks
      
      # Processing options
      preserve_sentences: true        # Don't break mid-sentence
      preserve_paragraphs: true       # Prefer paragraph boundaries
      merge_short_chunks: true        # Combine very short chunks
```

### **Performance Tuning**
```yaml
# High-quality setup (slower, better results)
semantic:
  chunk_size: 800                   # Smaller, more focused chunks
  overlap: 150                      # More overlap for context
  use_llm_boundary: true            # AI boundary detection
  similarity_threshold: 0.85        # Higher similarity requirement

# Balanced setup (recommended)
semantic:
  chunk_size: 1000                  # Good balance
  overlap: 200                      # Standard overlap
  use_llm_boundary: true            # AI boundaries
  similarity_threshold: 0.8         # Reasonable similarity

# Fast setup (faster, simpler results)
semantic:
  chunk_size: 1200                  # Larger chunks
  overlap: 100                      # Less overlap
  use_llm_boundary: false           # No AI processing
  similarity_threshold: 0.7         # Lower similarity requirement
```

## 🪟 Sliding Window Chunker

### **Configuration**
```yaml
ingestion:
  chunking:
    provider: "sliding_window"
    sliding_window:
      chunk_size: 1000              # Fixed chunk size (characters)
      overlap: 200                  # Overlap between chunks
      stride: 800                   # Step size (chunk_size - overlap)
      preserve_words: true          # Don't break words
      preserve_sentences: false     # Allow sentence breaks
```

### **How It Works**
```python
# Example: Document with 3000 characters, chunk_size=1000, overlap=200
# Chunk 1: characters 0-1000
# Chunk 2: characters 800-1800 (overlap of 200)
# Chunk 3: characters 1600-2600 (overlap of 200)
# Chunk 4: characters 2400-3000 (remaining content)
```

### **Use Cases**
```python
# Best for:
- Uniform document structure
- Fast processing requirements
- Simple text documents
- Development and testing

# Consider semantic chunker for:
- Complex document layouts
- Mixed content types
- Production deployments
- High-quality retrieval
```

## 🧠 Semantic Chunker Deep Dive

### **AI-Powered Boundary Detection**
The semantic chunker uses Vertex AI Gemini to identify natural breaking points:

```python
# The AI analyzes text for:
- Topic transitions
- Section headers
- Paragraph boundaries
- Logical breaks
- Context shifts
```

### **Boundary Detection Process**
1. **Text Analysis**: AI examines document structure
2. **Boundary Identification**: Finds natural break points
3. **Chunk Creation**: Creates chunks at identified boundaries
4. **Size Optimization**: Adjusts chunks to target size
5. **Overlap Addition**: Adds context overlap between chunks

### **Quality Metrics**
- **Semantic Coherence**: 85-95% topic consistency within chunks
- **Context Preservation**: 90%+ relevant context maintained
- **Boundary Accuracy**: 80-90% natural break point detection
- **Size Distribution**: 90%+ chunks within target size range

## 🛠️ Chunker Implementation

### **Using Semantic Chunker**
```python
from src.rag.ingestion.chunkers.semantic_chunker import SemanticChunker

# Initialize chunker
chunker = SemanticChunker({
    "chunk_size": 1000,
    "overlap": 200,
    "use_llm_boundary": True,
    "min_chunk_size": 100,
    "max_chunk_size": 2000
})

# Chunk a document
chunks = await chunker.chunk_documents([document])

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.content)} characters")
    print(f"Metadata: {chunk.metadata}")
```

### **Using Sliding Window Chunker**
```python
from src.rag.ingestion.chunkers.sliding_window_chunker import SlidingWindowChunker

# Initialize chunker
chunker = SlidingWindowChunker({
    "chunk_size": 1000,
    "overlap": 200,
    "preserve_words": True
})

# Chunk a document
chunks = await chunker.chunk_documents([document])

print(f"Created {len(chunks)} chunks with consistent overlap")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.content)} characters")
```

## 📊 Chunk Metadata

### **Automatic Metadata**
Each chunk includes comprehensive metadata:

```python
{
    "chunk_id": "doc_123_chunk_001",      # Unique chunk identifier
    "source_document": "document.pdf",    # Original document
    "chunk_index": 0,                     # Position in document
    "chunk_size": 987,                    # Actual chunk size
    "overlap_start": 0,                   # Overlap with previous chunk
    "overlap_end": 200,                   # Overlap with next chunk
    "page_numbers": [1, 2],               # Source pages
    "section": "Introduction",            # Document section
    "chunking_method": "semantic",        # Chunker used
    "boundary_type": "topic_change",      # Type of boundary detected
    "semantic_score": 0.85,               # Semantic coherence score
    "processing_time": 1.2,               # Processing time (seconds)
    "created_at": "2024-01-15T10:30:00Z"  # Creation timestamp
}
```

### **Custom Metadata**
Add domain-specific metadata:

```python
# Custom metadata for legal documents
legal_metadata = {
    "document_type": "contract",
    "jurisdiction": "california",
    "practice_area": "employment",
    "confidentiality": "attorney_client"
}

# Custom metadata for technical manuals
technical_metadata = {
    "product": "software_v2.1",
    "category": "user_guide",
    "difficulty": "intermediate",
    "last_updated": "2024-01-15"
}
```

## 🚨 Error Handling

### **Common Issues and Solutions**

**Issue**: "Chunks too large for embedding model"
```yaml
# Reduce maximum chunk size
chunking:
  semantic:
    max_chunk_size: 1500  # Reduce from 2000
    chunk_size: 800       # Reduce target size
```

**Issue**: "Chunks losing context"
```yaml
# Increase overlap
chunking:
  semantic:
    overlap: 300          # Increase from 200
    preserve_sentences: true
    preserve_paragraphs: true
```

**Issue**: "Too many small chunks"
```yaml
# Increase minimum chunk size
chunking:
  semantic:
    min_chunk_size: 200   # Increase from 100
    merge_short_chunks: true
```

**Issue**: "Semantic chunker too slow"
```yaml
# Disable AI boundary detection for speed
chunking:
  semantic:
    use_llm_boundary: false  # Faster processing
    chunk_size: 1200         # Larger chunks
```

### **Fallback Mechanisms**
```python
# The chunker implements fallbacks:
1. AI boundary detection fails → Use paragraph boundaries
2. Paragraph boundaries fail → Use sentence boundaries  
3. Sentence boundaries fail → Use word boundaries
4. All methods fail → Use character-based chunking
```

## 📈 Performance Optimization

### **Processing Speed**
- **Semantic Chunker**: 1-5 seconds per 1000 characters
- **Sliding Window**: 0.1-0.5 seconds per 1000 characters
- **AI Boundary Detection**: Adds 2-3x processing time
- **Batch Processing**: 50-80% speed improvement

### **Memory Usage**
- **Base Memory**: 50-100MB per chunker instance
- **AI Processing**: Additional 200-500MB
- **Large Documents**: Scales with document size
- **Concurrent Processing**: Memory × concurrent instances

### **Quality vs Speed Trade-offs**
```yaml
# Maximum Quality (slowest)
semantic:
  use_llm_boundary: true
  chunk_size: 800
  overlap: 250
  similarity_threshold: 0.9

# Balanced (recommended)
semantic:
  use_llm_boundary: true
  chunk_size: 1000
  overlap: 200
  similarity_threshold: 0.8

# Maximum Speed (fastest)
sliding_window:
  chunk_size: 1200
  overlap: 100
  preserve_words: true
```

## 🎯 Best Practices

### **Chunk Size Guidelines**
1. **Small Chunks (500-800 chars)**: High precision, may lose context
2. **Medium Chunks (800-1200 chars)**: Balanced precision and context
3. **Large Chunks (1200-2000 chars)**: More context, may reduce precision

### **Overlap Recommendations**
1. **Low Overlap (100-150 chars)**: Fast processing, potential context loss
2. **Medium Overlap (150-250 chars)**: Good balance (recommended)
3. **High Overlap (250-400 chars)**: Maximum context, slower processing

### **Domain-Specific Optimization**
```yaml
# Legal documents (preserve structure)
semantic:
  chunk_size: 1200
  overlap: 300
  preserve_paragraphs: true
  boundary_prompt: "Identify legal section boundaries and clause breaks"

# Technical manuals (preserve procedures)
semantic:
  chunk_size: 800
  overlap: 200
  preserve_sentences: true
  boundary_prompt: "Identify step boundaries and procedure breaks"

# Research papers (preserve arguments)
semantic:
  chunk_size: 1000
  overlap: 250
  use_llm_boundary: true
  boundary_prompt: "Identify argument boundaries and topic transitions"
```

## 🔧 Custom Chunker Development

### **Creating a Custom Chunker**
```python
from src.rag.ingestion.chunkers.base_chunker import BaseChunker
from src.rag.shared.models.document import Document

class CustomChunker(BaseChunker):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_size = config.get("custom_size", 1000)
        self.custom_strategy = config.get("strategy", "default")
    
    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _chunk_single_document(self, document: Document) -> List[Document]:
        # Implement custom chunking logic
        content = document.content
        chunks = self._split_content(content)
        
        return [
            Document(
                content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunking_method": "custom",
                    "chunk_size": len(chunk)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_content(self, content: str) -> List[str]:
        # Custom splitting logic
        pass
```

### **Registering Custom Chunker**
```python
# In chunker factory
from src.rag.ingestion.chunkers.chunker_factory import ChunkerFactory

ChunkerFactory.register_chunker("custom", CustomChunker)

# Use in configuration
ingestion:
  chunking:
    provider: "custom"
    custom:
      custom_size: 1500
      strategy: "domain_specific"
```

## 📚 Related Documentation

- **[Document Parsers](./parsers.md)** - Previous step: extract text from documents
- **[Embedding Models](./embedders.md)** - Next step: convert chunks to vectors
- **[Vector Stores](./vector-stores.md)** - Store and search chunk embeddings
- **[Configuration Guide](../../configuration.md)** - Complete configuration reference

## 🚀 Quick Examples

### **Test Chunking Strategy**
```python
# Test different chunking approaches
from src.rag.ingestion.chunkers import SemanticChunker, SlidingWindowChunker

# Sample document
document = Document(content="Your long document content here...")

# Test semantic chunking
semantic_chunker = SemanticChunker({"chunk_size": 1000, "overlap": 200})
semantic_chunks = await semantic_chunker.chunk_documents([document])

# Test sliding window chunking
window_chunker = SlidingWindowChunker({"chunk_size": 1000, "overlap": 200})
window_chunks = await window_chunker.chunk_documents([document])

print(f"Semantic chunker: {len(semantic_chunks)} chunks")
print(f"Sliding window: {len(window_chunks)} chunks")
```

### **Analyze Chunk Quality**
```python
# Analyze chunk characteristics
def analyze_chunks(chunks):
    sizes = [len(chunk.content) for chunk in chunks]
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Average size: {sum(sizes) / len(sizes):.0f} characters")
    print(f"Size range: {min(sizes)}-{max(sizes)} characters")
    print(f"Size std dev: {statistics.stdev(sizes):.0f}")

analyze_chunks(semantic_chunks)
```

---

**Next Steps**: 
- [Configure Embedding Generation](./embedders.md)
- [Set up Vector Storage](./vector-stores.md)
- [Use the Ingestion API](./api.md)
