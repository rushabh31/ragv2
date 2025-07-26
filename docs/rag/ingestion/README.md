# Ingestion Pipeline Overview

## 🎯 What is the Ingestion Pipeline?

The ingestion pipeline is the first stage of the RAG system that transforms your raw documents into searchable, AI-ready knowledge. It takes documents in various formats (PDF, Word, images) and converts them into vector embeddings stored in a searchable database.

## 🔄 Pipeline Flow

```
📄 Raw Documents
       ↓
🔧 Document Parsing (Extract text/OCR)
       ↓
✂️ Text Chunking (Split into manageable pieces)
       ↓
🧮 Embedding Generation (Convert to vectors)
       ↓
💾 Vector Storage (Store in searchable database)
       ↓
✅ Ready for Chatbot Queries
```

## 🏗️ Architecture Components

### **1. Document Parsers**
Transform various document formats into plain text:
- **Vision Parser**: OCR for PDFs, images, scanned documents using Vertex AI Gemini Vision
- **Simple Text Parser**: Fast processing for plain text files
- **Future**: Word documents, PowerPoint, Excel files

### **2. Text Chunkers**
Split large documents into manageable pieces:
- **Semantic Chunker**: AI-powered boundary detection for natural breaks
- **Sliding Window Chunker**: Overlap-based chunking with fixed sizes
- **Future**: Paragraph-aware, table-aware chunking

### **3. Embedding Generators**
Convert text chunks into numerical vectors:
- **Vertex AI Embeddings**: Google's text-embedding-004 model
- **OpenAI Embeddings**: text-embedding-3-large model
- **Future**: Custom domain-specific embeddings

### **4. Vector Stores**
Store and index embeddings for fast retrieval:
- **FAISS**: Facebook's similarity search library
- **Future**: Pinecone, Weaviate, Chroma integration

## 🚀 Key Features

### **📊 Parallel Processing**
- Process multiple PDF pages simultaneously
- Configurable concurrency levels (2-12 pages)
- 2-5x speedup for large documents
- Automatic fallback to sequential processing

### **🔍 Vision OCR Capabilities**
- Extract text from scanned PDFs
- Process images with text content
- Handle complex layouts and tables
- Multi-language text recognition

### **🧠 Intelligent Chunking**
- AI-powered semantic boundary detection
- Preserve context across chunk boundaries
- Configurable chunk sizes and overlaps
- Maintain document structure information

### **⚡ Batch Processing**
- Efficient embedding generation
- Configurable batch sizes (10-200 texts)
- Optimized API usage
- Progress tracking and logging

## 📋 Supported File Formats

### **Currently Supported**
- **PDF**: Text and image-based PDFs
- **Text Files**: .txt, .md, .csv
- **Images**: .png, .jpg, .jpeg (with OCR)

### **Planned Support**
- **Microsoft Office**: .docx, .pptx, .xlsx
- **Other Formats**: .html, .xml, .json
- **Archives**: .zip processing

## ⚙️ Configuration Options

### **Basic Configuration**
```yaml
ingestion:
  parsing:
    provider: "vision_parser"          # Document parser type
    vision:
      model: "gemini-1.5-pro-002"     # Vertex AI model
      max_pages: 100                  # Maximum pages to process
      max_concurrent_pages: 5         # Parallel processing level
  
  chunking:
    provider: "semantic"               # Chunking strategy
    semantic:
      chunk_size: 1000                # Target chunk size
      overlap: 200                    # Overlap between chunks
  
  embedding:
    provider: "vertex"                 # Embedding provider
    vertex:
      model: "text-embedding-004"     # Embedding model
      batch_size: 100                 # Batch processing size
  
  vector_store:
    provider: "faiss"                  # Vector database
    faiss:
      index_type: "HNSW"              # Index algorithm
      dimension: 768                  # Embedding dimension
```

### **Performance Tuning**
```yaml
# High-performance setup
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 10        # More parallel processing
  embedding:
    vertex:
      batch_size: 200                 # Larger batches
  vector_store:
    faiss:
      ef_construction: 400            # Better index quality

# Memory-optimized setup
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 2         # Reduce memory usage
  embedding:
    vertex:
      batch_size: 20                  # Smaller batches
```

## 🛠️ API Endpoints

### **Document Upload**
```bash
POST /ingest/file
Content-Type: multipart/form-data

# Upload a single file
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@document.pdf" \
  -F "metadata={\"source\": \"user_upload\", \"category\": \"manual\"}"
```

### **Batch Upload**
```bash
POST /ingest/batch
Content-Type: multipart/form-data

# Upload multiple files
curl -X POST "http://localhost:8000/ingest/batch" \
  -H "X-API-Key: test-api-key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "metadata={\"batch_id\": \"batch_001\"}"
```

### **URL Processing**
```bash
POST /ingest/url
Content-Type: application/json

# Process document from URL
curl -X POST "http://localhost:8000/ingest/url" \
  -H "X-API-Key: test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "metadata": {"source": "web"}
  }'
```

### **Processing Status**
```bash
GET /ingest/status/{job_id}

# Check processing status
curl -X GET "http://localhost:8000/ingest/status/job_123" \
  -H "X-API-Key: test-api-key"
```

## 📊 Performance Characteristics

### **Processing Speed**
- **Small PDFs (1-5 pages)**: 10-30 seconds
- **Medium PDFs (10-20 pages)**: 1-3 minutes
- **Large PDFs (50+ pages)**: 5-15 minutes
- **Parallel processing**: 2-5x speedup typical

### **Accuracy Metrics**
- **Text extraction**: 95%+ accuracy for clear text
- **OCR accuracy**: 85-95% for scanned documents
- **Chunk boundary**: 90%+ semantic coherence
- **Embedding quality**: Cosine similarity 0.8+ for related content

### **Resource Usage**
- **Memory**: 100-500MB per concurrent page
- **CPU**: Scales with concurrency level
- **Storage**: ~1KB per 1000 characters
- **API calls**: 1-3 calls per page (vision + embedding)

## 🚨 Error Handling

### **Robust Fallback System**
1. **Vision parsing fails** → Fall back to text extraction
2. **Text extraction fails** → Use error placeholder
3. **Embedding fails** → Retry with smaller batches
4. **Vector storage fails** → Queue for retry

### **Common Issues and Solutions**

**Issue**: "Authentication failed"
```bash
# Check Vertex AI credentials
python -c "
from src.models.vision import VertexVisionAI
model = VertexVisionAI()
print('Auth status:', model.get_auth_health_status())
"
```

**Issue**: "Out of memory"
```yaml
# Reduce concurrency in config.yaml
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 2  # Reduce from 5
```

**Issue**: "Processing too slow"
```yaml
# Increase concurrency (if you have enough memory)
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 8  # Increase from 5
  embedding:
    vertex:
      batch_size: 200  # Increase from 100
```

## 🔍 Monitoring and Logging

### **Built-in Metrics**
- Processing time per document
- Success/failure rates
- Memory usage tracking
- API call statistics

### **Log Levels**
```python
# Enable detailed logging
import logging
logging.getLogger('ingestion').setLevel(logging.DEBUG)
```

### **Health Checks**
```bash
# Check ingestion service health
curl -X GET "http://localhost:8000/health"

# Response
{
  "status": "healthy",
  "components": {
    "parser": "ready",
    "embedder": "ready",
    "vector_store": "ready"
  }
}
```

## 🎯 Best Practices

### **Document Preparation**
1. **Clean PDFs**: Remove password protection
2. **Optimize Images**: Use high-resolution scans (300+ DPI)
3. **Structure**: Maintain clear document hierarchy
4. **Metadata**: Include relevant document metadata

### **Performance Optimization**
1. **Batch Processing**: Upload multiple documents together
2. **Concurrency**: Tune based on available memory
3. **Monitoring**: Watch for memory and API limits
4. **Caching**: Avoid reprocessing identical documents

### **Quality Assurance**
1. **Test Samples**: Process sample documents first
2. **Verify Output**: Check extracted text quality
3. **Monitor Metrics**: Track processing success rates
4. **Iterative Improvement**: Adjust settings based on results

## 📚 Component Deep Dives

- **[Document Parsers](./parsers.md)** - Detailed parser configuration and usage
- **[Text Chunkers](./chunkers.md)** - Chunking strategies and optimization
- **[Embedding Models](./embedders.md)** - Embedding generation and providers
- **[Vector Stores](./vector-stores.md)** - Vector database setup and management
- **[Ingestion API](./api.md)** - Complete API reference

## 🚀 Quick Start Example

```python
# Python SDK example
from src.rag.ingestion import IngestionPipeline

# Initialize pipeline
pipeline = IngestionPipeline(config_path="config.yaml")

# Process a document
result = await pipeline.process_file(
    file_path="document.pdf",
    metadata={"source": "manual", "category": "technical"}
)

print(f"Processed {result.chunks_created} chunks")
print(f"Processing time: {result.processing_time:.2f}s")
```

---

**Next Steps**: 
- [Configure Document Parsers](./parsers.md)
- [Set up Text Chunking](./chunkers.md)
- [Start Processing Documents](./api.md)
