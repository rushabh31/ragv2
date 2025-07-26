# Document Parsers

## 🎯 Overview

Document parsers are the first step in the ingestion pipeline, responsible for extracting text content from various document formats. The system supports multiple parsing strategies optimized for different document types and use cases.

## 🔧 Available Parsers

### **1. Vision Parser (Recommended)**
Uses Vertex AI Gemini Vision for OCR and text extraction from any document type.

**Best for:**
- Scanned PDFs and images
- Documents with complex layouts
- Mixed text and image content
- High-accuracy text extraction

**Features:**
- Parallel page processing
- Advanced OCR capabilities
- Layout preservation
- Multi-language support

### **2. Simple Text Parser**
Fast text extraction for plain text documents.

**Best for:**
- Plain text files (.txt, .md)
- Development and testing
- Simple document structures
- High-speed processing

**Features:**
- Minimal processing overhead
- Direct text extraction
- No AI dependencies
- Fast processing

## 📋 Vision Parser Configuration

### **Basic Setup**
```yaml
ingestion:
  parsing:
    provider: "vision_parser"
    vision:
      model: "gemini-1.5-pro-002"      # Vertex AI model
      region: "us-central1"            # GCP region
      max_pages: 100                   # Maximum pages to process
      max_concurrent_pages: 5          # Parallel processing level
      prompt_template: "Extract all text content from this document image."
```

### **Advanced Configuration**
```yaml
ingestion:
  parsing:
    provider: "vision_parser"
    vision:
      model: "gemini-1.5-pro-002"
      region: "us-central1"
      max_pages: 100
      max_concurrent_pages: 8          # Higher concurrency
      prompt_template: |               # Custom prompt
        Extract all text content from this document image.
        Preserve formatting, tables, and structure.
        Include any visible text, headers, and captions.
      timeout_seconds: 300             # Processing timeout
      retry_attempts: 3                # Retry failed pages
```

### **Performance Tuning**
```yaml
# High-performance setup (requires more memory)
vision:
  max_concurrent_pages: 12            # Maximum parallelism
  batch_size: 10                      # Process pages in batches

# Memory-optimized setup
vision:
  max_concurrent_pages: 2             # Reduce memory usage
  batch_size: 5                       # Smaller batches

# Balanced setup (recommended)
vision:
  max_concurrent_pages: 5             # Good balance
  batch_size: 8                       # Efficient processing
```

## 🚀 Parallel Processing

### **How It Works**
The Vision Parser processes multiple PDF pages simultaneously using asyncio and semaphore-based concurrency control:

```python
# Example: Processing a 20-page PDF with max_concurrent_pages=5
# Pages 1-5: Process in parallel (batch 1)
# Pages 6-10: Process in parallel (batch 2)  
# Pages 11-15: Process in parallel (batch 3)
# Pages 16-20: Process in parallel (batch 4)
```

### **Performance Guidelines**
- **Small documents (1-5 pages)**: `max_concurrent_pages: 2-3`
- **Medium documents (6-20 pages)**: `max_concurrent_pages: 5-8`
- **Large documents (20+ pages)**: `max_concurrent_pages: 8-12`

### **Memory Requirements**
- **Per concurrent page**: ~100-200MB RAM
- **5 concurrent pages**: ~1GB RAM
- **10 concurrent pages**: ~2GB RAM

## 🔍 Vision Parser Features

### **OCR Capabilities**
```python
# The Vision Parser can extract text from:
- Scanned PDF documents
- Image files (PNG, JPG, JPEG)
- Screenshots and photos of documents
- Complex layouts with tables and charts
- Multi-column text layouts
- Handwritten text (limited accuracy)
```

### **Text Extraction Quality**
- **Printed text**: 95-99% accuracy
- **Clear scans**: 90-95% accuracy
- **Poor quality scans**: 70-85% accuracy
- **Handwritten text**: 60-80% accuracy

### **Language Support**
The Vision Parser supports text extraction in multiple languages:
- English (primary)
- Spanish, French, German
- Chinese, Japanese, Korean
- Arabic, Hebrew
- Many other languages supported by Vertex AI

## 📄 Simple Text Parser

### **Configuration**
```yaml
ingestion:
  parsing:
    provider: "simple_text"
    simple_text:
      encoding: "utf-8"               # Text encoding
      chunk_on_read: false            # Read entire file at once
      preserve_whitespace: true       # Keep original formatting
```

### **Supported Formats**
- `.txt` - Plain text files
- `.md` - Markdown files
- `.csv` - Comma-separated values
- `.log` - Log files
- Any UTF-8 encoded text file

### **Use Cases**
```python
# Best for:
- Configuration files
- Log files
- Code documentation
- Simple reports
- Development and testing

# Not suitable for:
- PDF documents
- Image files
- Complex formatted documents
- Scanned documents
```

## 🛠️ Parser Implementation

### **Using Vision Parser**
```python
from src.rag.ingestion.parsers.vision_parser import VisionParser

# Initialize parser
parser = VisionParser({
    "model": "gemini-1.5-pro-002",
    "max_pages": 50,
    "max_concurrent_pages": 5
})

# Parse a document
documents = await parser.parse_file("document.pdf", {
    "source": "user_upload",
    "category": "manual"
})

print(f"Extracted {len(documents)} document chunks")
for doc in documents:
    print(f"Page {doc.metadata.get('page_number')}: {len(doc.content)} characters")
```

### **Using Simple Text Parser**
```python
from src.rag.ingestion.parsers.simple_text_parser import SimpleTextParser

# Initialize parser
parser = SimpleTextParser({
    "encoding": "utf-8",
    "preserve_whitespace": True
})

# Parse a text file
documents = await parser.parse_file("document.txt", {
    "source": "user_upload",
    "type": "text"
})

print(f"Extracted content: {len(documents[0].content)} characters")
```

## 🚨 Error Handling

### **Vision Parser Fallbacks**
The Vision Parser implements multiple fallback mechanisms:

1. **Individual page failure**: Continue processing other pages
2. **Vision API failure**: Fall back to text extraction
3. **Text extraction failure**: Use error placeholder
4. **Parallel processing failure**: Fall back to sequential processing

```python
# Example error handling flow
try:
    # Try vision extraction
    text = await vision_model.parse_text_from_image(image)
except VisionAPIError:
    # Fall back to text extraction
    text = extract_text_from_pdf_page(page)
except Exception:
    # Use placeholder
    text = f"[Error processing page {page_num}]"
```

### **Common Issues and Solutions**

**Issue**: "Authentication failed"
```bash
# Check Vertex AI authentication
python -c "
from src.models.vision import VertexVisionAI
model = VertexVisionAI()
print('Auth status:', model.get_auth_health_status())
"
```

**Issue**: "Out of memory during processing"
```yaml
# Reduce concurrency in config
ingestion:
  parsing:
    vision:
      max_concurrent_pages: 2  # Reduce from default 5
```

**Issue**: "Processing timeout"
```yaml
# Increase timeout
ingestion:
  parsing:
    vision:
      timeout_seconds: 600  # Increase from default 300
```

**Issue**: "Poor OCR quality"
```yaml
# Use custom prompt for better extraction
ingestion:
  parsing:
    vision:
      prompt_template: |
        Extract all text from this document image with high accuracy.
        Pay special attention to:
        - Tables and structured data
        - Headers and titles
        - Small text and footnotes
        Preserve the original formatting and structure.
```

## 📊 Performance Monitoring

### **Built-in Metrics**
```python
# The parser tracks:
- Processing time per page
- Success/failure rates
- Memory usage
- API call statistics
- OCR accuracy estimates
```

### **Logging Configuration**
```python
import logging

# Enable detailed parser logging
logging.getLogger('vision_parser').setLevel(logging.DEBUG)
logging.getLogger('simple_text_parser').setLevel(logging.INFO)

# Example log output
# INFO:vision_parser:Processing document.pdf (20 pages)
# DEBUG:vision_parser:Starting parallel processing with 5 workers
# INFO:vision_parser:Page 1/20 processed (1.2s, 1,234 characters)
# INFO:vision_parser:Completed processing in 15.3s
```

### **Performance Analysis**
```python
# Check processing statistics
from src.rag.ingestion.parsers import get_parser_stats

stats = get_parser_stats()
print(f"Documents processed: {stats.total_documents}")
print(f"Average processing time: {stats.avg_processing_time:.2f}s")
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Total pages processed: {stats.total_pages}")
```

## 🎯 Best Practices

### **Document Preparation**
1. **PDF Quality**: Use high-resolution scans (300+ DPI)
2. **File Size**: Optimize large files before processing
3. **Password Protection**: Remove password protection
4. **Orientation**: Ensure correct page orientation

### **Configuration Optimization**
1. **Start Conservative**: Begin with `max_concurrent_pages: 3`
2. **Monitor Memory**: Watch system memory usage
3. **Adjust Gradually**: Increase concurrency if memory allows
4. **Test Different Documents**: Verify quality with various document types

### **Quality Assurance**
1. **Sample Testing**: Test with representative documents
2. **Accuracy Verification**: Manually check extracted text quality
3. **Error Monitoring**: Track processing failures
4. **Performance Tracking**: Monitor processing times

## 🔧 Custom Parser Development

### **Creating a Custom Parser**
```python
from src.rag.ingestion.parsers.base_parser import BaseParser
from src.rag.shared.models.document import Document

class CustomParser(BaseParser):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_setting = config.get("custom_setting", "default")
    
    async def _parse_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        # Implement custom parsing logic
        content = self._extract_content(file_path)
        
        return [Document(
            content=content,
            metadata={
                **metadata,
                "parser": "custom",
                "extraction_method": "custom_logic"
            }
        )]
    
    def _extract_content(self, file_path: str) -> str:
        # Custom extraction logic
        pass
```

### **Registering Custom Parser**
```python
# In parser factory
from src.rag.ingestion.parsers.parser_factory import ParserFactory

ParserFactory.register_parser("custom", CustomParser)

# Use in configuration
ingestion:
  parsing:
    provider: "custom"
    custom:
      custom_setting: "value"
```

## 📚 Related Documentation

- **[Text Chunkers](./chunkers.md)** - Next step: chunk extracted text
- **[Embedding Models](./embedders.md)** - Convert chunks to vectors
- **[Ingestion API](./api.md)** - REST API for document processing
- **[Configuration Guide](../../configuration.md)** - Complete configuration reference

## 🚀 Quick Examples

### **Process a PDF with Vision Parser**
```bash
# Upload and process a PDF
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@document.pdf" \
  -F "metadata={\"source\": \"manual\", \"parser\": \"vision\"}"
```

### **Process Text Files**
```bash
# Upload and process a text file
curl -X POST "http://localhost:8000/ingest/file" \
  -H "X-API-Key: test-api-key" \
  -F "file=@document.txt" \
  -F "metadata={\"source\": \"text\", \"parser\": \"simple\"}"
```

### **Check Processing Status**
```bash
# Monitor processing progress
curl -X GET "http://localhost:8000/ingest/status/job_123" \
  -H "X-API-Key: test-api-key"
```

---

**Next Steps**: 
- [Configure Text Chunking](./chunkers.md)
- [Set up Embedding Generation](./embedders.md)
- [Use the Ingestion API](./api.md)
