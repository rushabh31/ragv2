# Vision Models

Vision models enable the RAG system to process and understand visual content, including documents, images, charts, and diagrams. These models are essential for document parsing, OCR, image analysis, and multimodal AI capabilities.

## Overview

The vision model system provides:

- **Multi-provider support**: Vertex AI Gemini Vision, OpenAI GPT-4 Vision, Groq Vision
- **Universal authentication**: Consistent auth across cloud providers
- **Factory pattern**: Easy model instantiation and switching
- **Document parsing**: Extract text and structure from PDFs and images
- **Parallel processing**: Efficient processing of multi-page documents
- **Structured extraction**: Extract structured data from visual content

## Supported Providers

### Vertex AI Gemini Vision (Google Cloud)

**Provider**: `vertex_ai`  
**Models**: gemini-1.5-pro-002, gemini-1.5-flash-002, gemini-1.0-pro-vision  
**Capabilities**: Document parsing, OCR, image analysis, structured extraction  
**Authentication**: Universal auth with OAuth2 tokens

```python
from src.models.vision import VertexVisionAI

# Create model instance
model = VertexVisionAI(
    model="gemini-1.5-pro-002",
    region="us-central1"
)

# Parse text from image
import base64
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()

text = await model.parse_text_from_image(
    base64_image=pdf_data,
    prompt="Extract all text content from this document"
)

# Analyze image content
analysis = await model.analyze_image(
    base64_image=pdf_data,
    prompt="Describe the structure and content of this document"
)

# Extract structured data
structured_data = await model.extract_structured_data(
    base64_image=pdf_data,
    prompt="Extract key information as JSON: title, author, main topics"
)
```

**Configuration**:
```yaml
vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"
    region: "us-central1"
    max_pages: 50
    max_concurrent_pages: 5
```

### OpenAI GPT-4 Vision

**Provider**: `openai`  
**Models**: gpt-4o, gpt-4-vision-preview, gpt-4o-mini  
**Capabilities**: Image analysis, document understanding, visual reasoning  
**Authentication**: Universal auth with API key headers

```python
from src.models.vision import OpenAIVisionAI

# Create model instance
model = OpenAIVisionAI(
    model="gpt-4o",
    max_tokens=4096
)

# Analyze image
image_analysis = await model.analyze_image(
    base64_image=base64_image,
    prompt="What do you see in this image? Describe in detail."
)

# Parse document content
document_text = await model.parse_text_from_image(
    base64_image=pdf_page,
    prompt="Extract all text from this document page, maintaining structure"
)
```

**Configuration**:
```yaml
vision:
  provider: "openai"
  config:
    model: "gpt-4o"
    max_tokens: 4096
    api_base: "https://api.openai.com/v1"
    max_pages: 100
    max_concurrent_pages: 5
```

### Groq Vision

**Provider**: `groq`  
**Models**: llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview  
**Capabilities**: Fast vision inference, document processing  
**Authentication**: API key-based

```python
from src.models.vision import GroqVisionAI

# Create model instance
model = GroqVisionAI(
    model_name="llama-3.2-11b-vision-preview",
    max_tokens=4096
)

# Fast document processing
text_content = await model.parse_text_from_image(
    base64_image=document_image,
    prompt="Extract text content from this document"
)
```

**Configuration**:
```yaml
vision:
  provider: "groq"
  config:
    model_name: "llama-3.2-11b-vision-preview"
    max_tokens: 4096
    api_key: "${GROQ_API_KEY}"
    max_concurrent_pages: 8
```

## Factory Usage

### Creating Models

Use the `VisionModelFactory` to create model instances:

```python
from src.models.vision import VisionModelFactory

# Create model using factory
model = VisionModelFactory.create_model(
    provider="vertex_ai",
    model="gemini-1.5-pro-002",
    region="us-central1"
)

# List available providers
providers = VisionModelFactory.list_providers()
print(f"Available providers: {providers}")
```

### Configuration-Based Creation

Create models from configuration files:

```python
import yaml
from src.models.vision import VisionModelFactory

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create model from config
vision_config = config["vision"]
model = VisionModelFactory.create_model(
    provider=vision_config["provider"],
    **vision_config["config"]
)
```

## API Reference

### Common Methods

All vision models implement these methods:

#### `parse_text_from_image(base64_image: str, prompt: str) -> str`

Extract text content from images or documents:

```python
text = await model.parse_text_from_image(
    base64_image=base64_encoded_image,
    prompt="Extract all text from this document, preserving formatting"
)
```

#### `analyze_image(base64_image: str, prompt: str) -> str`

Analyze and describe image content:

```python
analysis = await model.analyze_image(
    base64_image=base64_encoded_image,
    prompt="Describe the content, layout, and key elements in this image"
)
```

#### `extract_structured_data(base64_image: str, prompt: str) -> str`

Extract structured information from visual content:

```python
structured_data = await model.extract_structured_data(
    base64_image=base64_encoded_image,
    prompt="Extract the following as JSON: title, date, key points, summary"
)
```

## RAG Integration

### Vision Parser Components

Vision models are used in RAG vision parser components:

```python
from src.rag.ingestion.parsers.vision_parser import VisionParser

# Create RAG vision parser
parser = VisionParser({
    "model": "gemini-1.5-pro-002",
    "max_pages": 50,
    "max_concurrent_pages": 5
})

# Parse document
documents = await parser.parse_file(
    file_path="document.pdf",
    metadata={"source": "research_paper"}
)
```

### Custom Vision Parsers

Create custom parsers for specific document types:

```python
from src.rag.ingestion.parsers.base_parser import BaseParser
from src.models.vision import VisionModelFactory

class InvoiceParser(BaseParser):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vision_model = VisionModelFactory.create_model(
            provider=config["provider"],
            **config["config"]
        )
    
    async def parse_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        # Extract invoice data using vision model
        invoice_data = await self.vision_model.extract_structured_data(
            base64_image=pdf_page,
            prompt="""
            Extract invoice information as JSON:
            {
                "invoice_number": "",
                "date": "",
                "vendor": "",
                "total_amount": "",
                "line_items": []
            }
            """
        )
        
        return [Document(content=invoice_data, metadata=metadata)]
```

## Performance Optimization

### Parallel Processing

Optimize processing of multi-page documents:

```python
import asyncio
from typing import List

async def process_multi_page_document(
    model, 
    pdf_pages: List[str], 
    prompt: str,
    max_concurrent: int = 5
) -> List[str]:
    """Process multiple PDF pages concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_page(page_data: str) -> str:
        async with semaphore:
            return await model.parse_text_from_image(page_data, prompt)
    
    tasks = [process_page(page) for page in pdf_pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [result for result in results if isinstance(result, str)]
```

### Caching

Implement vision processing caching:

```python
import hashlib
from typing import Dict, Optional

class VisionCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
    
    def _get_cache_key(self, image_data: str, prompt: str, model_name: str) -> str:
        content = f"{model_name}:{prompt}:{image_data[:1000]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, image_data: str, prompt: str, model_name: str) -> Optional[str]:
        key = self._get_cache_key(image_data, prompt, model_name)
        return self.cache.get(key)
    
    def set(self, image_data: str, prompt: str, model_name: str, result: str):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_cache_key(image_data, prompt, model_name)
        self.cache[key] = result
```

## Error Handling

### Authentication Errors

```python
from src.models.exceptions import AuthenticationError

try:
    result = await model.parse_text_from_image(image_data, prompt)
except AuthenticationError as e:
    logger.error(f"Authentication failed: {e}")
    # Handle authentication failure
```

### Fallback Strategies

```python
async def vision_processing_with_fallback(
    image_data: str, 
    prompt: str, 
    providers: List[str]
):
    """Try multiple vision providers with fallback."""
    for provider in providers:
        try:
            model = VisionModelFactory.create_model(provider, "default-model")
            result = await model.parse_text_from_image(image_data, prompt)
            return result
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue
    
    # All providers failed, try OCR fallback
    try:
        import pytesseract
        from PIL import Image
        import io
        import base64
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(image)
        
        return ocr_text
    except Exception as e:
        raise Exception("All vision processing methods failed")
```

## Best Practices

### Model Selection

1. **Vertex AI Gemini**: Best for Google Cloud, high-quality document parsing
2. **OpenAI GPT-4 Vision**: Excellent image understanding and reasoning
3. **Groq Vision**: Fast inference for real-time applications

### Prompt Engineering

```python
# Good prompt examples
DOCUMENT_PARSING_PROMPT = """
Extract all text content from this document page while preserving:
1. Paragraph structure
2. Bullet points and lists
3. Headers and subheaders
4. Table content (if any)

Format the output as clean, readable text.
"""

STRUCTURED_EXTRACTION_PROMPT = """
Extract the following information from this invoice as JSON:
{
    "invoice_number": "string",
    "date": "YYYY-MM-DD",
    "vendor_name": "string",
    "total_amount": "number"
}

If any field is not found, use null as the value.
"""
```

### Image Quality

1. **Resolution**: Use at least 150 DPI for text documents
2. **Format**: PNG or JPEG for best compatibility
3. **Size**: Balance quality vs. processing time
4. **Preprocessing**: Consider image enhancement for poor quality scans

## Troubleshooting

### Common Issues

**Authentication Failures**:
```bash
# Check environment variables
echo $COIN_CONSUMER_CLIENT_ID
echo $PROJECT_ID

# Test authentication
python -c "
from src.models.vision import VertexVisionAI
import asyncio
async def test():
    model = VertexVisionAI()
    token = await model.get_coin_token()
    print(f'Token: {token[:50]}...')
asyncio.run(test())
"
```

**Image Format Issues**:
```python
import base64
from PIL import Image
import io

def debug_image_format(base64_image: str):
    try:
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Format: {image.format}, Mode: {image.mode}, Size: {image.size}")
        return True
    except Exception as e:
        print(f"Image validation failed: {e}")
        return False
```

**Performance Issues**:
```python
import time
import psutil

async def monitor_vision_performance(model, image_data: str, prompt: str):
    print(f"Memory: {psutil.virtual_memory().percent}%")
    print(f"CPU: {psutil.cpu_percent()}%")
    
    start_time = time.time()
    result = await model.parse_text_from_image(image_data, prompt)
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Result length: {len(result)} characters")
```

## Related Documentation

- [Model Factories Overview](./README.md) - Factory system overview
- [Generation Models](./generation.md) - Generation model documentation
- [Embedding Models](./embedding.md) - Embedding model capabilities
- [Vision Parsers](../rag/ingestion/parsers.md) - Parser components
- [Configuration Guide](../configuration.md) - Configuration reference

## Examples

For complete examples and test scripts, see:

- `examples/test_vision_models.py` - Vision model testing
- `examples/test_parallel_parsers.py` - Parallel processing examples
- `examples/rag/ingestion/` - Ingestion pipeline examples
- `src/rag/ingestion/parsers/` - Parser implementations
