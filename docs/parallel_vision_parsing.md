# Parallel Vision Parsing

This document describes the parallel processing capabilities of all vision parsers in the RAG system, which allows processing multiple PDF pages simultaneously for improved performance.

## Supported Parsers

Parallel processing has been implemented for all vision-based parsers:

- **VisionParser**: Uses Vertex AI Gemini Vision models
- **GroqVisionParser**: Uses Groq vision models (llama-3.2-11b-vision-preview)
- **OpenAIVisionParser**: Uses OpenAI-compatible vision models (GPT-4o)

## Overview

The VisionParser has been enhanced to process multiple pages of PDF documents in parallel, significantly reducing processing time for multi-page documents. The implementation uses `asyncio.gather()` with semaphore-based concurrency control to manage resource usage and API rate limits.

## Key Features

### 🚀 Parallel Processing
- **Concurrent Page Processing**: Multiple pages processed simultaneously
- **Configurable Concurrency**: Control the number of parallel operations
- **Semaphore-Based Control**: Prevents overwhelming the API or system resources
- **Order Preservation**: Results maintain the original page order

### 🛡️ Robust Error Handling
- **Individual Page Failures**: One page failure doesn't stop the entire process
- **Fallback Mechanisms**: Automatic fallback to text extraction if vision fails
- **Sequential Fallback**: Falls back to sequential processing if parallel fails
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

### ⚙️ Configurable Parameters
- **max_concurrent_pages**: Number of pages to process in parallel (default: 5)
- **max_pages**: Maximum number of pages to process (default: 100)
- **model**: Vision model to use (default: "gemini-1.5-pro-002")

## Configuration

### Basic Configuration

```yaml
# In your config.yaml
ingestion:
  parsing:
    default_parser: vision_parser  # or groq_vision_parser, openai_vision_parser
    vision:
      enabled: true
      model: gemini-1.5-pro-002
      max_pages: 100
      max_concurrent_pages: 5  # Parallel processing setting
```

### Parser-Specific Configurations

#### VisionParser (Vertex AI)
```yaml
parser:
  provider: "vision_parser"
  config:
    model: "gemini-1.5-pro-002"
    max_pages: 100
    max_concurrent_pages: 5
```

#### GroqVisionParser
```yaml
parser:
  provider: "groq_vision_parser"
  config:
    model_name: "llama-3.2-11b-vision-preview"
    prompt_template: "Extract and structure the text content from this document."
    max_pages: 50
    max_concurrent_pages: 5
```

#### OpenAIVisionParser
```yaml
parser:
  provider: "openai_vision_parser"
  config:
    model: "gpt-4o"
    api_base: "https://api.openai.com/v1"
    max_pages: 100
    max_concurrent_pages: 5
```

### Parser-Specific Configuration

```python
# VisionParser (Vertex AI)
vertex_parser = VisionParser({
    "model": "gemini-1.5-pro-002",
    "max_pages": 50,
    "max_concurrent_pages": 8
})

# GroqVisionParser
groq_parser = GroqVisionParser({
    "model_name": "llama-3.2-11b-vision-preview",
    "prompt_template": "Extract and structure the text content from this document.",
    "max_pages": 50,
    "max_concurrent_pages": 8
})

# OpenAIVisionParser
openai_parser = OpenAIVisionParser({
    "model": "gpt-4o",
    "api_base": "https://api.openai.com/v1",
    "max_pages": 50,
    "max_concurrent_pages": 8
})
```

## Performance Guidelines

### Concurrency Recommendations

| Document Size | Recommended max_concurrent_pages | Reasoning |
|---------------|----------------------------------|-----------|
| 1-5 pages     | 2-3                             | Small overhead, moderate parallelism |
| 6-20 pages    | 5-8                             | Good balance of speed and resource usage |
| 20+ pages     | 8-12                            | Maximum parallelism for large documents |

### Factors to Consider

1. **API Rate Limits**: Vision APIs have rate limits that may restrict concurrency
2. **Memory Usage**: Each concurrent page uses memory for image processing
3. **Network Bandwidth**: Multiple simultaneous API calls require bandwidth
4. **System Resources**: CPU and I/O capacity for image conversion

## Implementation Details

### Architecture

```
PDF Document
    ↓
Page Extraction (Sequential)
    ↓
Parallel Processing Pipeline:
┌─────────────────────────────────────┐
│  Page 1 → Image → Vision API → Text │
│  Page 2 → Image → Vision API → Text │
│  Page 3 → Image → Vision API → Text │
│  ...                                │
│  Page N → Image → Vision API → Text │
└─────────────────────────────────────┘
    ↓
Result Aggregation (Order Preserved)
    ↓
Combined Document
```

### Key Components

1. **_process_single_page()**: Processes individual pages with semaphore control
2. **_convert_page_to_base64()**: Thread-pool based image conversion
3. **_process_pages_sequentially()**: Fallback for sequential processing
4. **asyncio.gather()**: Orchestrates parallel execution

### Error Handling Strategy

```python
# Parallel processing with error handling
try:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for page_num, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle individual page failure
            fallback_to_text_extraction(page_num)
        else:
            # Use successful vision extraction
            process_result(result)
            
except Exception as e:
    # Fall back to sequential processing
    logger.info("Falling back to sequential processing...")
    results = await self._process_pages_sequentially(...)
```

## Usage Examples

### Usage Examples

```python
import asyncio
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.parsers.groq_vision_parser import GroqVisionParser
from src.rag.ingestion.parsers.openai_vision_parser import OpenAIVisionParser

async def parse_with_vertex_ai():
    """Parse using Vertex AI Gemini Vision."""
    parser = VisionParser({
        "model": "gemini-1.5-pro-002",
        "max_pages": 20,
        "max_concurrent_pages": 6
    })
    documents = await parser._parse_file("document.pdf", {})
    return documents

async def parse_with_groq():
    """Parse using Groq Vision."""
    parser = GroqVisionParser({
        "model_name": "llama-3.2-11b-vision-preview",
        "prompt_template": "Extract and structure the text content from this document.",
        "max_pages": 20,
        "max_concurrent_pages": 6
    })
    documents = await parser._parse_file("document.pdf", {})
    return documents

async def parse_with_openai():
    """Parse using OpenAI Vision."""
    parser = OpenAIVisionParser({
        "model": "gpt-4o",
        "max_pages": 20,
        "max_concurrent_pages": 6
    })
    documents = await parser._parse_file("document.pdf", {})
    return documents

# Run any parser
results = asyncio.run(parse_with_vertex_ai())
# or
results = asyncio.run(parse_with_groq())
# or
results = asyncio.run(parse_with_openai())
```

### Performance Testing

```python
import time
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.parsers.groq_vision_parser import GroqVisionParser
from src.rag.ingestion.parsers.openai_vision_parser import OpenAIVisionParser

async def compare_parser_performance():
    """Compare performance across different parsers and concurrency levels."""
    
    parsers = [
        ("Vertex AI", VisionParser, {"model": "gemini-1.5-pro-002"}),
        ("Groq", GroqVisionParser, {"model_name": "llama-3.2-11b-vision-preview"}),
        ("OpenAI", OpenAIVisionParser, {"model": "gpt-4o"})
    ]
    
    concurrency_levels = [1, 3, 5]
    
    for parser_name, parser_class, base_config in parsers:
        print(f"\nTesting {parser_name}:")
        
        for concurrency in concurrency_levels:
            config = {**base_config, "max_concurrent_pages": concurrency}
            parser = parser_class(config)
            
            start_time = time.time()
            results = await parser._parse_file("document.pdf", {})
            processing_time = time.time() - start_time
            
            print(f"  Concurrency {concurrency}: {processing_time:.2f}s")
    
    return "Performance comparison completed"

# Run comprehensive test
results = asyncio.run(compare_parser_performance())
```

## Monitoring and Debugging

### Logging Output

The parallel processing includes comprehensive logging:

```
INFO - Processing 10 pages in parallel with max concurrency: 5
INFO - Successfully processed page 1 of document.pdf
INFO - Successfully processed page 3 of document.pdf
INFO - Successfully processed page 2 of document.pdf
ERROR - Vision extraction failed for page 4: API rate limit exceeded
INFO - Successfully processed page 5 of document.pdf
...
```

### Performance Metrics

Monitor these metrics to optimize performance:

- **Processing Time**: Total time to process all pages
- **Success Rate**: Percentage of pages successfully processed with vision
- **Fallback Rate**: Percentage of pages that required text extraction fallback
- **Concurrency Utilization**: How effectively the parallel slots are used

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   - **Symptom**: Frequent vision extraction failures
   - **Solution**: Reduce `max_concurrent_pages` value
   - **Example**: Change from 10 to 3-5

2. **Memory Issues**
   - **Symptom**: Out of memory errors during processing
   - **Solution**: Reduce concurrency or process in smaller batches
   - **Example**: Use `max_concurrent_pages: 2` for large documents

3. **Network Timeouts**
   - **Symptom**: Timeout errors in vision API calls
   - **Solution**: Reduce concurrency to decrease network load
   - **Example**: Set `max_concurrent_pages: 3`

4. **Authentication Failures**
   - **Symptom**: All pages fail with auth errors
   - **Solution**: Check environment variables and token refresh
   - **Example**: Verify `COIN_CONSUMER_*` variables

### Debug Configuration

For debugging, use these settings:

```yaml
# Debug configuration
parser:
  provider: "vision_parser"
  config:
    model: "gemini-1.5-pro-002"
    max_pages: 5  # Limit pages for testing
    max_concurrent_pages: 1  # Sequential for debugging
```

## Best Practices

### 1. Start Conservative
Begin with low concurrency and increase gradually:
```python
# Start with low concurrency
parser = VisionParser({"max_concurrent_pages": 2})

# Monitor performance and increase if needed
parser = VisionParser({"max_concurrent_pages": 5})
```

### 2. Monitor Resource Usage
- Watch memory consumption during processing
- Monitor API usage and rate limits
- Track processing times and success rates

### 3. Handle Failures Gracefully
- Always have fallback mechanisms
- Log failures for analysis
- Don't fail the entire document for single page issues

### 4. Optimize for Your Use Case
- **Batch Processing**: Higher concurrency for offline processing
- **Real-time Processing**: Lower concurrency for responsive systems
- **Resource-Constrained**: Conservative settings for limited resources

## Testing

Use the provided test scripts to validate parallel processing:

```bash
# Test all parallel parsers comprehensively
python examples/test_all_parallel_parsers.py

# Test only the original VisionParser
python examples/test_parallel_vision_parser.py
```

The comprehensive test script will:
- Test all three vision parsers (Vertex AI, Groq, OpenAI)
- Compare different concurrency levels (1, 2, 5)
- Measure performance improvements and speedup
- Validate authentication for each parser
- Provide detailed performance analysis

## Future Enhancements

### Planned Improvements

1. **Dynamic Concurrency**: Automatically adjust based on API response times
2. **Batch Optimization**: Intelligent batching based on page complexity
3. **Resource Monitoring**: Real-time monitoring of memory and CPU usage
4. **Advanced Fallbacks**: Multiple fallback strategies for different failure types

### Configuration Extensions

Future configuration options may include:

```yaml
parser:
  provider: "vision_parser"
  config:
    # Current options
    max_concurrent_pages: 5
    
    # Future options
    adaptive_concurrency: true
    memory_limit_mb: 1024
    timeout_seconds: 30
    retry_attempts: 3
```

## Conclusion

The parallel vision parsing feature significantly improves processing performance for multi-page documents while maintaining robustness and reliability. By carefully configuring concurrency levels and monitoring performance, you can achieve optimal processing speeds for your specific use case and infrastructure constraints.
