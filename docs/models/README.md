# Model Factories

The Model Factories system provides a unified, multi-provider architecture for AI models used throughout the RAG system. This factory-based approach enables seamless switching between different AI providers while maintaining consistent interfaces and universal authentication.

## Overview

The model factories support three main categories of AI models:

- **Generation Models**: Text generation and chat completion models
- **Embedding Models**: Text embedding and vector representation models  
- **Vision Models**: Image analysis and document parsing models

Each category supports multiple providers with a consistent API and configuration structure.

## Architecture

### Factory Pattern

The system uses the factory pattern to create model instances:

```python
from src.models.generation import GenerationModelFactory
from src.models.embedding import EmbeddingModelFactory
from src.models.vision import VisionModelFactory

# Create models using factories
generation_model = GenerationModelFactory.create_model("vertex", "gemini-1.5-pro-002")
embedding_model = EmbeddingModelFactory.create_model("vertex_ai", "text-embedding-004")
vision_model = VisionModelFactory.create_model("vertex_ai", "gemini-1.5-pro-002")
```

### Universal Authentication

All models use a unified authentication system through the `get_coin_token()` method:

```python
# Universal authentication across all providers
token = await model.get_coin_token()
health_status = await model.get_auth_health_status()
is_valid = await model.validate_authentication()
```

### Configuration-Driven

Models are configured through YAML files with provider-specific sections:

```yaml
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    max_tokens: 8192
    temperature: 0.7

embedding:
  provider: "vertex_ai"
  config:
    model: "text-embedding-004"
    dimensions: 768

vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"
    region: "us-central1"
```

## Supported Providers

### Generation Models

| Provider | Models | Authentication | Use Case |
|----------|--------|---------------|----------|
| `vertex` | Gemini 1.5 Pro/Flash | Universal Auth | Google Cloud |
| `anthropic_vertex` | Claude 3.5 Sonnet | Universal Auth | Anthropic on Vertex |
| `openai` | GPT-4, GPT-3.5 | Universal Auth | OpenAI API |
| `azure_openai` | GPT-4, GPT-3.5 | Universal Auth | Azure OpenAI |
| `groq` | Llama, Mixtral | API Key | Fast inference |

### Embedding Models

| Provider | Models | Dimensions | Use Case |
|----------|--------|------------|----------|
| `vertex_ai` | text-embedding-004 | 768 | Google Cloud |
| `openai_universal` | text-embedding-3-large | 3072 | OpenAI API |
| `azure_openai` | text-embedding-ada-002 | 1536 | Azure OpenAI |
| `sentence_transformer` | all-mpnet-base-v2 | 768 | Local/offline |

### Vision Models

| Provider | Models | Capabilities | Use Case |
|----------|--------|--------------|----------|
| `vertex_ai` | Gemini 1.5 Pro Vision | Document parsing, OCR | Google Cloud |
| `openai` | GPT-4 Vision | Image analysis | OpenAI API |
| `groq` | Llama 3.2 Vision | Fast vision inference | Groq API |

## Usage Examples

### Generation Models

```python
from src.models.generation import GenerationModelFactory

# Create a Vertex AI model
model = GenerationModelFactory.create_model(
    provider="vertex",
    model_name="gemini-1.5-pro-002",
    max_tokens=4096,
    temperature=0.7
)

# Generate content
response = await model.generate_content("Explain quantum computing")

# Chat completion
messages = [{"role": "user", "content": "Hello!"}]
response = await model.chat_completion(messages)
```

### Embedding Models

```python
from src.models.embedding import EmbeddingModelFactory

# Create an embedding model
model = EmbeddingModelFactory.create_model(
    provider="vertex_ai",
    model="text-embedding-004"
)

# Get embeddings
texts = ["Hello world", "Machine learning"]
embeddings = await model.get_embeddings(texts)

# Get single embedding
embedding = await model.get_embedding("Single text")
```

### Vision Models

```python
from src.models.vision import VisionModelFactory

# Create a vision model
model = VisionModelFactory.create_model(
    provider="vertex_ai",
    model="gemini-1.5-pro-002"
)

# Parse text from image
text = await model.parse_text_from_image(
    base64_image, 
    "Extract all text from this document"
)

# Analyze image
analysis = await model.analyze_image(
    base64_image,
    "Describe what you see in this image"
)
```

## Configuration

### Environment Variables

Set up authentication credentials:

```bash
# Universal authentication
export COIN_CONSUMER_ENDPOINT_URL="https://oauth-server/oauth2/token"
export COIN_CONSUMER_CLIENT_ID="your-client-id"
export COIN_CONSUMER_CLIENT_SECRET="your-client-secret"
export COIN_CONSUMER_SCOPE="https://www.googleapis.com/auth/cloud-platform"

# Project configuration
export PROJECT_ID="your-gcp-project-id"
export VERTEXAI_API_ENDPOINT="us-central1-aiplatform.googleapis.com"

# Provider-specific (if needed)
export GROQ_API_KEY="your-groq-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### YAML Configuration

Configure models in your `config.yaml`:

```yaml
# Generation configuration
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    max_tokens: 8192
    temperature: 0.7
    top_p: 0.95

# Embedding configuration
embedding:
  provider: "vertex_ai"
  config:
    model: "text-embedding-004"
    dimensions: 768
    batch_size: 100

# Vision configuration
vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"
    region: "us-central1"
    max_pages: 50
```

## RAG System Integration

### Ingestion Pipeline

Models are automatically used in the ingestion pipeline:

```python
# Vision parsing (uses vision.provider from config)
from src.rag.ingestion.parsers.vision_parser import VisionParser
parser = VisionParser(config["parser"])

# Embedding generation (uses embedding.provider from config)
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder
embedder = VertexEmbedder(config["embedder"])
```

### Chatbot Service

Models are used in the chatbot for generation:

```python
# Response generation (uses generation.provider from config)
from src.rag.chatbot.generators.vertex_generator import VertexGenerator
generator = VertexGenerator(config["generator"])
```

## Performance and Optimization

### Caching

Models support response caching to improve performance:

```python
# Enable caching in configuration
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    enable_caching: true
    cache_ttl: 3600  # 1 hour
```

### Batch Processing

Embedding models support batch processing:

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await model.get_embeddings(texts, batch_size=10)
```

### Parallel Processing

Vision models support parallel page processing:

```python
# Configure parallel processing
vision:
  provider: "vertex_ai"
  config:
    model: "gemini-1.5-pro-002"
    max_concurrent_pages: 5  # Process 5 pages in parallel
```

## Error Handling

### Authentication Errors

```python
try:
    token = await model.get_coin_token()
except AuthenticationError as e:
    logger.error(f"Authentication failed: {e}")
    # Handle authentication failure
```

### API Errors

```python
try:
    response = await model.generate_content(prompt)
except APIError as e:
    logger.error(f"API call failed: {e}")
    # Handle API failure
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    # Handle rate limiting
```

### Fallback Strategies

```python
# Try multiple providers
providers = ["vertex", "openai", "anthropic_vertex"]
for provider in providers:
    try:
        model = GenerationModelFactory.create_model(provider, model_name)
        response = await model.generate_content(prompt)
        break
    except Exception as e:
        logger.warning(f"Provider {provider} failed: {e}")
        continue
```

## Monitoring and Health Checks

### Health Status

Check model health across all providers:

```python
async def check_model_health():
    providers = ["vertex", "openai", "anthropic_vertex"]
    health_status = {}
    
    for provider in providers:
        try:
            model = GenerationModelFactory.create_model(provider, "default-model")
            status = await model.get_auth_health_status()
            health_status[provider] = status
        except Exception as e:
            health_status[provider] = {"status": "unhealthy", "error": str(e)}
    
    return health_status
```

### Performance Metrics

Track model performance:

```python
import time

async def track_model_performance(model, prompt):
    start_time = time.time()
    try:
        response = await model.generate_content(prompt)
        latency = time.time() - start_time
        
        # Log metrics
        logger.info(f"Model response time: {latency:.2f}s")
        logger.info(f"Response length: {len(response)} characters")
        
        return response
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Model failed after {latency:.2f}s: {e}")
        raise
```

## Best Practices

### Model Selection

1. **Use Vertex AI** for Google Cloud environments with enterprise features
2. **Use OpenAI** for cutting-edge models and broad compatibility
3. **Use Groq** for fast inference and cost optimization
4. **Use Anthropic** for safety-focused applications

### Configuration Management

1. **Environment-specific configs**: Use different configs for dev/staging/prod
2. **Provider fallbacks**: Configure multiple providers for reliability
3. **Resource limits**: Set appropriate token limits and timeouts
4. **Caching strategies**: Enable caching for frequently used prompts

### Security

1. **Credential management**: Use secure credential storage
2. **API key rotation**: Regularly rotate API keys
3. **Access control**: Implement proper access controls
4. **Audit logging**: Log all model interactions

### Performance

1. **Batch processing**: Use batch APIs when available
2. **Parallel processing**: Process multiple requests concurrently
3. **Caching**: Cache responses for repeated queries
4. **Resource monitoring**: Monitor token usage and costs

## Troubleshooting

### Common Issues

**Authentication Failures**:
```bash
# Check environment variables
echo $COIN_CONSUMER_CLIENT_ID
echo $PROJECT_ID

# Validate token service
curl -X POST "$COIN_CONSUMER_ENDPOINT_URL" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=$COIN_CONSUMER_CLIENT_ID&client_secret=$COIN_CONSUMER_CLIENT_SECRET&scope=$COIN_CONSUMER_SCOPE"
```

**Model Loading Errors**:
```python
# Check available providers
from src.models.generation import GenerationModelFactory
print(GenerationModelFactory.list_providers())

# Validate configuration
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
    print(config["generation"])
```

**Performance Issues**:
```python
# Enable debug logging
import logging
logging.getLogger("src.models").setLevel(logging.DEBUG)

# Monitor resource usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
print(f"CPU usage: {psutil.cpu_percent()}%")
```

## Related Documentation

- [Generation Models](./generation.md) - Detailed generation model documentation
- [Embedding Models](./embedding.md) - Embedding model configuration and usage
- [Vision Models](./vision.md) - Vision model capabilities and examples
- [Configuration Guide](../configuration.md) - Complete configuration reference
- [Development Guide](../development/README.md) - Development and testing
- [API Reference](../rag/chatbot/api.md) - REST API documentation

## Examples

For complete examples and test scripts, see:

- `examples/test_generation_models.py` - Generation model examples
- `examples/test_embedding_models.py` - Embedding model examples
- `examples/test_vision_models.py` - Vision model examples
- `examples/test_factory_system.py` - Complete factory system test
