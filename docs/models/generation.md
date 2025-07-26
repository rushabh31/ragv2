# Generation Models

Generation models are responsible for text generation, chat completion, and conversational AI capabilities in the RAG system. The system supports multiple providers through a unified factory pattern with consistent APIs and universal authentication.

## Overview

The generation model system provides:

- **Multi-provider support**: Vertex AI, OpenAI, Anthropic, Azure OpenAI, Groq
- **Universal authentication**: Consistent auth across all providers
- **Factory pattern**: Easy model instantiation and switching
- **RAG integration**: Seamless integration with retrieval and reranking
- **Streaming support**: Real-time response streaming
- **Caching**: Response caching for performance optimization

## Supported Providers

### Vertex AI (Google Cloud)

**Provider**: `vertex`  
**Models**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro  
**Authentication**: Universal auth with OAuth2 tokens

```python
from src.models.generation import VertexGenAI

# Create model instance
model = VertexGenAI(
    model_name="gemini-1.5-pro-002",
    max_tokens=8192,
    temperature=0.7
)

# Generate content
response = await model.generate_content("Explain quantum computing")

# Chat completion
messages = [
    {"role": "user", "content": "What is machine learning?"}
]
response = await model.chat_completion(messages)
```

**Configuration**:
```yaml
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    max_tokens: 8192
    temperature: 0.7
    top_p: 0.95
    region: "us-central1"
```

### Anthropic on Vertex AI

**Provider**: `anthropic_vertex`  
**Models**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus  
**Authentication**: Universal auth with OAuth2 tokens

```python
from src.models.generation import AnthropicVertexGenAI

# Create model instance
model = AnthropicVertexGenAI(
    model_name="claude-3-5-sonnet@20240229",
    max_tokens=4096,
    temperature=0.7
)

# Generate content
response = await model.generate_content("Write a Python function to sort a list")

# Chat completion with system message
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Help me debug this code"}
]
response = await model.chat_completion(messages)
```

**Configuration**:
```yaml
generation:
  provider: "anthropic_vertex"
  config:
    model_name: "claude-3-5-sonnet@20240229"
    max_tokens: 4096
    temperature: 0.7
    region: "us-east5"
```

### OpenAI

**Provider**: `openai`  
**Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, Meta-Llama models  
**Authentication**: Universal auth with API key headers

```python
from src.models.generation import OpenAIGenAI

# Create model instance
model = OpenAIGenAI(
    model_name="gpt-4",
    max_tokens=4096,
    temperature=0.7
)

# Generate content
response = await model.generate_content("Explain neural networks")

# Chat completion with conversation history
messages = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "Tell me about AI"}
]
response = await model.chat_completion(messages)
```

**Configuration**:
```yaml
generation:
  provider: "openai"
  config:
    model_name: "gpt-4"
    max_tokens: 4096
    temperature: 0.7
    api_base: "https://api.openai.com/v1"
```

### Azure OpenAI

**Provider**: `azure_openai`  
**Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo  
**Authentication**: Universal auth with Azure AD tokens

```python
from src.models.generation import AzureOpenAIGenAI

# Create model instance
model = AzureOpenAIGenAI(
    model_name="GPT4-o",
    max_tokens=4096,
    temperature=0.7
)

# Generate content
response = await model.generate_content("Summarize this document")

# Streaming chat completion
messages = [{"role": "user", "content": "Tell me a story"}]
async for chunk in model.chat_completion_stream(messages):
    print(chunk, end="")
```

**Configuration**:
```yaml
generation:
  provider: "azure_openai"
  config:
    model_name: "GPT4-o"
    max_tokens: 4096
    temperature: 0.7
    api_version: "2024-02-15-preview"
    azure_endpoint: "https://your-resource.openai.azure.com/"
```

### Groq

**Provider**: `groq`  
**Models**: Llama 3.1, Llama 3, Mixtral, Gemma  
**Authentication**: API key-based

```python
from src.models.generation import GroqGenAI

# Create model instance
model = GroqGenAI(
    model_name="llama-3.1-70b-versatile",
    max_tokens=4096,
    temperature=0.7
)

# Fast inference
response = await model.generate_content("Quick answer: What is Python?")

# Chat completion
messages = [{"role": "user", "content": "Explain recursion"}]
response = await model.chat_completion(messages)
```

**Configuration**:
```yaml
generation:
  provider: "groq"
  config:
    model_name: "llama-3.1-70b-versatile"
    max_tokens: 4096
    temperature: 0.7
    api_key: "${GROQ_API_KEY}"
```

## Factory Usage

### Creating Models

Use the `GenerationModelFactory` to create model instances:

```python
from src.models.generation import GenerationModelFactory

# Create model using factory
model = GenerationModelFactory.create_model(
    provider="vertex",
    model_name="gemini-1.5-pro-002",
    max_tokens=8192,
    temperature=0.7
)

# List available providers
providers = GenerationModelFactory.list_providers()
print(f"Available providers: {providers}")

# Check if provider is supported
is_supported = GenerationModelFactory.is_provider_supported("vertex")
```

### Configuration-Based Creation

Create models from configuration files:

```python
import yaml
from src.models.generation import GenerationModelFactory

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create model from config
generation_config = config["generation"]
model = GenerationModelFactory.create_model(
    provider=generation_config["provider"],
    **generation_config["config"]
)
```

## API Reference

### Common Methods

All generation models implement these methods:

#### `generate_content(prompt: str) -> str`

Generate text content from a prompt:

```python
response = await model.generate_content("Write a haiku about coding")
```

#### `chat_completion(messages: List[Dict]) -> str`

Complete a chat conversation:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
response = await model.chat_completion(messages)
```

#### `get_coin_token() -> str`

Get authentication token (universal auth):

```python
token = await model.get_coin_token()
```

#### `validate_authentication() -> bool`

Validate authentication status:

```python
is_valid = await model.validate_authentication()
```

#### `get_auth_health_status() -> Dict`

Get authentication health status:

```python
health = await model.get_auth_health_status()
print(f"Auth status: {health['status']}")
```

### Advanced Methods

#### Streaming (where supported)

```python
# Stream chat completion
async for chunk in model.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

#### Batch Processing

```python
# Process multiple prompts
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
responses = await model.batch_generate(prompts)
```

#### Function Calling (where supported)

```python
# Define functions
functions = [
    {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

# Chat with function calling
response = await model.chat_completion(
    messages=messages,
    functions=functions
)
```

## RAG Integration

### Generator Components

Generation models are used in RAG generator components:

```python
from src.rag.chatbot.generators.vertex_generator import VertexGenerator

# Create RAG generator
generator = VertexGenerator({
    "model_name": "gemini-1.5-pro-002",
    "max_tokens": 8192,
    "temperature": 0.7
})

# Generate response with context
response = await generator.generate_response(
    query="What is machine learning?",
    context_documents=retrieved_docs,
    conversation_history=chat_history
)
```

### Prompt Templates

Use templates for consistent prompting:

```python
# RAG prompt template
template = """
Context: {context}

Conversation History:
{history}

User Question: {query}

Please provide a helpful and accurate response based on the context provided.
"""

# Generate response
response = await model.generate_content(
    template.format(
        context="\n".join(doc.content for doc in documents),
        history="\n".join(f"{msg['role']}: {msg['content']}" for msg in history),
        query=user_query
    )
)
```

### Custom Generators

Create custom generators for specific use cases:

```python
from src.rag.chatbot.generators.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = GenerationModelFactory.create_model(
            provider=config["provider"],
            **config["config"]
        )
    
    async def generate_response(
        self,
        query: str,
        context_documents: List[Document],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        # Custom generation logic
        prompt = self._build_custom_prompt(query, context_documents, conversation_history)
        return await self.model.generate_content(prompt)
    
    def _build_custom_prompt(self, query, documents, history):
        # Custom prompt building logic
        pass
```

## Performance Optimization

### Caching

Enable response caching for improved performance:

```python
# Configuration with caching
generation:
  provider: "vertex"
  config:
    model_name: "gemini-1.5-pro-002"
    enable_caching: true
    cache_ttl: 3600  # 1 hour
    cache_max_size: 1000  # Max cached responses
```

### Batch Processing

Process multiple requests efficiently:

```python
# Batch generation
prompts = [
    "Explain AI",
    "What is ML?",
    "Define neural networks"
]

# Process in batches
batch_size = 5
responses = []
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    batch_responses = await model.batch_generate(batch)
    responses.extend(batch_responses)
```

### Parallel Processing

Use async processing for concurrent requests:

```python
import asyncio

async def process_queries(queries):
    tasks = []
    for query in queries:
        task = model.generate_content(query)
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    return responses
```

### Token Management

Monitor and optimize token usage:

```python
# Token counting (approximate)
def estimate_tokens(text: str) -> int:
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4

# Optimize prompts
def optimize_prompt(prompt: str, max_tokens: int) -> str:
    estimated_tokens = estimate_tokens(prompt)
    if estimated_tokens > max_tokens:
        # Truncate or summarize prompt
        words = prompt.split()
        target_words = len(words) * max_tokens // estimated_tokens
        return " ".join(words[:target_words])
    return prompt
```

## Error Handling

### Authentication Errors

```python
from src.models.exceptions import AuthenticationError

try:
    token = await model.get_coin_token()
except AuthenticationError as e:
    logger.error(f"Authentication failed: {e}")
    # Handle authentication failure
    # - Check credentials
    # - Retry with backoff
    # - Switch to fallback provider
```

### API Errors

```python
from src.models.exceptions import APIError, RateLimitError

try:
    response = await model.generate_content(prompt)
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    # Wait and retry
    await asyncio.sleep(e.retry_after or 60)
    response = await model.generate_content(prompt)
except APIError as e:
    logger.error(f"API error: {e}")
    # Handle API failure
    # - Check API status
    # - Switch to fallback provider
    # - Return error response
```

### Timeout Handling

```python
import asyncio

async def generate_with_timeout(model, prompt, timeout=30):
    try:
        response = await asyncio.wait_for(
            model.generate_content(prompt),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.warning(f"Generation timed out after {timeout}s")
        return "I apologize, but I'm taking too long to respond. Please try again."
```

### Fallback Strategies

```python
async def generate_with_fallback(prompt: str, providers: List[str]):
    for provider in providers:
        try:
            model = GenerationModelFactory.create_model(provider, "default-model")
            response = await model.generate_content(prompt)
            logger.info(f"Successfully generated response using {provider}")
            return response
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue
    
    # All providers failed
    raise Exception("All generation providers failed")
```

## Monitoring and Logging

### Performance Metrics

Track generation performance:

```python
import time
from typing import Dict, Any

class GenerationMetrics:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0,
            "average_latency": 0
        }
    
    async def track_generation(self, model, prompt: str) -> str:
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            response = await model.generate_content(prompt)
            latency = time.time() - start_time
            
            self.metrics["successful_requests"] += 1
            self.metrics["total_latency"] += latency
            self.metrics["average_latency"] = (
                self.metrics["total_latency"] / self.metrics["successful_requests"]
            )
            
            logger.info(f"Generation completed in {latency:.2f}s")
            return response
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Generation failed: {e}")
            raise
```

### Health Monitoring

Monitor model health across providers:

```python
async def check_generation_health():
    providers = ["vertex", "openai", "anthropic_vertex"]
    health_status = {}
    
    for provider in providers:
        try:
            model = GenerationModelFactory.create_model(provider, "default-model")
            
            # Check authentication
            auth_status = await model.get_auth_health_status()
            
            # Test generation
            start_time = time.time()
            test_response = await model.generate_content("Hello")
            latency = time.time() - start_time
            
            health_status[provider] = {
                "status": "healthy",
                "auth_status": auth_status,
                "latency": latency,
                "test_response_length": len(test_response)
            }
            
        except Exception as e:
            health_status[provider] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status
```

### Usage Analytics

Track usage patterns:

```python
from collections import defaultdict
import json

class UsageAnalytics:
    def __init__(self):
        self.usage_stats = defaultdict(lambda: {
            "requests": 0,
            "tokens_used": 0,
            "average_response_time": 0,
            "error_rate": 0
        })
    
    def record_usage(self, provider: str, tokens: int, response_time: float, success: bool):
        stats = self.usage_stats[provider]
        stats["requests"] += 1
        
        if success:
            stats["tokens_used"] += tokens
            # Update average response time
            current_avg = stats["average_response_time"]
            new_avg = (current_avg * (stats["requests"] - 1) + response_time) / stats["requests"]
            stats["average_response_time"] = new_avg
        else:
            # Update error rate
            error_count = stats["requests"] * stats["error_rate"] + 1
            stats["error_rate"] = error_count / stats["requests"]
    
    def get_usage_report(self) -> str:
        return json.dumps(dict(self.usage_stats), indent=2)
```

## Best Practices

### Model Selection

1. **Vertex AI**: Best for Google Cloud environments, enterprise features
2. **OpenAI**: Cutting-edge models, broad compatibility
3. **Anthropic**: Safety-focused, constitutional AI
4. **Azure OpenAI**: Enterprise Azure integration
5. **Groq**: Fast inference, cost optimization

### Prompt Engineering

1. **Be specific**: Clear, detailed prompts get better results
2. **Use examples**: Few-shot prompting improves performance
3. **Set context**: Provide relevant background information
4. **Control output**: Specify format and length requirements

```python
# Good prompt example
prompt = """
You are an expert Python developer. Given the following code snippet, 
identify any bugs and suggest fixes.

Code:
```python
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
```

Please provide:
1. Bug identification
2. Fixed code
3. Explanation of the fix
"""
```

### Configuration Management

1. **Environment-specific configs**: Different settings for dev/staging/prod
2. **Provider fallbacks**: Configure multiple providers for reliability
3. **Resource limits**: Set appropriate token limits and timeouts
4. **Security**: Secure credential management

### Performance Optimization

1. **Caching**: Cache responses for repeated queries
2. **Batch processing**: Process multiple requests together
3. **Async processing**: Use async/await for concurrency
4. **Token optimization**: Minimize token usage where possible

## Troubleshooting

### Common Issues

**Authentication Failures**:
```bash
# Check environment variables
echo $COIN_CONSUMER_CLIENT_ID
echo $PROJECT_ID

# Test token generation
python -c "
from src.models.generation import VertexGenAI
import asyncio
async def test():
    model = VertexGenAI()
    token = await model.get_coin_token()
    print(f'Token: {token[:50]}...')
asyncio.run(test())
"
```

**Model Loading Errors**:
```python
# Check available providers
from src.models.generation import GenerationModelFactory
print(GenerationModelFactory.list_providers())

# Test model creation
try:
    model = GenerationModelFactory.create_model("vertex", "gemini-1.5-pro-002")
    print("Model created successfully")
except Exception as e:
    print(f"Model creation failed: {e}")
```

**Generation Failures**:
```python
# Test with simple prompt
try:
    response = await model.generate_content("Hello")
    print(f"Response: {response}")
except Exception as e:
    print(f"Generation failed: {e}")
    
    # Check authentication
    auth_status = await model.get_auth_health_status()
    print(f"Auth status: {auth_status}")
```

**Performance Issues**:
```python
# Enable debug logging
import logging
logging.getLogger("src.models.generation").setLevel(logging.DEBUG)

# Monitor resource usage
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"CPU: {psutil.cpu_percent()}%")
```

## Related Documentation

- [Model Factories Overview](./README.md) - Factory system overview
- [Embedding Models](./embedding.md) - Embedding model documentation
- [Vision Models](./vision.md) - Vision model capabilities
- [Chatbot Generators](../rag/chatbot/generators.md) - RAG generator components
- [Configuration Guide](../configuration.md) - Configuration reference
- [API Reference](../rag/chatbot/api.md) - REST API documentation

## Examples

For complete examples and test scripts, see:

- `examples/test_generation_models.py` - Generation model testing
- `examples/test_factory_system.py` - Factory system testing
- `examples/rag/chatbot/` - RAG integration examples
- `src/rag/chatbot/generators/` - Generator implementations
