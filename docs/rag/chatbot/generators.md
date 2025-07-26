# Response Generators

## 🎯 Overview

Response generators are the AI models that create intelligent answers based on user queries and retrieved document context. They combine the user's question with relevant information from the knowledge base to generate accurate, contextual, and helpful responses.

## 🧠 How Generation Works

### **The Process**
1. **Context Assembly**: Combine user query with retrieved documents and conversation history
2. **Prompt Construction**: Create a structured prompt for the AI model
3. **AI Generation**: Use language model to generate response
4. **Post-Processing**: Format and validate the generated response
5. **Response Delivery**: Return formatted answer with source attribution

```python
# Example generation process
User Query: "How does machine learning work?"

Retrieved Context:
- Doc 1: "Machine learning algorithms learn patterns from data..."
- Doc 2: "Training involves feeding data to algorithms..."
- Doc 3: "Models make predictions on new, unseen data..."

Conversation History:
- Previous: "What is AI?" → "AI is artificial intelligence..."

Generated Response:
"Based on your documents, machine learning works by training algorithms 
on data to recognize patterns. The process involves feeding training data 
to algorithms, which learn to make predictions on new, unseen data. This 
builds on our previous discussion about AI..."

Sources: [Doc 1, Doc 2, Doc 3]
```

## 🏭 Available Generators

### **1. Vertex Generator (Recommended)**
Uses Google's Vertex AI Gemini models for response generation.

**Features:**
- Latest Gemini Pro models (gemini-1.5-pro-002)
- High-quality text generation
- Large context windows (up to 1M tokens)
- Multilingual support
- Enterprise-grade reliability

**Best for:**
- Production deployments
- Enterprise environments
- High-quality responses
- Large context requirements

### **2. OpenAI Generator**
Uses OpenAI's GPT models for response generation.

**Features:**
- GPT-4 and GPT-3.5 models
- Excellent reasoning capabilities
- Strong instruction following
- Good multilingual support

**Best for:**
- High accuracy requirements
- Complex reasoning tasks
- Creative responses
- Broad compatibility

## 📋 Vertex Generator Configuration

### **Basic Setup**
```yaml
chatbot:
  generation:
    provider: "vertex"
    vertex:
      model_name: "gemini-1.5-pro-002"  # Latest Gemini model
      max_tokens: 1000                  # Response length limit
      temperature: 0.7                  # Creativity (0.0-1.0)
      top_p: 0.9                       # Nucleus sampling
      top_k: 40                        # Top-k sampling
      region: "us-central1"            # GCP region
```

### **Advanced Configuration**
```yaml
chatbot:
  generation:
    provider: "vertex"
    vertex:
      model_name: "gemini-1.5-pro-002"
      max_tokens: 1000
      temperature: 0.7
      top_p: 0.9
      top_k: 40
      region: "us-central1"
      
      # System prompt configuration
      system_prompt: |
        You are a helpful AI assistant that answers questions based on 
        provided context. Always be accurate, helpful, and cite your sources.
        If you're unsure about something, say so clearly.
      
      # Response formatting
      response_format:
        include_sources: true          # Include source citations
        include_confidence: true       # Include confidence scores
        format: "markdown"             # Response format
        max_source_length: 100         # Max characters per source
      
      # Context management
      context:
        max_context_tokens: 8000       # Maximum context size
        context_truncation: "smart"    # How to handle long context
        preserve_recent: true          # Keep recent conversation
      
      # Safety and filtering
      safety:
        filter_harmful: true           # Filter harmful content
        filter_bias: true              # Reduce biased responses
        content_filter_level: "medium" # Content filtering level
      
      # Performance settings
      timeout_seconds: 60              # Generation timeout
      retry_attempts: 3                # Retry failed generations
      stream_response: false           # Enable response streaming
```

### **Domain-Specific Configuration**
```yaml
# Customer support configuration
vertex:
  model_name: "gemini-1.5-pro-002"
  temperature: 0.3                   # More conservative
  system_prompt: |
    You are a customer support assistant. Provide clear, step-by-step 
    solutions to customer problems. Always be helpful and professional.
    If you cannot solve the problem, suggest next steps.
  response_format:
    include_sources: true
    format: "structured"             # Use structured responses

# Research assistant configuration  
vertex:
  model_name: "gemini-1.5-pro-002"
  temperature: 0.7                   # More creative
  max_tokens: 1500                   # Longer responses
  system_prompt: |
    You are a research assistant. Provide comprehensive, well-researched 
    answers with detailed explanations. Include relevant examples and 
    cite all sources accurately.
```

## 🛠️ Generator Implementation

### **Using Vertex Generator**
```python
from src.rag.chatbot.generators.vertex_generator import VertexGenerator

# Initialize generator
generator = VertexGenerator({
    "model_name": "gemini-1.5-pro-002",
    "max_tokens": 1000,
    "temperature": 0.7,
    "system_prompt": "You are a helpful AI assistant."
})

# Generate response
query = "How does machine learning work?"
context_documents = [...]  # Retrieved documents
conversation_history = [...]  # Previous messages

response = await generator.generate(
    query=query,
    documents=context_documents,
    conversation_history=conversation_history,
    metadata={"user_id": "user123", "session_id": "session456"}
)

print(f"Response: {response.answer}")
print(f"Sources: {response.source_documents}")
print(f"Confidence: {response.confidence_score}")
print(f"Generation time: {response.generation_time:.2f}s")
```

### **Advanced Generation with Custom Prompts**
```python
# Custom prompt template
custom_prompt_template = """
You are an expert assistant for {domain}. 

Context from knowledge base:
{context}

Conversation history:
{history}

User question: {query}

Instructions:
1. Answer based primarily on the provided context
2. If context is insufficient, clearly state limitations
3. Provide specific examples when helpful
4. Include relevant source citations
5. Be concise but comprehensive

Response:
"""

# Generate with custom template
response = await generator.generate_with_template(
    template=custom_prompt_template,
    query=query,
    context=context_documents,
    history=conversation_history,
    domain="machine learning",
    additional_instructions="Focus on practical applications"
)
```

## 📊 Response Quality and Performance

### **Vertex AI Gemini Performance**
- **Model**: gemini-1.5-pro-002
- **Context Window**: 1M tokens
- **Response Speed**: 1-5 seconds typical
- **Quality**: 90-95% factual accuracy
- **Languages**: 100+ languages supported
- **Cost**: Optimized for enterprise use

### **Response Quality Metrics**
```python
# Quality assessment criteria
Quality Metrics:
- Factual Accuracy: 90-95%
- Relevance to Query: 85-95%
- Source Attribution: 95%+
- Coherence: 90-95%
- Helpfulness: 85-90%
- Appropriate Length: 90%+
```

### **Performance Characteristics**
```python
# Typical performance ranges
Response Time by Complexity:
- Simple queries: 1-2 seconds
- Medium queries: 2-4 seconds  
- Complex queries: 4-8 seconds
- Long context: +1-2 seconds

Token Usage:
- Input tokens: 500-8000 typical
- Output tokens: 100-1000 typical
- Context tokens: 200-5000 typical
```

## 🎨 Prompt Engineering

### **Effective Prompt Structure**
```python
# Well-structured prompt template
PROMPT_TEMPLATE = """
System: {system_prompt}

Context Information:
{context_documents}

Conversation History:
{conversation_history}

Current Question: {user_query}

Instructions:
- Answer based on the provided context
- If information is not in the context, say so clearly
- Cite specific sources for your claims
- Be helpful and accurate
- Use a {tone} tone

Response:
"""
```

### **Context Management**
```python
def prepare_context(documents, max_tokens=4000):
    """Intelligently prepare context within token limits."""
    context_parts = []
    current_tokens = 0
    
    # Sort documents by relevance score
    sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
    
    for doc in sorted_docs:
        # Estimate tokens (rough approximation)
        doc_tokens = len(doc.content.split()) * 1.3
        
        if current_tokens + doc_tokens > max_tokens:
            # Try to include partial content
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # Minimum useful content
                partial_content = truncate_smartly(doc.content, remaining_tokens)
                context_parts.append(f"Source: {doc.metadata['filename']}\n{partial_content}")
            break
        
        context_parts.append(f"Source: {doc.metadata['filename']}\n{doc.content}")
        current_tokens += doc_tokens
    
    return "\n\n".join(context_parts)

def truncate_smartly(content, max_tokens):
    """Truncate content while preserving meaning."""
    words = content.split()
    target_words = int(max_tokens / 1.3)
    
    if len(words) <= target_words:
        return content
    
    # Try to truncate at sentence boundary
    sentences = content.split('.')
    truncated = ""
    word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words > target_words:
            break
        truncated += sentence + "."
        word_count += sentence_words
    
    return truncated + "..." if truncated else " ".join(words[:target_words]) + "..."
```

### **Conversation History Integration**
```python
def format_conversation_history(history, max_turns=5):
    """Format conversation history for context."""
    if not history:
        return "No previous conversation."
    
    # Keep only recent turns
    recent_history = history[-max_turns:]
    
    formatted_turns = []
    for turn in recent_history:
        formatted_turns.append(f"User: {turn['query']}")
        formatted_turns.append(f"Assistant: {turn['response']}")
    
    return "\n".join(formatted_turns)
```

## ⚡ Performance Optimization

### **Response Caching**
```python
class CachedGenerator:
    def __init__(self, base_generator):
        self.base_generator = base_generator
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def generate(self, query, documents, **kwargs):
        # Create cache key
        cache_key = self._create_cache_key(query, documents)
        
        # Check cache
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_response
        
        # Generate new response
        response = await self.base_generator.generate(query, documents, **kwargs)
        
        # Cache response
        self.response_cache[cache_key] = (response, time.time())
        
        return response
    
    def _create_cache_key(self, query, documents):
        # Create deterministic cache key
        doc_hashes = [hash(doc.content) for doc in documents]
        return hash((query, tuple(sorted(doc_hashes))))
```

### **Streaming Responses**
```python
async def generate_streaming_response(generator, query, documents):
    """Generate response with streaming for better UX."""
    
    async for chunk in generator.generate_stream(
        query=query,
        documents=documents,
        stream=True
    ):
        # Yield partial response as it's generated
        yield {
            "type": "partial",
            "content": chunk.content,
            "is_complete": False
        }
    
    # Final response with metadata
    yield {
        "type": "complete",
        "content": chunk.content,
        "sources": chunk.source_documents,
        "confidence": chunk.confidence_score,
        "is_complete": True
    }
```

### **Batch Generation**
```python
async def batch_generate(generator, query_batches):
    """Process multiple queries efficiently."""
    
    # Prepare all prompts
    prompts = []
    for query, documents, history in query_batches:
        prompt = generator.prepare_prompt(query, documents, history)
        prompts.append(prompt)
    
    # Batch generation
    responses = await generator.batch_generate(prompts)
    
    # Process responses
    results = []
    for i, response in enumerate(responses):
        query, documents, history = query_batches[i]
        processed_response = generator.process_response(
            response, documents, query
        )
        results.append(processed_response)
    
    return results
```

## 🚨 Common Issues and Solutions

### **Poor Response Quality**
```yaml
# Issue: Responses not using context properly
# Solution: Improve prompt engineering
generation:
  vertex:
    system_prompt: |
      You MUST base your answers on the provided context documents.
      If the context doesn't contain relevant information, clearly state:
      "I don't have enough information in the provided context to answer this question."
    
    temperature: 0.3                 # More conservative
    max_tokens: 800                  # Force conciseness
```

### **Slow Response Times**
```yaml
# Issue: Generation taking too long
# Solution: Optimize for speed
generation:
  vertex:
    max_tokens: 500                  # Shorter responses
    temperature: 0.5                 # Faster generation
    timeout_seconds: 30              # Shorter timeout
    
    context:
      max_context_tokens: 4000       # Reduce context size
```

### **Inconsistent Responses**
```yaml
# Issue: Responses vary significantly for same query
# Solution: Reduce randomness
generation:
  vertex:
    temperature: 0.1                 # Very low creativity
    top_p: 0.8                       # More focused sampling
    top_k: 20                        # Limit choices
```

### **Context Overflow**
```python
# Issue: Context exceeds model limits
# Solution: Smart context truncation
def smart_context_truncation(documents, max_tokens=6000):
    # Prioritize by relevance score
    sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
    
    # Include full high-scoring documents first
    included_docs = []
    token_count = 0
    
    for doc in sorted_docs:
        doc_tokens = estimate_tokens(doc.content)
        
        if token_count + doc_tokens <= max_tokens:
            included_docs.append(doc)
            token_count += doc_tokens
        else:
            # Try to include summary of remaining docs
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 100:
                summary = summarize_document(doc.content, remaining_tokens)
                doc.content = summary
                included_docs.append(doc)
            break
    
    return included_docs
```

## 🎯 Best Practices

### **Prompt Design**
1. **Clear Instructions**: Be specific about what you want
2. **Context First**: Provide context before asking questions
3. **Examples**: Include examples of good responses
4. **Constraints**: Set clear boundaries and limitations
5. **Format Specification**: Specify desired response format

### **Quality Assurance**
```python
def validate_response_quality(response, query, documents):
    """Validate generated response quality."""
    issues = []
    
    # Check if response uses provided context
    context_text = " ".join(doc.content for doc in documents)
    if not has_context_overlap(response.answer, context_text):
        issues.append("Response doesn't appear to use provided context")
    
    # Check response length
    if len(response.answer.split()) < 10:
        issues.append("Response too short")
    elif len(response.answer.split()) > 500:
        issues.append("Response too long")
    
    # Check for source attribution
    if response.include_sources and not response.source_documents:
        issues.append("No sources provided despite configuration")
    
    # Check for harmful content
    if contains_harmful_content(response.answer):
        issues.append("Response contains potentially harmful content")
    
    return issues

def has_context_overlap(response, context, min_overlap=0.1):
    """Check if response uses context information."""
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    
    overlap = len(response_words & context_words)
    overlap_ratio = overlap / len(response_words) if response_words else 0
    
    return overlap_ratio >= min_overlap
```

### **Error Handling**
```python
class RobustGenerator:
    def __init__(self, primary_generator, fallback_generator=None):
        self.primary_generator = primary_generator
        self.fallback_generator = fallback_generator
    
    async def generate(self, query, documents, **kwargs):
        try:
            # Try primary generator
            response = await self.primary_generator.generate(
                query, documents, **kwargs
            )
            
            # Validate response
            if self._is_valid_response(response):
                return response
            else:
                raise ValueError("Invalid response generated")
        
        except Exception as e:
            logger.warning(f"Primary generation failed: {e}")
            
            if self.fallback_generator:
                try:
                    # Try fallback generator
                    return await self.fallback_generator.generate(
                        query, documents, **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {fallback_error}")
            
            # Return error response
            return self._create_error_response(query, str(e))
    
    def _is_valid_response(self, response):
        return (
            response and 
            response.answer and 
            len(response.answer.strip()) > 10 and
            not contains_harmful_content(response.answer)
        )
    
    def _create_error_response(self, query, error_msg):
        return GenerationResponse(
            answer="I apologize, but I'm unable to generate a response at this time. Please try again later.",
            source_documents=[],
            confidence_score=0.0,
            error=error_msg
        )
```

## 🔧 Custom Generator Development

### **Creating a Custom Generator**
```python
from src.rag.chatbot.generators.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_model = self._initialize_custom_model(config)
        self.custom_prompt_template = config.get("prompt_template")
    
    def _initialize_custom_model(self, config):
        # Initialize your custom AI model
        pass
    
    async def generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: List[Dict] = None,
        metadata: Dict[str, Any] = None
    ) -> GenerationResponse:
        # Prepare context
        context = self._prepare_context(documents)
        history = self._format_history(conversation_history)
        
        # Create prompt
        prompt = self._create_prompt(query, context, history)
        
        # Generate response
        raw_response = await self.custom_model.generate(prompt)
        
        # Process response
        processed_response = self._process_response(
            raw_response, documents, query
        )
        
        return processed_response
    
    def _create_prompt(self, query, context, history):
        # Custom prompt creation logic
        return self.custom_prompt_template.format(
            query=query,
            context=context,
            history=history,
            system_instructions=self.system_prompt
        )
    
    def _process_response(self, raw_response, documents, query):
        # Custom response processing
        return GenerationResponse(
            answer=raw_response.text,
            source_documents=documents,
            confidence_score=raw_response.confidence,
            generation_time=raw_response.time,
            metadata={"model": "custom", "query": query}
        )
```

### **Registering Custom Generator**
```python
# In generator factory
from src.rag.chatbot.generators.generator_factory import GeneratorFactory

GeneratorFactory.register_generator("custom", CustomGenerator)

# Use in configuration
chatbot:
  generation:
    provider: "custom"
    custom:
      model_name: "custom-model-v1"
      prompt_template: "Custom prompt: {query}\nContext: {context}\nResponse:"
```

## 📚 Related Documentation

- **[Document Retrievers](./retrievers.md)** - Retrieve context for generation
- **[Result Rerankers](./rerankers.md)** - Improve context quality
- **[Memory Systems](./memory.md)** - Integrate conversation history
- **[Model Providers](../../models/generation.md)** - Detailed model information

## 🚀 Quick Examples

### **Basic Response Generation**
```python
# Simple generation example
generator = VertexGenerator({
    "model_name": "gemini-1.5-pro-002",
    "temperature": 0.7
})

response = await generator.generate(
    query="What is machine learning?",
    documents=retrieved_documents,
    conversation_history=[]
)

print(f"Answer: {response.answer}")
print(f"Sources: {[doc.metadata['filename'] for doc in response.source_documents]}")
```

### **API Usage**
```bash
# Generate response via API
curl -X POST "http://localhost:8001/chat/message/json" \
  -H "X-API-Key: test-api-key" \
  -H "soeid: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I troubleshoot network issues?",
    "use_retrieval": true,
    "use_history": true
  }'
```

---

**Next Steps**: 
- [Configure Memory Systems](./memory.md)
- [Set up Workflow Management](./workflow.md)
- [Use the Chatbot API](./api.md)
