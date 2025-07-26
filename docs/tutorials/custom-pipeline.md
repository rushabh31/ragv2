# Custom Pipeline Tutorial

Learn how to build custom RAG pipelines tailored to your specific use cases, including custom parsers, embedders, retrievers, and generators.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Custom Document Parser](#custom-document-parser)
- [Custom Embedder](#custom-embedder)
- [Custom Retriever](#custom-retriever)
- [Custom Generator](#custom-generator)
- [Pipeline Integration](#pipeline-integration)
- [Configuration Management](#configuration-management)
- [Testing Custom Components](#testing-custom-components)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

This tutorial demonstrates how to extend the RAG system with custom components for specialized use cases:

### What You'll Learn
- How to create custom document parsers for specific formats
- Building custom embedding models and strategies
- Implementing custom retrieval algorithms
- Creating custom response generators
- Integrating custom components into the pipeline

### Use Cases
- **Domain-Specific Documents**: Medical records, legal documents, financial reports
- **Custom Data Sources**: APIs, databases, streaming data
- **Specialized Retrieval**: Hybrid search, semantic + keyword, graph-based
- **Custom Generation**: Domain-specific prompts, multi-modal responses

## Pipeline Architecture

### RAG Pipeline Components

```python
# src/rag/pipeline/custom_pipeline.py
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class RAGPipeline:
    """Custom RAG pipeline orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parser = self._create_parser()
        self.embedder = self._create_embedder()
        self.retriever = self._create_retriever()
        self.generator = self._create_generator()
        self.memory = self._create_memory()
    
    async def ingest_document(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Custom document ingestion pipeline."""
        # Parse document
        documents = await self.parser.parse(file_path, metadata)
        
        # Generate embeddings
        embeddings = await self.embedder.embed_documents(documents)
        
        # Store in vector database
        document_id = await self.retriever.store_documents(documents, embeddings)
        
        return document_id
    
    async def query(self, query: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """Custom query processing pipeline."""
        # Retrieve relevant documents
        candidates = await self.retriever.retrieve(query, **kwargs)
        
        # Generate response
        response = await self.generator.generate(
            query=query,
            context=candidates,
            session_id=session_id,
            **kwargs
        )
        
        # Store in memory
        await self.memory.add_interaction(session_id, query, response)
        
        return response
```

## Custom Document Parser

### Creating a Custom JSON Parser

```python
# src/rag/ingestion/parsers/custom_json_parser.py
from typing import List, Dict, Any, Optional
import json
from src.rag.ingestion.parsers.base_parser import BaseParser
from src.rag.shared.models import Document

class CustomJSONParser(BaseParser):
    """Custom parser for JSON documents with specific schema."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.schema_mapping = config.get('schema_mapping', {})
        self.extract_fields = config.get('extract_fields', [])
    
    async def _parse_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Parse JSON file with custom schema."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            # Handle array of objects
            for i, item in enumerate(data):
                doc = await self._process_json_item(item, i, file_path, metadata)
                if doc:
                    documents.append(doc)
        else:
            # Handle single object
            doc = await self._process_json_item(data, 0, file_path, metadata)
            if doc:
                documents.append(doc)
        
        return documents
    
    async def _process_json_item(self, item: Dict, index: int, file_path: str, metadata: Dict) -> Optional[Document]:
        """Process individual JSON item."""
        try:
            # Extract content based on schema mapping
            content_parts = []
            
            for field in self.extract_fields:
                if field in item:
                    value = item[field]
                    if isinstance(value, (str, int, float)):
                        content_parts.append(f"{field}: {value}")
                    elif isinstance(value, dict):
                        content_parts.append(f"{field}: {json.dumps(value)}")
            
            if not content_parts:
                return None
            
            content = "\n".join(content_parts)
            
            # Create document metadata
            doc_metadata = {
                **metadata,
                'source_file': file_path,
                'item_index': index,
                'parser_type': 'custom_json',
                'original_fields': list(item.keys())
            }
            
            # Add mapped fields to metadata
            for original_field, mapped_field in self.schema_mapping.items():
                if original_field in item:
                    doc_metadata[mapped_field] = item[original_field]
            
            return Document(
                content=content,
                metadata=doc_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error processing JSON item {index}: {e}")
            return None

# Usage configuration
custom_json_config = {
    'parser_type': 'custom_json',
    'schema_mapping': {
        'id': 'document_id',
        'title': 'document_title',
        'category': 'document_category'
    },
    'extract_fields': ['title', 'description', 'content', 'tags']
}
```

### Database Record Parser

```python
# src/rag/ingestion/parsers/database_parser.py
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional

class DatabaseParser(BaseParser):
    """Custom parser for database records."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_string = config['connection_string']
        self.query = config['query']
        self.content_fields = config.get('content_fields', [])
        self.metadata_fields = config.get('metadata_fields', [])
    
    async def _parse_file(self, query_params: str, metadata: Dict[str, Any]) -> List[Document]:
        """Parse database records as documents."""
        documents = []
        
        conn = await asyncpg.connect(self.connection_string)
        try:
            # Execute query with parameters
            rows = await conn.fetch(self.query, *query_params.split(','))
            
            for i, row in enumerate(rows):
                doc = await self._process_database_row(dict(row), i, metadata)
                if doc:
                    documents.append(doc)
        
        finally:
            await conn.close()
        
        return documents
    
    async def _process_database_row(self, row: Dict, index: int, metadata: Dict) -> Optional[Document]:
        """Process individual database row."""
        try:
            # Build content from specified fields
            content_parts = []
            for field in self.content_fields:
                if field in row and row[field]:
                    content_parts.append(str(row[field]))
            
            content = "\n".join(content_parts)
            
            # Build metadata
            doc_metadata = {
                **metadata,
                'source_type': 'database',
                'row_index': index,
                'parser_type': 'database'
            }
            
            # Add specified metadata fields
            for field in self.metadata_fields:
                if field in row:
                    doc_metadata[field] = row[field]
            
            return Document(
                content=content,
                metadata=doc_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error processing database row {index}: {e}")
            return None
```

## Custom Embedder

### Domain-Specific Embedder

```python
# src/rag/ingestion/embedders/domain_embedder.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from src.rag.ingestion.embedders.base_embedder import BaseEmbedder

class DomainSpecificEmbedder(BaseEmbedder):
    """Custom embedder with domain-specific preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'all-mpnet-base-v2')
        self.domain = config.get('domain', 'general')
        
        # Load model
        self.model = SentenceTransformer(self.model_name)
        
        # Load domain-specific vocabulary
        self.domain_vocab = self._load_domain_vocabulary()
    
    def _load_domain_vocabulary(self) -> Dict[str, str]:
        """Load domain-specific vocabulary mappings."""
        vocab_mappings = {
            'medical': {
                'MI': 'myocardial infarction',
                'HTN': 'hypertension',
                'DM': 'diabetes mellitus',
                'COPD': 'chronic obstructive pulmonary disease'
            },
            'legal': {
                'LLC': 'limited liability company',
                'IP': 'intellectual property',
                'NDA': 'non-disclosure agreement'
            },
            'financial': {
                'ROI': 'return on investment',
                'EBITDA': 'earnings before interest taxes depreciation amortization',
                'YoY': 'year over year'
            }
        }
        return vocab_mappings.get(self.domain, {})
    
    def _preprocess_text(self, text: str) -> str:
        """Apply domain-specific preprocessing."""
        processed_text = text
        
        # Expand domain-specific abbreviations
        for abbrev, expansion in self.domain_vocab.items():
            processed_text = processed_text.replace(abbrev, expansion)
        
        return processed_text
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with domain-specific preprocessing."""
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        
        return embeddings.tolist()
```

## Custom Retriever

### Hybrid Search Retriever

```python
# src/rag/chatbot/retrievers/hybrid_retriever.py
from typing import List, Dict, Any, Tuple
from src.rag.chatbot.retrievers.base_retriever import BaseRetriever
from src.rag.shared.models import Document

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vector_weight = config.get('vector_weight', 0.7)
        self.keyword_weight = config.get('keyword_weight', 0.3)
    
    async def _retrieve_documents(self, query: str, top_k: int = 10, **kwargs) -> List[Tuple[Document, float]]:
        """Retrieve documents using hybrid search."""
        # Vector search
        vector_results = await self._vector_search(query, top_k * 2)
        
        # Keyword search
        keyword_results = await self._keyword_search(query, top_k * 2)
        
        # Combine and rerank
        combined_results = self._combine_results(vector_results, keyword_results)
        
        # Return top_k results
        return combined_results[:top_k]
    
    async def _vector_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Perform vector similarity search."""
        # Generate query embedding
        query_embedding = await self.embedder.embed_single(query)
        
        # Search vector store
        results = await self.vector_store.similarity_search_with_score(
            query_embedding, top_k
        )
        
        return results
    
    async def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Perform keyword search."""
        # Implementation depends on your keyword search system
        # This is a simplified example
        results = []
        # Add your keyword search logic here
        return results
    
    def _combine_results(self, vector_results: List[Tuple[Document, float]], 
                        keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Combine vector and keyword search results."""
        # Create document ID mapping
        doc_scores = {}
        
        # Add vector scores
        for doc, score in vector_results:
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                'document': doc,
                'vector_score': score,
                'keyword_score': 0.0
            }
        
        # Add keyword scores
        for doc, score in keyword_results:
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['keyword_score'] = score
            else:
                doc_scores[doc_id] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = (
                self.vector_weight * scores['vector_score'] +
                self.keyword_weight * scores['keyword_score']
            )
            
            combined_results.append((scores['document'], combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique document ID."""
        return f"{doc.metadata.get('source_file', 'unknown')}_{hash(doc.content[:100])}"
```

## Custom Generator

### Domain-Specific Generator

```python
# src/rag/chatbot/generators/domain_generator.py
from typing import List, Dict, Any
from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.shared.models import Document

class DomainSpecificGenerator(BaseGenerator):
    """Custom generator with domain-specific prompts."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain = config.get('domain', 'general')
        self.custom_prompts = self._load_domain_prompts()
    
    def _load_domain_prompts(self) -> Dict[str, str]:
        """Load domain-specific prompt templates."""
        prompts = {
            'medical': """You are a medical AI assistant. Based on the medical documents provided, 
            answer the question with accurate medical information. Always include disclaimers about 
            consulting healthcare professionals.
            
            Question: {query}
            Medical Context: {context}
            
            Provide a comprehensive medical answer with relevant details from the context.""",
            
            'legal': """You are a legal AI assistant. Based on the legal documents provided,
            answer the question with accurate legal information. Include relevant citations
            and always disclaim about consulting qualified legal professionals.
            
            Question: {query}
            Legal Context: {context}
            
            Provide a comprehensive legal analysis with relevant citations.""",
            
            'financial': """You are a financial AI assistant. Based on the financial documents,
            answer the question with accurate financial analysis. Include relevant data and metrics.
            Always disclaim about investment advice.
            
            Question: {query}
            Financial Context: {context}
            
            Provide comprehensive financial analysis with relevant data."""
        }
        
        return prompts.get(self.domain, 
            "Question: {query}\nContext: {context}\nAnswer based on the provided context:")
    
    async def _generate_response(self, query: str, context: List[Document], 
                               session_id: str, **kwargs) -> Dict[str, Any]:
        """Generate domain-specific response."""
        
        # Format context
        formatted_context = '\n\n'.join([doc.content for doc in context])
        
        # Build prompt
        prompt = self.custom_prompts.format(
            query=query,
            context=formatted_context
        )
        
        # Generate response
        response = await self.model.generate_content(prompt, **kwargs)
        
        return {
            'response': response,
            'domain': self.domain,
            'sources': [doc.metadata for doc in context],
            'session_id': session_id
        }
```

## Pipeline Integration

### Custom Pipeline Factory

```python
# src/rag/pipeline/pipeline_factory.py
from typing import Dict, Any
from src.rag.pipeline.custom_pipeline import RAGPipeline

class PipelineFactory:
    """Factory for creating custom RAG pipelines."""
    
    @staticmethod
    def create_pipeline(config: Dict[str, Any]) -> RAGPipeline:
        """Create custom pipeline based on configuration."""
        pipeline_type = config.get('pipeline_type', 'standard')
        
        if pipeline_type == 'medical':
            return PipelineFactory._create_medical_pipeline(config)
        elif pipeline_type == 'legal':
            return PipelineFactory._create_legal_pipeline(config)
        elif pipeline_type == 'financial':
            return PipelineFactory._create_financial_pipeline(config)
        else:
            return RAGPipeline(config)
    
    @staticmethod
    def _create_medical_pipeline(config: Dict[str, Any]) -> RAGPipeline:
        """Create medical-specific pipeline."""
        medical_config = {
            **config,
            'parser': {
                'type': 'custom_json',
                'schema_mapping': {
                    'patient_id': 'patient_identifier',
                    'diagnosis': 'medical_diagnosis'
                },
                'extract_fields': ['symptoms', 'diagnosis', 'treatment', 'medication']
            },
            'embedder': {
                'type': 'domain_specific',
                'domain': 'medical',
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'generator': {
                'type': 'domain_specific',
                'domain': 'medical'
            }
        }
        
        return RAGPipeline(medical_config)
```

## Configuration Management

### Custom Pipeline Configuration

```yaml
# config/medical_pipeline.yaml
pipeline_type: medical

parser:
  type: custom_json
  schema_mapping:
    patient_id: patient_identifier
    diagnosis: medical_diagnosis
    symptoms: patient_symptoms
  extract_fields:
    - symptoms
    - diagnosis
    - treatment
    - medication
    - notes

embedder:
  type: domain_specific
  domain: medical
  model_name: sentence-transformers/all-MiniLM-L6-v2
  preprocessing_rules:
    expand_abbreviations: true
    normalize_medical_terms: true

retriever:
  type: hybrid
  vector_weight: 0.7
  keyword_weight: 0.3
  elasticsearch_url: localhost:9200
  index_name: medical_documents

generator:
  type: domain_specific
  domain: medical
  model_name: gemini-1.5-pro-002
  include_disclaimers: true
  response_format: structured

memory:
  type: langgraph_checkpoint
  store_type: postgres
  session_timeout: 3600
```

## Testing Custom Components

### Component Testing

```python
# tests/test_custom_components.py
import pytest
import asyncio
from src.rag.ingestion.parsers.custom_json_parser import CustomJSONParser
from src.rag.ingestion.embedders.domain_embedder import DomainSpecificEmbedder

class TestCustomComponents:
    
    @pytest.mark.asyncio
    async def test_custom_json_parser(self):
        """Test custom JSON parser."""
        config = {
            'schema_mapping': {'id': 'doc_id', 'title': 'doc_title'},
            'extract_fields': ['title', 'content']
        }
        
        parser = CustomJSONParser(config)
        
        # Create test JSON file
        test_data = [
            {'id': '1', 'title': 'Test Doc', 'content': 'Test content'},
            {'id': '2', 'title': 'Another Doc', 'content': 'More content'}
        ]
        
        import json
        with open('test.json', 'w') as f:
            json.dump(test_data, f)
        
        # Parse file
        documents = await parser._parse_file('test.json', {})
        
        assert len(documents) == 2
        assert documents[0].metadata['doc_id'] == '1'
        assert 'title: Test Doc' in documents[0].content
    
    @pytest.mark.asyncio
    async def test_domain_embedder(self):
        """Test domain-specific embedder."""
        config = {
            'domain': 'medical',
            'model_name': 'all-MiniLM-L6-v2'
        }
        
        embedder = DomainSpecificEmbedder(config)
        
        # Test text with medical abbreviations
        texts = ['Patient has MI and HTN', 'Normal blood pressure']
        embeddings = await embedder._get_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        
        # Test that abbreviations are expanded
        processed = embedder._preprocess_text('Patient has MI')
        assert 'myocardial infarction' in processed

    def test_pipeline_factory(self):
        """Test custom pipeline factory."""
        config = {'pipeline_type': 'medical'}
        
        pipeline = PipelineFactory.create_pipeline(config)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'parser')
        assert hasattr(pipeline, 'embedder')
        assert hasattr(pipeline, 'generator')
```

## Best Practices

### 1. Component Design

```python
# Always extend base classes
class CustomParser(BaseParser):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Custom initialization

# Use proper error handling
try:
    result = await self.process_item(item)
except Exception as e:
    self.logger.error(f"Processing failed: {e}")
    return None

# Implement proper logging
self.logger.info("Processing started", extra={
    'extra_fields': {
        'component': 'custom_parser',
        'file_path': file_path
    }
})
```

### 2. Configuration Management

```python
# Use configuration validation
from pydantic import BaseModel

class CustomParserConfig(BaseModel):
    schema_mapping: Dict[str, str]
    extract_fields: List[str]
    domain: str = 'general'

# Validate configuration
config = CustomParserConfig(**raw_config)
```

### 3. Performance Optimization

```python
# Use async/await for I/O operations
async def process_documents(self, documents: List[Document]):
    tasks = [self.process_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def preprocess_text(self, text: str) -> str:
    # Expensive preprocessing
    return processed_text
```

## Examples

### Complete Medical Pipeline Example

```python
# examples/medical_pipeline_example.py
import asyncio
from src.rag.pipeline.pipeline_factory import PipelineFactory

async def main():
    # Configure medical pipeline
    config = {
        'pipeline_type': 'medical',
        'parser': {
            'type': 'custom_json',
            'domain': 'medical'
        },
        'embedder': {
            'type': 'domain_specific',
            'domain': 'medical'
        },
        'generator': {
            'type': 'domain_specific',
            'domain': 'medical'
        }
    }
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Ingest medical documents
    doc_id = await pipeline.ingest_document(
        'medical_records.json',
        {'source': 'hospital_system', 'type': 'patient_records'}
    )
    
    # Query the system
    response = await pipeline.query(
        'What are the common symptoms of hypertension?',
        session_id='medical_session_1'
    )
    
    print(f"Response: {response['response']}")
    print(f"Domain: {response['domain']}")
    print(f"Sources: {len(response['sources'])} documents")

if __name__ == "__main__":
    asyncio.run(main())
```

This tutorial provides a comprehensive guide to building custom RAG pipelines. You can extend these examples to create specialized systems for your specific domain and use cases.
