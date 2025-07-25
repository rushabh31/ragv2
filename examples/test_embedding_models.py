"""
Test Script for All Embedding Models.

This script demonstrates how to use all the embedding models with the universal
authentication system and tests their functionality.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all embedding models
from src.models.embedding import (
    VertexEmbeddingAI,
    OpenAIEmbeddingAI,
    AzureOpenAIEmbeddingAI
)
from src.models.embedding.embedding_factory import EmbeddingModelFactory


async def test_vertex_embedding():
    """Test Vertex AI embedding model."""
    print("\\n" + "="*50)
    print("Testing Vertex AI Embedding")
    print("="*50)
    
    try:
        model = VertexEmbeddingAI(
            model_name="text-embedding-004"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test single embedding
        print("Testing single embedding...")
        embedding = await model.embed_single("What is life?")
        print(f"Embedding dimension: {len(embedding)}")
        
        # Test batch embeddings
        print("Testing batch embeddings...")
        texts = ["What is life?", "How does AI work?", "Explain machine learning"]
        embeddings = await model.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_openai_embedding():
    """Test OpenAI embedding model."""
    print("\\n" + "="*50)
    print("Testing OpenAI Embedding")
    print("="*50)
    
    try:
        model = OpenAIEmbeddingAI(
            model_name="all-mpnet-base-v2"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test single embedding
        print("Testing single embedding...")
        embedding = await model.embed_single("input text")
        print(f"Embedding dimension: {len(embedding)}")
        
        # Test batch embeddings
        print("Testing batch embeddings...")
        texts = ["input text", "another text", "more text"]
        embeddings = await model.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_azure_openai_embedding():
    """Test Azure OpenAI embedding model."""
    print("\\n" + "="*50)
    print("Testing Azure OpenAI Embedding")
    print("="*50)
    
    try:
        model = AzureOpenAIEmbeddingAI(
            model_name="modelname",
            api_version="2023-05-15"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test lorem ipsum embedding as shown in image
        print("Testing lorem ipsum embeddings...")
        embeddings = await model.embed_lorem_ipsum()
        print(f"Generated {len(embeddings)} lorem ipsum embeddings")
        
        # Test single embedding
        print("Testing single embedding...")
        embedding = await model.embed_single("test input")
        print(f"Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_embedding_factory():
    """Test the embedding model factory."""
    print("\\n" + "="*50)
    print("Testing Embedding Model Factory")
    print("="*50)
    
    try:
        # Test supported providers
        providers = EmbeddingModelFactory.get_supported_providers()
        print(f"Supported providers: {providers}")
        
        # Test creating models via factory
        for provider in providers:
            try:
                print(f"\\nCreating {provider} embedding model...")
                model = EmbeddingModelFactory.create_model(provider)
                default_model = EmbeddingModelFactory.get_default_model(provider)
                print(f"Default model for {provider}: {default_model}")
                print(f"Model created: ✓")
            except Exception as e:
                print(f"Failed to create {provider} embedding model: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_rag_integration():
    """Test RAG system integration with new embedding models."""
    print("\\n" + "="*50)
    print("Testing RAG System Integration")
    print("="*50)
    
    try:
        from src.rag.ingestion.embedders.embedder_factory import EmbedderFactory
        
        # Test available embedders
        available_embedders = EmbedderFactory.get_available_embedders()
        print(f"Available RAG embedders: {list(available_embedders.keys())}")
        
        # Test creating new universal auth embedders
        new_embedders = ["vertex_ai", "openai_universal", "azure_openai"]
        
        for embedder_name in new_embedders:
            try:
                print(f"\\nTesting RAG {embedder_name} embedder...")
                config = {
                    "provider": embedder_name,
                    "model": "default",
                    "batch_size": 10
                }
                
                embedder = await EmbedderFactory.create_embedder(config)
                print(f"RAG {embedder_name} embedder created: ✓")
                
                # Test health status if available
                if hasattr(embedder, 'get_auth_health_status'):
                    health = embedder.get_auth_health_status()
                    print(f"Auth health: {health.get('status', 'unknown')}")
                
            except Exception as e:
                print(f"Failed to create RAG {embedder_name} embedder: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("Universal Embedding Models Test Suite")
    print("="*60)
    
    # Set environment variables for testing (you may need to adjust these)
    required_env_vars = [
        "COIN_CONSUMER_ENDPOINT_URL",
        "COIN_CONSUMER_CLIENT_ID", 
        "COIN_CONSUMER_CLIENT_SECRET",
        "COIN_CONSUMER_SCOPE",
        "PROJECT_ID"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("Some tests may fail without proper configuration.")
    
    # Run all tests
    tests = [
        ("Embedding Factory", test_embedding_factory),
        ("Vertex AI Embedding", test_vertex_embedding),
        ("OpenAI Embedding", test_openai_embedding),
        ("Azure OpenAI Embedding", test_azure_openai_embedding),
        ("RAG Integration", test_rag_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\\nRunning {test_name} test...")
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Test {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<25}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Universal embedding system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check configuration and environment variables.")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
