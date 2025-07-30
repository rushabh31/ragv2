"""
Complete System Test Script.

This script tests all components of the updated RAG system to ensure
everything is working with the new universal authentication and multi-provider models.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports for all new models
try:
    from src.models.generation import (
        AnthropicVertexGenAI,
        OpenAIGenAI,
        VertexGenAI,
        AzureOpenAIGenAI
    )
    from src.models.embedding import (
        VertexEmbeddingAI,
        OpenAIEmbeddingAI,
        AzureOpenAIEmbeddingAI
    )
    from src.models.vision import VertexVisionAI
    
    # Test RAG component imports
    from src.models.generation.model_factory import GenerationModelFactory
    from src.models.embedding.embedding_factory import EmbeddingModelFactory
    from src.rag.ingestion.parsers.vision_parser import VisionParser
    from src.rag.ingestion.chunkers.semantic_chunker import SemanticChunker
    
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


async def test_vision_model():
    """Test the new VertexVisionAI model."""
    print("\n" + "="*50)
    print("Testing VertexVisionAI Model")
    print("="*50)
    
    try:
        model = VertexVisionAI()
        
        # Test authentication
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test model info
        info = model.get_model_info()
        print(f"Model info: {info['model_name']} ({info['provider']})")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_generation_models():
    """Test all generation models."""
    print("\n" + "="*50)
    print("Testing Generation Models")
    print("="*50)
    
    models = [
        ("Vertex AI", VertexGenAI),
        ("Anthropic Vertex", AnthropicVertexGenAI),
        ("OpenAI", OpenAIGenAI),
        ("Azure OpenAI", AzureOpenAIGenAI)
    ]
    
    results = {}
    
    for name, model_class in models:
        try:
            print(f"\nTesting {name}...")
            model = model_class()
            
            # Test authentication
            token = model.get_coin_token()
            print(f"  Token acquired: {'✓' if token else '✗'}")
            
            # Test health status
            health = model.get_auth_health_status()
            print(f"  Auth health: {health.get('auth_manager_status', 'unknown')}")
            
            results[name] = True
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[name] = False
    
    return results


async def test_embedding_models():
    """Test all embedding models."""
    print("\n" + "="*50)
    print("Testing Embedding Models")
    print("="*50)
    
    models = [
        ("Vertex AI Embedding", VertexEmbeddingAI),
        ("OpenAI Embedding", OpenAIEmbeddingAI),
        ("Azure OpenAI Embedding", AzureOpenAIEmbeddingAI)
    ]
    
    results = {}
    
    for name, model_class in models:
        try:
            print(f"\nTesting {name}...")
            model = model_class()
            
            # Test authentication
            token = model.get_coin_token()
            print(f"  Token acquired: {'✓' if token else '✗'}")
            
            # Test health status
            health = model.get_auth_health_status()
            print(f"  Auth health: {health.get('auth_manager_status', 'unknown')}")
            
            results[name] = True
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[name] = False
    
    return results


async def test_rag_generators():
    """Test RAG generator factory and components."""
    print("\n" + "="*50)
    print("Testing RAG Generators")
    print("="*50)
    
    try:
        # Test available generation providers
        providers_to_test = ["vertex", "anthropic_vertex", "openai", "azure_openai"]
        print(f"Testing generation providers: {providers_to_test}")
        
        results = {}
        
        for provider in providers_to_test:
            try:
                print(f"\nTesting {provider} generation model...")
                generator = GenerationModelFactory.create_model(provider)
                print(f"  {provider} generation model created: ✓")
                results[provider] = True
            except Exception as e:
                print(f"  {provider} generation model failed: {str(e)}")
                results[provider] = False
        
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


async def test_rag_embedders():
    """Test RAG embedder factory and components."""
    print("\n" + "="*50)
    print("Testing RAG Embedders")
    print("="*50)
    
    try:
        # Test available embedding providers
        providers_to_test = ["vertex_ai", "openai", "azure_openai"]
        print(f"Testing embedding providers: {providers_to_test}")
        
        results = {}
        
        for provider in providers_to_test:
            try:
                print(f"\nTesting {provider} embedding model...")
                embedder = EmbeddingModelFactory.create_model(provider)
                print(f"  {provider} embedding model created: ✓")
                results[provider] = True
            except Exception as e:
                print(f"  {provider} embedding model failed: {str(e)}")
                results[provider] = False
        
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


async def test_vision_parser():
    """Test the updated vision parser."""
    print("\n" + "="*50)
    print("Testing Vision Parser")
    print("="*50)
    
    try:
        config = {"model": "gemini-1.5-pro-002", "max_pages": 10}
        parser = VisionParser(config)
        
        print("Vision parser created: ✓")
        print(f"Model: {parser.model_name}")
        print(f"Max pages: {parser.max_pages}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_semantic_chunker():
    """Test the updated semantic chunker."""
    print("\n" + "="*50)
    print("Testing Semantic Chunker")
    print("="*50)
    
    try:
        config = {"max_chunk_size": 1000, "use_llm_boundary": False}
        chunker = SemanticChunker(config)
        
        print("Semantic chunker created: ✓")
        print(f"Max chunk size: {chunker.max_chunk_size}")
        print(f"Use LLM boundary: {chunker.use_llm_boundary}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_environment_setup():
    """Test environment variable setup."""
    print("\n" + "="*50)
    print("Testing Environment Setup")
    print("="*50)
    
    required_vars = [
        "COIN_CONSUMER_ENDPOINT_URL",
        "COIN_CONSUMER_CLIENT_ID",
        "COIN_CONSUMER_CLIENT_SECRET",
        "COIN_CONSUMER_SCOPE",
        "PROJECT_ID"
    ]
    
    optional_vars = [
        "VERTEXAI_API_ENDPOINT",
        "VERTEXAI_API_TRANSPORT",
        "OPENAI_API_KEY"
    ]
    
    print("Required environment variables:")
    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        status = "✓" if value else "✗"
        print(f"  {var}: {status}")
        if not value:
            missing_required.append(var)
    
    print("\nOptional environment variables:")
    for var in optional_vars:
        value = os.getenv(var)
        status = "✓" if value else "○"
        print(f"  {var}: {status}")
    
    if missing_required:
        print(f"\n⚠️  Missing required variables: {missing_required}")
        return False
    else:
        print("\n✓ All required environment variables are set")
        return True


async def main():
    """Main test function."""
    print("Complete System Test Suite")
    print("="*60)
    
    # Run all tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Vision Model", test_vision_model),
        ("Generation Models", test_generation_models),
        ("Embedding Models", test_embedding_models),
        ("RAG Generators", test_rag_generators),
        ("RAG Embedders", test_rag_embedders),
        ("Vision Parser", test_vision_parser),
        ("Semantic Chunker", test_semantic_chunker),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Test {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            # Handle nested results
            passed = sum(result.values())
            total = len(result)
            status = f"{passed}/{total} passed"
            print(f"{test_name:<25}: {status}")
            for sub_test, sub_result in result.items():
                sub_status = "✓" if sub_result else "✗"
                print(f"  {sub_test:<21}: {sub_status}")
        else:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name:<25}: {status}")
    
    # Calculate overall results
    total_tests = 0
    passed_tests = 0
    
    for result in results.values():
        if isinstance(result, dict):
            total_tests += len(result)
            passed_tests += sum(result.values())
        else:
            total_tests += 1
            passed_tests += int(result)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Complete system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check configuration and environment variables.")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required environment variables are set")
        print("2. Check network connectivity to authentication endpoints")
        print("3. Verify GCP project permissions and quotas")
        print("4. Review logs for specific error details")


if __name__ == "__main__":
    # Run the complete test suite
    asyncio.run(main())
