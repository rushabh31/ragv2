"""
Test Script for All Generation Models.

This script demonstrates how to use all the generation models with the universal
authentication system and tests their functionality.
"""

import asyncio
import logging
import os
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all generation models
from src.models.generation import (
    AnthropicVertexGenAI,
    OpenAIGenAI,
    VertexGenAI,
    AzureOpenAIGenAI
)
from src.models.generation.model_factory import GenerationModelFactory


async def test_anthropic_vertex():
    """Test Anthropic Vertex AI model."""
    print("\\n" + "="*50)
    print("Testing Anthropic Vertex AI (Claude)")
    print("="*50)
    
    try:
        model = AnthropicVertexGenAI(
            model_name="claude-3-5-sonnet@20240229",
            region="us-east5"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test generation
        print("Testing generation...")
        response = await model.chat_completion(
            "What is artificial intelligence?",
            max_tokens=100,
            temperature=0.7
        )
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_openai():
    """Test OpenAI model."""
    print("\\n" + "="*50)
    print("Testing OpenAI (Meta-Llama)")
    print("="*50)
    
    try:
        model = OpenAIGenAI(
            model_name="Meta-Llama-3-70B-Instruct"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test generation
        print("Testing generation...")
        response = await model.chat_completion(
            "What is artificial intelligence?",
            system_message="You are a helpful AI assistant.",
            temperature=0.7
        )
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_vertex_ai():
    """Test Vertex AI model."""
    print("\\n" + "="*50)
    print("Testing Vertex AI (Gemini)")
    print("="*50)
    
    try:
        model = VertexGenAI(
            model_name="gemini-1.5-pro-002"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test generation
        print("Testing generation...")
        response = await model.chat_completion(
            "What is artificial intelligence?",
            temperature=0.7,
            max_output_tokens=100
        )
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_azure_openai():
    """Test Azure OpenAI model."""
    print("\\n" + "="*50)
    print("Testing Azure OpenAI (GPT-4)")
    print("="*50)
    
    try:
        model = AzureOpenAIGenAI(
            model_name="GPT4-o",
            api_version="2023-05-15"
        )
        
        # Test authentication
        print("Testing authentication...")
        token = model.get_coin_token()
        print(f"Token acquired: {'✓' if token else '✗'}")
        
        # Test health status
        health = model.get_auth_health_status()
        print(f"Auth health: {health.get('auth_manager_status', 'unknown')}")
        
        # Test generation
        print("Testing generation...")
        response = await model.chat_completion(
            "What is artificial intelligence?",
            system_message="You are a helpful AI assistant.",
            temperature=0.2
        )
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_model_factory():
    """Test the generation model factory."""
    print("\\n" + "="*50)
    print("Testing Generation Model Factory")
    print("="*50)
    
    try:
        # Test supported providers
        providers = GenerationModelFactory.get_supported_providers()
        print(f"Supported providers: {providers}")
        
        # Test creating models via factory
        for provider in providers:
            try:
                print(f"\\nCreating {provider} model...")
                model = GenerationModelFactory.create_model(provider)
                default_model = GenerationModelFactory.get_default_model(provider)
                print(f"Default model for {provider}: {default_model}")
                print(f"Model created: ✓")
            except Exception as e:
                print(f"Failed to create {provider} model: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("Universal Generation Models Test Suite")
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
        ("Model Factory", test_model_factory),
        ("Anthropic Vertex AI", test_anthropic_vertex),
        ("OpenAI", test_openai),
        ("Vertex AI", test_vertex_ai),
        ("Azure OpenAI", test_azure_openai),
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
        print(f"{test_name:<20}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Universal authentication system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check configuration and environment variables.")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
