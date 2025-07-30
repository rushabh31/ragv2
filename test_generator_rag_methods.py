#!/usr/bin/env python3
"""
Test script to verify all generator modules support RAG response methods with documents and conversation history.

This script tests:
1. All generator models have the generate_response method
2. The method accepts query, documents, and conversation_history parameters
3. The method returns appropriate responses
4. Configuration is properly read from ConfigManager
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import all generator models
from models.generation.vertex_gen import VertexGenAI
from models.generation.openai_gen import OpenAIGenAI
from models.generation.azure_openai_gen import AzureOpenAIGenAI
from models.generation.groq_gen import GroqGenAI

# Import config manager
from rag.shared.utils.config_manager import ConfigManager

class GeneratorRAGTester:
    """Test class for generator RAG methods."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.test_results = {}
        
        # Test data
        self.test_query = "What is machine learning and how does it work?"
        
        self.test_documents = [
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions."
            },
            {
                "content": "There are three main types of machine learning: supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards)."
            }
        ]
        
        self.test_conversation_history = [
            {"role": "user", "content": "Hello, I'm interested in learning about AI."},
            {"role": "assistant", "content": "Hello! I'd be happy to help you learn about artificial intelligence. What specific aspect would you like to know about?"},
            {"role": "user", "content": "Can you explain the basics?"}
        ]
    
    async def test_generator_model(self, model_class, model_name: str) -> Dict[str, Any]:
        """Test a specific generator model."""
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        result = {
            "model_name": model_name,
            "class_name": model_class.__name__,
            "initialization": False,
            "config_reading": False,
            "has_generate_response": False,
            "method_signature": False,
            "basic_response": False,
            "rag_response": False,
            "error": None
        }
        
        try:
            # Test 1: Model initialization
            print(f"1. Testing {model_name} initialization...")
            model = model_class()
            result["initialization"] = True
            print(f"   ✅ {model_name} initialized successfully")
            
            # Test 2: Config reading
            print(f"2. Testing config reading...")
            if hasattr(model, 'model_name') and hasattr(model, 'temperature'):
                result["config_reading"] = True
                print(f"   ✅ Config parameters loaded: model={model.model_name}, temp={model.temperature}")
            else:
                print(f"   ❌ Config parameters not found")
            
            # Test 3: Check if generate_response method exists
            print(f"3. Testing generate_response method existence...")
            if hasattr(model, 'generate_response'):
                result["has_generate_response"] = True
                print(f"   ✅ generate_response method exists")
                
                # Test 4: Check method signature
                print(f"4. Testing method signature...")
                import inspect
                sig = inspect.signature(model.generate_response)
                params = list(sig.parameters.keys())
                expected_params = ['query', 'documents', 'conversation_history']
                
                if all(param in params for param in expected_params):
                    result["method_signature"] = True
                    print(f"   ✅ Method signature correct: {params}")
                else:
                    print(f"   ❌ Method signature incorrect. Expected: {expected_params}, Got: {params}")
                
                # Test 5: Basic response (query only)
                print(f"5. Testing basic response generation...")
                try:
                    basic_response = await model.generate_response(
                        query=self.test_query
                    )
                    if basic_response and isinstance(basic_response, str):
                        result["basic_response"] = True
                        print(f"   ✅ Basic response generated (length: {len(basic_response)})")
                        print(f"   Response preview: {basic_response[:100]}...")
                    else:
                        print(f"   ❌ Basic response invalid: {type(basic_response)}")
                except Exception as e:
                    print(f"   ❌ Basic response failed: {str(e)}")
                
                # Test 6: RAG response (with documents and history)
                print(f"6. Testing RAG response generation...")
                try:
                    rag_response = await model.generate_response(
                        query=self.test_query,
                        documents=self.test_documents,
                        conversation_history=self.test_conversation_history
                    )
                    if rag_response and isinstance(rag_response, str):
                        result["rag_response"] = True
                        print(f"   ✅ RAG response generated (length: {len(rag_response)})")
                        print(f"   Response preview: {rag_response[:100]}...")
                    else:
                        print(f"   ❌ RAG response invalid: {type(rag_response)}")
                except Exception as e:
                    print(f"   ❌ RAG response failed: {str(e)}")
                    
            else:
                print(f"   ❌ generate_response method not found")
                
        except Exception as e:
            result["error"] = str(e)
            print(f"   ❌ Error testing {model_name}: {str(e)}")
        
        return result
    
    async def run_all_tests(self):
        """Run tests on all generator models."""
        print("🧪 Generator RAG Methods Test Suite")
        print("=" * 80)
        
        # Test models
        models_to_test = [
            (VertexGenAI, "Vertex AI Generator"),
            (OpenAIGenAI, "OpenAI Generator"),
            (AzureOpenAIGenAI, "Azure OpenAI Generator"),
            (GroqGenAI, "Groq Generator")
        ]
        
        for model_class, model_name in models_to_test:
            result = await self.test_generator_model(model_class, model_name)
            self.test_results[model_name] = result
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print(f"\n{'='*80}")
        print("📊 TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for model_name, result in self.test_results.items():
            print(f"\n{model_name} ({result['class_name']}):")
            
            tests = [
                ("Initialization", result["initialization"]),
                ("Config Reading", result["config_reading"]),
                ("Has generate_response", result["has_generate_response"]),
                ("Method Signature", result["method_signature"]),
                ("Basic Response", result["basic_response"]),
                ("RAG Response", result["rag_response"])
            ]
            
            for test_name, passed in tests:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {test_name:20} {status}")
            
            if result["error"]:
                print(f"  Error: {result['error']}")
        
        # Overall statistics
        total_models = len(self.test_results)
        models_with_rag = sum(1 for r in self.test_results.values() if r["has_generate_response"])
        models_working = sum(1 for r in self.test_results.values() if r["rag_response"])
        
        print(f"\n{'='*80}")
        print("📈 OVERALL STATISTICS")
        print(f"{'='*80}")
        print(f"Total models tested: {total_models}")
        print(f"Models with generate_response method: {models_with_rag}/{total_models}")
        print(f"Models with working RAG response: {models_working}/{total_models}")
        
        if models_working == total_models:
            print(f"\n🎉 SUCCESS: All generator models support RAG response methods!")
        else:
            print(f"\n⚠️  WARNING: {total_models - models_working} models need attention")

async def main():
    """Main test function."""
    print("Starting Generator RAG Methods Test...")
    
    # Check environment
    print("\n🔍 Environment Check:")
    required_vars = [
        "PROJECT_ID",
        "COIN_CONSUMER_ENDPOINT_URL",
        "COIN_CONSUMER_CLIENT_ID",
        "COIN_CONSUMER_CLIENT_SECRET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var} is set")
        else:
            print(f"   ❌ {var} is missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Warning: Missing environment variables: {missing_vars}")
        print("Some tests may fail due to authentication issues.")
    
    # Run tests
    tester = GeneratorRAGTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
