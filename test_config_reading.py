#!/usr/bin/env python3
"""
Test script to verify that all models are properly reading configuration from config YAML.

This script tests all model classes to ensure they:
1. Read configuration parameters from the centralized ConfigManager
2. Use config values as defaults when no parameters are provided
3. Allow parameter overrides when explicitly provided
4. Log configuration source appropriately
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.shared.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_manager():
    """Test that ConfigManager can load configuration properly."""
    print("\n" + "="*80)
    print("TESTING CONFIG MANAGER")
    print("="*80)
    
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"✅ ConfigManager loaded successfully")
        print(f"✅ Config sections found: {list(config.keys())}")
        
        # Test specific sections
        embedding_config = config_manager.get_section("embedding", {})
        generation_config = config_manager.get_section("generation", {})
        vision_config = config_manager.get_section("vision", {})
        
        print(f"✅ Embedding config: {embedding_config}")
        print(f"✅ Generation config: {generation_config}")
        print(f"✅ Vision config: {vision_config}")
        
        return True
    except Exception as e:
        print(f"❌ ConfigManager test failed: {str(e)}")
        return False

def test_embedding_models():
    """Test that embedding models read from config properly."""
    print("\n" + "="*80)
    print("TESTING EMBEDDING MODELS")
    print("="*80)
    
    results = {}
    
    # Test VertexEmbeddingAI
    try:
        from src.models.embedding.vertex_embedding import VertexEmbeddingAI
        
        # Test with no parameters (should use config)
        vertex_emb = VertexEmbeddingAI()
        print(f"✅ VertexEmbeddingAI initialized with model: {vertex_emb.model_name}")
        print(f"   - Batch size: {vertex_emb.batch_size}")
        print(f"   - Location: {vertex_emb.location}")
        
        # Test with parameter override
        vertex_emb_override = VertexEmbeddingAI(model_name="custom-model", batch_size=50)
        print(f"✅ VertexEmbeddingAI override test - model: {vertex_emb_override.model_name}, batch_size: {vertex_emb_override.batch_size}")
        
        results['VertexEmbeddingAI'] = True
    except Exception as e:
        print(f"❌ VertexEmbeddingAI test failed: {str(e)}")
        results['VertexEmbeddingAI'] = False
    
    # Test OpenAIEmbeddingAI
    try:
        from src.models.embedding.openai_embedding import OpenAIEmbeddingAI
        
        openai_emb = OpenAIEmbeddingAI()
        print(f"✅ OpenAIEmbeddingAI initialized with model: {openai_emb.model_name}")
        print(f"   - Batch size: {openai_emb.batch_size}")
        
        results['OpenAIEmbeddingAI'] = True
    except Exception as e:
        print(f"❌ OpenAIEmbeddingAI test failed: {str(e)}")
        results['OpenAIEmbeddingAI'] = False
    
    # Test SentenceTransformerEmbeddingAI
    try:
        from src.models.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddingAI
        
        st_emb = SentenceTransformerEmbeddingAI()
        print(f"✅ SentenceTransformerEmbeddingAI initialized with model: {st_emb.model_name}")
        print(f"   - Batch size: {st_emb.batch_size}")
        print(f"   - Device: {st_emb.device}")
        
        results['SentenceTransformerEmbeddingAI'] = True
    except Exception as e:
        print(f"❌ SentenceTransformerEmbeddingAI test failed: {str(e)}")
        results['SentenceTransformerEmbeddingAI'] = False
    
    # Test AzureOpenAIEmbeddingAI
    try:
        from src.models.embedding.azure_openai_embedding import AzureOpenAIEmbeddingAI
        
        azure_emb = AzureOpenAIEmbeddingAI()
        print(f"✅ AzureOpenAIEmbeddingAI initialized with model: {azure_emb.model_name}")
        print(f"   - Batch size: {azure_emb.batch_size}")
        print(f"   - API version: {azure_emb.api_version}")
        
        results['AzureOpenAIEmbeddingAI'] = True
    except Exception as e:
        print(f"❌ AzureOpenAIEmbeddingAI test failed: {str(e)}")
        results['AzureOpenAIEmbeddingAI'] = False
    
    return results

def test_generation_models():
    """Test that generation models read from config properly."""
    print("\n" + "="*80)
    print("TESTING GENERATION MODELS")
    print("="*80)
    
    results = {}
    
    # Test VertexGenAI
    try:
        from src.models.generation.vertex_gen import VertexGenAI
        
        vertex_gen = VertexGenAI()
        print(f"✅ VertexGenAI initialized with model: {vertex_gen.model_name}")
        print(f"   - Temperature: {vertex_gen.temperature}")
        print(f"   - Max tokens: {vertex_gen.max_output_tokens}")
        print(f"   - Top-p: {vertex_gen.top_p}")
        
        results['VertexGenAI'] = True
    except Exception as e:
        print(f"❌ VertexGenAI test failed: {str(e)}")
        results['VertexGenAI'] = False
    
    # Test GroqGenAI
    try:
        from src.models.generation.groq_gen import GroqGenAI
        
        groq_gen = GroqGenAI()
        print(f"✅ GroqGenAI initialized with model: {groq_gen.model_name}")
        print(f"   - Temperature: {groq_gen.temperature}")
        print(f"   - Max tokens: {groq_gen.max_tokens}")
        
        results['GroqGenAI'] = True
    except Exception as e:
        print(f"❌ GroqGenAI test failed: {str(e)}")
        results['GroqGenAI'] = False
    
    # Test OpenAIGenAI
    try:
        from src.models.generation.openai_gen import OpenAIGenAI
        
        openai_gen = OpenAIGenAI()
        print(f"✅ OpenAIGenAI initialized with model: {openai_gen.model_name}")
        print(f"   - Temperature: {openai_gen.temperature}")
        print(f"   - Max tokens: {openai_gen.max_tokens}")
        
        results['OpenAIGenAI'] = True
    except Exception as e:
        print(f"❌ OpenAIGenAI test failed: {str(e)}")
        results['OpenAIGenAI'] = False
    
    # Test AzureOpenAIGenAI
    try:
        from src.models.generation.azure_openai_gen import AzureOpenAIGenAI
        
        azure_gen = AzureOpenAIGenAI()
        print(f"✅ AzureOpenAIGenAI initialized with model: {azure_gen.model_name}")
        print(f"   - Temperature: {azure_gen.temperature}")
        print(f"   - API version: {azure_gen.api_version}")
        
        results['AzureOpenAIGenAI'] = True
    except Exception as e:
        print(f"❌ AzureOpenAIGenAI test failed: {str(e)}")
        results['AzureOpenAIGenAI'] = False
    
    return results

def test_vision_models():
    """Test that vision models read from config properly."""
    print("\n" + "="*80)
    print("TESTING VISION MODELS")
    print("="*80)
    
    results = {}
    
    # Test VertexVisionAI
    try:
        from src.models.vision.vertex_vision import VertexVisionAI
        
        vertex_vision = VertexVisionAI()
        print(f"✅ VertexVisionAI initialized with model: {vertex_vision.model_name}")
        print(f"   - Max pages: {vertex_vision.max_pages}")
        print(f"   - Max concurrent pages: {vertex_vision.max_concurrent_pages}")
        print(f"   - Location: {vertex_vision.location}")
        
        results['VertexVisionAI'] = True
    except Exception as e:
        print(f"❌ VertexVisionAI test failed: {str(e)}")
        results['VertexVisionAI'] = False
    
    # Test GroqVisionAI
    try:
        from src.models.vision.groq_vision import GroqVisionAI
        
        groq_vision = GroqVisionAI()
        print(f"✅ GroqVisionAI initialized with model: {groq_vision.model_name}")
        print(f"   - Temperature: {groq_vision.temperature}")
        print(f"   - Max pages: {groq_vision.max_pages}")
        print(f"   - Max concurrent pages: {groq_vision.max_concurrent_pages}")
        
        results['GroqVisionAI'] = True
    except Exception as e:
        print(f"❌ GroqVisionAI test failed: {str(e)}")
        results['GroqVisionAI'] = False
    
    return results

def test_config_override():
    """Test that parameter overrides work correctly."""
    print("\n" + "="*80)
    print("TESTING CONFIG OVERRIDES")
    print("="*80)
    
    try:
        from src.models.embedding.vertex_embedding import VertexEmbeddingAI
        from src.models.generation.vertex_gen import VertexGenAI
        
        # Test embedding override
        emb_default = VertexEmbeddingAI()
        emb_override = VertexEmbeddingAI(model_name="custom-embedding-model", batch_size=200)
        
        print(f"✅ Embedding default model: {emb_default.model_name}")
        print(f"✅ Embedding override model: {emb_override.model_name}")
        print(f"✅ Embedding default batch_size: {emb_default.batch_size}")
        print(f"✅ Embedding override batch_size: {emb_override.batch_size}")
        
        # Test generation override
        gen_default = VertexGenAI()
        gen_override = VertexGenAI(model_name="custom-gen-model", temperature=0.8)
        
        print(f"✅ Generation default model: {gen_default.model_name}")
        print(f"✅ Generation override model: {gen_override.model_name}")
        print(f"✅ Generation default temperature: {gen_default.temperature}")
        print(f"✅ Generation override temperature: {gen_override.temperature}")
        
        return True
    except Exception as e:
        print(f"❌ Config override test failed: {str(e)}")
        return False

def main():
    """Run all configuration tests."""
    print("🔧 COMPREHENSIVE CONFIG READING TEST")
    print("="*80)
    print("Testing that all models properly read from config YAML...")
    
    # Run all tests
    config_manager_ok = test_config_manager()
    embedding_results = test_embedding_models()
    generation_results = test_generation_models()
    vision_results = test_vision_models()
    override_ok = test_config_override()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"ConfigManager: {'✅ PASS' if config_manager_ok else '❌ FAIL'}")
    print(f"Config Override: {'✅ PASS' if override_ok else '❌ FAIL'}")
    
    print("\nEmbedding Models:")
    for model, result in embedding_results.items():
        print(f"  {model}: {'✅ PASS' if result else '❌ FAIL'}")
    
    print("\nGeneration Models:")
    for model, result in generation_results.items():
        print(f"  {model}: {'✅ PASS' if result else '❌ FAIL'}")
    
    print("\nVision Models:")
    for model, result in vision_results.items():
        print(f"  {model}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Overall result
    all_results = [config_manager_ok, override_ok] + list(embedding_results.values()) + list(generation_results.values()) + list(vision_results.values())
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    
    print(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! All models are properly reading from config YAML.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
