#!/usr/bin/env python3
"""
Test script to verify ConfigManager integration with model factories.

This script tests that all model factories can successfully read configuration
from YAML files using ConfigManager.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_config_integration():
    """Test ConfigManager integration with all model factories."""
    
    print("🧪 Testing ConfigManager Integration with Model Factories")
    print("=" * 60)
    
    try:
        # Test EmbeddingModelFactory
        print("\n📊 Testing EmbeddingModelFactory...")
        from src.models.embedding.embedding_factory import EmbeddingModelFactory
        
        try:
            embedder = EmbeddingModelFactory.create_model()
            print(f"✅ EmbeddingModelFactory created successfully: {type(embedder).__name__}")
        except Exception as e:
            print(f"❌ EmbeddingModelFactory failed: {str(e)}")
        
        # Test GenerationModelFactory
        print("\n🤖 Testing GenerationModelFactory...")
        from src.models.generation.model_factory import GenerationModelFactory
        
        try:
            generator = GenerationModelFactory.create_model()
            print(f"✅ GenerationModelFactory created successfully: {type(generator).__name__}")
        except Exception as e:
            print(f"❌ GenerationModelFactory failed: {str(e)}")
        
        # Test VisionModelFactory
        print("\n👁️ Testing VisionModelFactory...")
        from src.models.vision.vision_factory import VisionModelFactory
        
        try:
            vision_model = VisionModelFactory.create_model()
            print(f"✅ VisionModelFactory created successfully: {type(vision_model).__name__}")
        except Exception as e:
            print(f"❌ VisionModelFactory failed: {str(e)}")
        
        # Test ConfigManager reading
        print("\n⚙️ Testing ConfigManager configuration reading...")
        from src.rag.shared.utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # Test embedding config
        embedding_config = config_manager.get_section("embedding", {})
        print(f"📊 Embedding config: provider={embedding_config.get('provider', 'not found')}")
        
        # Test generation config
        generation_config = config_manager.get_section("generation", {})
        print(f"🤖 Generation config: provider={generation_config.get('provider', 'not found')}")
        
        # Test vision config
        vision_config = config_manager.get_section("vision", {})
        print(f"👁️ Vision config: provider={vision_config.get('provider', 'not found')}")
        
        print("\n🎉 ConfigManager Integration Test Completed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_config_integration())
    sys.exit(0 if success else 1)
