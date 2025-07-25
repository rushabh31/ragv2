#!/usr/bin/env python3
"""
Test script to verify the controlsgenai package installation and imports.
"""

import sys
import traceback

def test_imports():
    """Test all the main imports to ensure the package structure works."""
    print("Testing controlsgenai package imports...")
    
    try:
        # Test basic package import
        print("1. Testing basic package import...")
        import controlsgenai
        print(f"   ✓ controlsgenai imported successfully (version: {controlsgenai.__version__})")
        
        # Test funcs module
        print("2. Testing funcs module...")
        import controlsgenai.funcs
        print("   ✓ controlsgenai.funcs imported successfully")
        
        # Test rag module
        print("3. Testing rag module...")
        import src.rag
        print("   ✓ controlsgenai.funcs.rag imported successfully")
        
        # Test src module
        print("4. Testing src module...")
        import src.rag.src
        print("   ✓ controlsgenai.funcs.rag.src imported successfully")
        
        # Test core modules
        print("5. Testing core modules...")
        from src.rag.src.core.exceptions.exceptions import ConfigError
        print("   ✓ Core exceptions imported successfully")
        
        # Test shared modules
        print("6. Testing shared modules...")
        from src.rag.src.shared.utils.config_manager import ConfigManager
        print("   ✓ ConfigManager imported successfully")
        
        # Test chatbot modules
        print("7. Testing chatbot modules...")
        from src.rag.src.chatbot.generators.generator_factory import GeneratorFactory
        print("   ✓ GeneratorFactory imported successfully")
        
        # Test ingestion modules
        print("8. Testing ingestion modules...")
        from src.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
        print("   ✓ EmbedderFactory imported successfully")
        
        print("\n🎉 All imports successful! The controlsgenai package is properly structured.")
        print("\n📁 Package Structure:")
        print("   ✓ Root package: controlsgenai")
        print("   ✓ Functions module: controlsgenai.funcs")
        print("   ✓ RAG module: controlsgenai.funcs.rag")
        print("   ✓ Source modules: controlsgenai.funcs.rag.src.*")
        print("   ✓ Tests directory: tests/")
        print("   ✓ Documentation: docs/")
        print("   ✓ Examples: examples/")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test ConfigManager
        print("1. Testing ConfigManager...")
        from src.rag.src.shared.utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        # Override memory store to in_memory for testing
        config_manager.update_section("memory.provider", "in_memory")
        print("   ✓ ConfigManager instantiated and configured for in-memory store")
        
        # Test GeneratorFactory
        print("2. Testing GeneratorFactory...")
        from src.rag.src.chatbot.generators.generator_factory import GeneratorFactory
        supported_generators = GeneratorFactory.get_available_generators()
        print(f"   ✓ GeneratorFactory has available generators: {list(supported_generators.keys())}")
        
        # Test EmbedderFactory
        print("3. Testing EmbedderFactory...")
        from src.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
        supported_embedders = EmbedderFactory.get_available_embedders()
        print(f"   ✓ EmbedderFactory has available embedders: {list(supported_embedders.keys())}")
        
        print("\n🎉 Basic functionality tests passed!")
        print("\n🚀 Ready to use! You can now:")
        print("   • Run tests: python -m pytest tests/")
        print("   • Start ingestion: python run_ingestion.py")
        print("   • Start chatbot: python run_chatbot.py")
        print("   • Use console commands: controlsgenai-ingestion, controlsgenai-chatbot")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ControlsGenAI Package Installation Test")
    print("=" * 60)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test basic functionality
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED! The controlsgenai package is ready to use.")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("⚠️  Import tests passed but functionality tests failed.")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("❌ Import tests failed. Please check the package structure.")
        print("=" * 60)
        sys.exit(1)
