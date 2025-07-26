#!/usr/bin/env python3
"""
Installation Test Script for RAG System

This script validates that the RAG system is properly installed and configured.
It checks dependencies, imports, basic functionality, and environment setup.
"""

import sys
import os
import importlib
from pathlib import Path
import subprocess
import json

def print_banner():
    """Print installation test banner."""
    print("=" * 80)
    print("🔧 RAG System Installation Test")
    print("=" * 80)
    print()

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version < (3, 9):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.9+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "asyncio",
        "aiofiles",
        "python-multipart",
        "jinja2",
        "pyyaml",
        "python-dotenv",
        "httpx",
        "groq",
        "sentence_transformers",
        "faiss_cpu",
        "pymupdf",
        "langgraph",
        "langgraph_checkpoint",
        "psycopg2_binary"
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == "faiss_cpu":
                import faiss
            elif package == "pymupdf":
                import fitz
            elif package == "python-multipart":
                import multipart
            elif package == "python-dotenv":
                import dotenv
            elif package == "psycopg2_binary":
                import psycopg2
            elif package == "langgraph_checkpoint":
                import langgraph.checkpoint
            else:
                importlib.import_module(package)
            
            installed_packages.append(package)
            print(f"✅ {package}")
            
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Not installed")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All {len(installed_packages)} required packages installed")
        return True

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        "src",
        "src/models",
        "src/models/generation",
        "src/models/embedding", 
        "src/models/vision",
        "src/rag",
        "src/rag/ingestion",
        "src/rag/chatbot",
        "examples",
        "examples/rag",
        "tests",
        "docs"
    ]
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "run_tests.py",
        "src/models/__init__.py",
        "src/rag/__init__.py"
    ]
    
    project_root = Path(__file__).parent
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n⚠️  Missing items: {len(missing_items)}")
        return False
    else:
        print(f"\n✅ Project structure complete")
        return True

def check_imports():
    """Check if core modules can be imported."""
    print("\n🔗 Checking core imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    core_imports = [
        ("src.models.generation", "GenerationModelFactory"),
        ("src.models.embedding", "EmbeddingModelFactory"),
        ("src.models.vision", "VisionModelFactory"),
        ("src.rag.ingestion.parsers.groq_vision_parser", "GroqVisionParser"),
        ("src.rag.chatbot.generators.groq_generator", "GroqGenerator"),
        ("src.rag.ingestion.embedders.sentence_transformer_embedder", "SentenceTransformerEmbedder"),
        ("src.utils.auth.universal_auth_manager", "UniversalAuthManager")
    ]
    
    failed_imports = []
    
    for module_name, class_name in core_imports:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {str(e)[:50]}...")
            failed_imports.append(f"{module_name}.{class_name}")
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {len(failed_imports)}")
        return False
    else:
        print(f"\n✅ All core imports successful")
        return True

def check_configuration():
    """Check if configuration files are valid."""
    print("\n⚙️  Checking configuration...")
    
    config_files = [
        "config.yaml",
        "examples/rag/ingestion/config.yaml",
        "examples/rag/chatbot/config.yaml"
    ]
    
    project_root = Path(__file__).parent
    valid_configs = 0
    
    for config_file in config_files:
        config_path = project_root / config_file
        
        if not config_path.exists():
            print(f"❌ {config_file} - File not found")
            continue
            
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            if isinstance(config, dict) and len(config) > 0:
                print(f"✅ {config_file} - Valid YAML")
                valid_configs += 1
            else:
                print(f"❌ {config_file} - Invalid structure")
                
        except Exception as e:
            print(f"❌ {config_file} - {str(e)[:50]}...")
    
    if valid_configs == len(config_files):
        print(f"\n✅ All configuration files valid")
        return True
    else:
        print(f"\n⚠️  {len(config_files) - valid_configs} configuration files have issues")
        return False

def check_environment_variables():
    """Check environment variables setup."""
    print("\n🔐 Checking environment variables...")
    
    # Check for .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env file loaded")
        except Exception as e:
            print(f"❌ Error loading .env: {e}")
    else:
        print("⚠️  .env file not found (optional)")
    
    # Check key environment variables
    env_vars = {
        "GROQ_API_KEY": "Groq API access",
        "COIN_CONSUMER_ENDPOINT_URL": "Universal auth endpoint",
        "COIN_CONSUMER_CLIENT_ID": "Universal auth client ID",
        "PROJECT_ID": "Google Cloud project ID"
    }
    
    found_vars = 0
    for var, description in env_vars.items():
        if os.getenv(var):
            print(f"✅ {var} - Set")
            found_vars += 1
        else:
            print(f"⚠️  {var} - Not set ({description})")
    
    if found_vars > 0:
        print(f"\n✅ {found_vars}/{len(env_vars)} environment variables configured")
        return True
    else:
        print(f"\n⚠️  No environment variables configured")
        print("   Some features may not work without API keys")
        return False

def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("\n🧪 Testing basic functionality...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Factory instantiation
    total_tests += 1
    try:
        from src.models.generation.generation_factory import GenerationModelFactory
        from src.models.generation.generation_factory import GenerationProvider
        
        # Test factory registry
        providers = list(GenerationProvider)
        if len(providers) >= 3:  # Should have vertex, groq, openai at minimum
            print("✅ Generation factory - Registry populated")
            tests_passed += 1
        else:
            print("❌ Generation factory - Insufficient providers")
            
    except Exception as e:
        print(f"❌ Generation factory - {str(e)[:50]}...")
    
    # Test 2: Configuration loading
    total_tests += 1
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ["generation", "embedding", "vision"]
        if all(section in config for section in required_sections):
            print("✅ Configuration loading - All sections present")
            tests_passed += 1
        else:
            print("❌ Configuration loading - Missing sections")
            
    except Exception as e:
        print(f"❌ Configuration loading - {str(e)[:50]}...")
    
    # Test 3: Parser instantiation (without API calls)
    total_tests += 1
    try:
        from src.rag.ingestion.parsers.groq_vision_parser import GroqVisionParser
        
        parser = GroqVisionParser({
            "model_name": "llama-3.2-11b-vision-preview",
            "max_concurrent_pages": 2
        })
        
        if hasattr(parser, '_parse_file') and hasattr(parser, 'max_concurrent_pages'):
            print("✅ Parser instantiation - GroqVisionParser created")
            tests_passed += 1
        else:
            print("❌ Parser instantiation - Missing methods")
            
    except Exception as e:
        print(f"❌ Parser instantiation - {str(e)[:50]}...")
    
    # Test 4: Embedder instantiation (local model)
    total_tests += 1
    try:
        from src.rag.ingestion.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
        
        embedder = SentenceTransformerEmbedder({
            "model": "all-mpnet-base-v2",
            "device": "cpu"
        })
        
        if hasattr(embedder, 'embed_documents'):
            print("✅ Embedder instantiation - SentenceTransformerEmbedder created")
            tests_passed += 1
        else:
            print("❌ Embedder instantiation - Missing methods")
            
    except Exception as e:
        print(f"❌ Embedder instantiation - {str(e)[:50]}...")
    
    print(f"\n📊 Basic functionality: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def run_quick_test():
    """Run a quick integration test if possible."""
    print("\n⚡ Running quick integration test...")
    
    # Check if we can run the test runner
    try:
        result = subprocess.run(
            [sys.executable, "run_tests.py", "--check-env"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Test runner - Executable and functional")
            return True
        else:
            print("❌ Test runner - Issues detected")
            print(f"   Error: {result.stderr[:100]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test runner - Timeout (may still be functional)")
        return False
    except Exception as e:
        print(f"❌ Test runner - {str(e)[:50]}...")
        return False

def print_summary(results):
    """Print installation test summary."""
    print("\n" + "=" * 80)
    print("📊 INSTALLATION TEST SUMMARY")
    print("=" * 80)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n📈 Overall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 INSTALLATION SUCCESSFUL!")
        print("Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Set up environment variables in .env file")
        print("2. Run: python run_tests.py --all")
        print("3. Start services: python examples/rag/ingestion/run_ingestion.py")
    else:
        print("\n⚠️  INSTALLATION INCOMPLETE")
        print("Please address the failed checks above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check project structure and file locations")
        print("3. Verify configuration files are valid")
        print("4. Set up environment variables")
    
    return passed_checks == total_checks

def main():
    """Main installation test function."""
    print_banner()
    
    # Run all checks
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Core Imports": check_imports(),
        "Configuration": check_configuration(),
        "Environment Variables": check_environment_variables(),
        "Basic Functionality": test_basic_functionality(),
        "Test Runner": run_quick_test()
    }
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
