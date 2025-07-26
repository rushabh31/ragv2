#!/usr/bin/env python3

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Set the config file path to the ingestion example config
config_path = os.path.abspath(os.path.join(project_root, 'examples', 'rag', 'ingestion', 'config.yaml'))
os.environ['CONFIG_PATH'] = config_path

print(f"Testing fixes with config: {config_path}")
print(f"Config file exists: {os.path.exists(config_path)}")

try:
    # Test 1: Configuration loading
    print("\n=== Testing Configuration Loading ===")
    from src.rag.shared.utils.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"✅ Configuration loaded successfully")
    print(f"Available sections: {list(config.keys())}")
    
    # Check if vision section exists
    if 'vision' in config:
        print(f"✅ Vision section found: {config['vision']}")
    else:
        print("❌ Vision section missing from config")
    
    # Test 2: Vision model factory
    print("\n=== Testing Vision Model Factory ===")
    from src.models.vision.vision_factory import VisionModelFactory
    
    # Try to create a vertex_ai vision model
    vision_model = VisionModelFactory.create_model("vertex_ai", model_name="gemini-1.5-pro-002")
    print(f"✅ Vision model created successfully: {type(vision_model).__name__}")
    
    # Test authentication
    auth_status = vision_model.get_auth_health_status()
    print(f"✅ Authentication status: {auth_status}")
    
    print("\n🎉 All tests passed! The fixes are working.")
    
except Exception as e:
    print(f"\n❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
