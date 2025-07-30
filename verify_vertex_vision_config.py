#!/usr/bin/env python3
"""
Script to verify VertexVisionAI model configuration and kwargs.
This script will trace through the configuration flow and show what parameters
are being passed to the VertexVisionAI model.
"""

import os
import asyncio
import logging
from typing import Dict, Any

# Set up logging to see detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_vertex_vision_config():
    """Verify VertexVisionAI configuration and kwargs."""
    
    print("=" * 80)
    print("VERTEX VISION AI CONFIGURATION VERIFICATION")
    print("=" * 80)
    
    # 1. Check environment variables
    print("\n1. ENVIRONMENT VARIABLES:")
    print("-" * 40)
    env_vars = [
        "PROJECT_ID",
        "VERTEXAI_API_ENDPOINT", 
        "VERTEXAI_API_TRANSPORT",
        "SSL_CERT_FILE",
        "USERNAME",
        "COIN_CONSUMER_ENDPOINT_URL",
        "COIN_CONSUMER_CLIENT_ID",
        "COIN_CONSUMER_CLIENT_SECRET",
        "COIN_CONSUMER_SCOPE"
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "SECRET" in var or "PASSWORD" in var:
                display_value = "*" * 8
            elif len(value) > 50:
                display_value = value[:20] + "..." + value[-10:]
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: NOT SET")
    
    # 2. Test direct VertexVisionAI instantiation
    print("\n2. DIRECT MODEL INSTANTIATION:")
    print("-" * 40)
    
    try:
        from src.models.vision.vertex_vision import VertexVisionAI
        
        # Test with default parameters
        print("Creating VertexVisionAI with default parameters...")
        model_default = VertexVisionAI()
        
        print(f"✅ Model Name: {model_default.model_name}")
        print(f"✅ Project ID: {model_default.project_id}")
        print(f"✅ Location: {model_default.location}")
        print(f"✅ Config kwargs: {model_default.config}")
        
        # Test with custom parameters
        print("\nCreating VertexVisionAI with custom parameters...")
        custom_kwargs = {
            "timeout": 120,
            "max_output_tokens": 4096,
            "custom_param": "test_value"
        }
        
        model_custom = VertexVisionAI(
            model_name="gemini-1.5-pro-002",
            project_id="custom-project",
            location="us-west1",
            **custom_kwargs
        )
        
        print(f"✅ Model Name: {model_custom.model_name}")
        print(f"✅ Project ID: {model_custom.project_id}")
        print(f"✅ Location: {model_custom.location}")
        print(f"✅ Config kwargs: {model_custom.config}")
        
    except Exception as e:
        print(f"❌ Error creating VertexVisionAI: {str(e)}")
    
    # 3. Test factory instantiation
    print("\n3. FACTORY MODEL INSTANTIATION:")
    print("-" * 40)
    
    try:
        from src.models.vision import VisionModelFactory
        
        # Test factory with default parameters
        print("Creating model via VisionModelFactory with defaults...")
        factory_model_default = VisionModelFactory.create_model("vertex_ai")
        
        print(f"✅ Model Name: {factory_model_default.model_name}")
        print(f"✅ Project ID: {factory_model_default.project_id}")
        print(f"✅ Location: {factory_model_default.location}")
        print(f"✅ Config kwargs: {factory_model_default.config}")
        
        # Test factory with custom parameters
        print("\nCreating model via VisionModelFactory with custom kwargs...")
        custom_factory_kwargs = {
            "project_id": "factory-project",
            "location": "europe-west1",
            "timeout": 180,
            "temperature": 0.7,
            "api_endpoint": "custom-endpoint.com"
        }
        
        factory_model_custom = VisionModelFactory.create_model(
            provider="vertex_ai",
            model_name="gemini-1.5-flash-002",
            **custom_factory_kwargs
        )
        
        print(f"✅ Model Name: {factory_model_custom.model_name}")
        print(f"✅ Project ID: {factory_model_custom.project_id}")
        print(f"✅ Location: {factory_model_custom.location}")
        print(f"✅ Config kwargs: {factory_model_custom.config}")
        
    except Exception as e:
        print(f"❌ Error creating model via factory: {str(e)}")
    
    # 4. Test VisionParser configuration
    print("\n4. VISION PARSER CONFIGURATION:")
    print("-" * 40)
    
    try:
        from src.rag.ingestion.parsers.vision_parser import VisionParser
        from src.rag.shared.utils.config_manager import ConfigManager
        
        # Check system configuration
        config_manager = ConfigManager()
        vision_config = config_manager.get_config("vision")
        
        print(f"System vision config: {vision_config}")
        
        # Create vision parser
        parser_config = {
            "model": "gemini-1.5-pro-002",
            "max_pages": 50,
            "max_concurrent_pages": 3
        }
        
        parser = VisionParser(parser_config)
        
        print(f"✅ Parser model name: {parser.model_name}")
        print(f"✅ Parser vision provider: {parser.vision_provider}")
        print(f"✅ Parser vision config: {parser.vision_config}")
        print(f"✅ Parser max pages: {parser.max_pages}")
        print(f"✅ Parser max concurrent: {parser.max_concurrent_pages}")
        
        # Initialize the vision model
        print("\nInitializing vision model through parser...")
        await parser._init_vision_model()
        
        if parser.vision_model:
            print(f"✅ Vision model initialized successfully")
            print(f"✅ Model type: {type(parser.vision_model)}")
            print(f"✅ Model name: {parser.vision_model.model_name}")
            print(f"✅ Model project: {parser.vision_model.project_id}")
            print(f"✅ Model location: {parser.vision_model.location}")
            print(f"✅ Model config: {parser.vision_model.config}")
            
            # Test authentication
            print("\nTesting authentication...")
            auth_valid = await parser.vision_model.validate_authentication()
            print(f"✅ Authentication valid: {auth_valid}")
            
            if auth_valid:
                # Check what gets passed to vertexai.init()
                print("\nChecking Vertex AI initialization parameters...")
                print(f"✅ API Endpoint: {os.environ.get('VERTEXAI_API_ENDPOINT', 'NOT SET')}")
                print(f"✅ API Transport: rest (hardcoded)")
                print(f"✅ Project: {parser.vision_model.project_id}")
                print(f"✅ Location: {parser.vision_model.location}")
                
        else:
            print("❌ Vision model not initialized")
            
    except Exception as e:
        print(f"❌ Error testing VisionParser: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 5. Summary
    print("\n5. CONFIGURATION SUMMARY:")
    print("-" * 40)
    print("The VertexVisionAI model receives configuration through:")
    print("1. Constructor parameters: model_name, project_id, location")
    print("2. **kwargs passed through the factory or direct instantiation")
    print("3. Environment variables: PROJECT_ID, VERTEXAI_API_ENDPOINT, etc.")
    print("4. System configuration via ConfigManager (vision section)")
    print("\nKey parameters passed to vertexai.init():")
    print("- project: from project_id parameter or PROJECT_ID env var")
    print("- location: from location parameter (default: us-central1)")
    print("- api_transport: 'rest' (hardcoded)")
    print("- api_endpoint: from VERTEXAI_API_ENDPOINT env var")
    print("- credentials: from UniversalAuthManager")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(verify_vertex_vision_config())
