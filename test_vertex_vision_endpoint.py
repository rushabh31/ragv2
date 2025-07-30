#!/usr/bin/env python3
"""
Test script to verify VertexVisionAI is using the correct API endpoint.
"""

import os
import asyncio
import logging

# Set up logging to see the debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_vertex_vision_endpoint():
    """Test VertexVisionAI endpoint configuration."""
    
    print("=" * 60)
    print("TESTING VERTEX VISION AI ENDPOINT CONFIGURATION")
    print("=" * 60)
    
    # Show current environment variable
    api_endpoint = os.environ.get("VERTEXAI_API_ENDPOINT")
    print(f"VERTEXAI_API_ENDPOINT environment variable: {api_endpoint}")
    
    if not api_endpoint:
        print("❌ VERTEXAI_API_ENDPOINT is not set!")
        print("Please set it with: export VERTEXAI_API_ENDPOINT=your-custom-endpoint")
        return
    
    print(f"✅ VERTEXAI_API_ENDPOINT is set to: {api_endpoint}")
    
    # Test 1: Direct instantiation
    print("\n1. TESTING DIRECT INSTANTIATION:")
    print("-" * 40)
    
    try:
        from src.models.vision.vertex_vision import VertexVisionAI
        
        print("Creating VertexVisionAI with default parameters...")
        model = VertexVisionAI()
        
        print("Initializing model (this will show endpoint logs)...")
        await model._ensure_initialized()
        
        print("✅ Model initialized successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Factory instantiation
    print("\n2. TESTING FACTORY INSTANTIATION:")
    print("-" * 40)
    
    try:
        from src.models.vision import VisionModelFactory
        
        print("Creating model via VisionModelFactory...")
        factory_model = VisionModelFactory.create_model("vertex_ai")
        
        print("Initializing factory model (this will show endpoint logs)...")
        await factory_model._ensure_initialized()
        
        print("✅ Factory model initialized successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 3: With custom API endpoint in config
    print("\n3. TESTING WITH CUSTOM API ENDPOINT IN CONFIG:")
    print("-" * 40)
    
    try:
        custom_endpoint = "https://custom-vertex-endpoint.googleapis.com"
        print(f"Creating model with custom api_endpoint in config: {custom_endpoint}")
        
        custom_model = VertexVisionAI(
            api_endpoint=custom_endpoint
        )
        
        print("Initializing custom model (this will show endpoint logs)...")
        await custom_model._ensure_initialized()
        
        print("✅ Custom model initialized successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 4: VisionParser integration
    print("\n4. TESTING VISION PARSER INTEGRATION:")
    print("-" * 40)
    
    try:
        from src.rag.ingestion.parsers.vision_parser import VisionParser
        
        print("Creating VisionParser...")
        parser = VisionParser({"model": "gemini-1.5-pro-002"})
        
        print("Initializing vision model through parser...")
        await parser._init_vision_model()
        
        if parser.vision_model:
            print("✅ VisionParser model initialized successfully")
            print(f"Model type: {type(parser.vision_model)}")
        else:
            print("❌ VisionParser model not initialized")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("Check the logs above to see which API endpoint is being used.")
    print("You should see 'Using custom Vertex AI API endpoint: ...' if working correctly.")

if __name__ == "__main__":
    asyncio.run(test_vertex_vision_endpoint())
