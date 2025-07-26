#!/usr/bin/env python3
"""
Debug script to identify vision parser timeout issues.
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.vision.vertex_vision import VertexVisionAI

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_step_by_step():
    """Test each step of vision parser initialization and execution."""
    
    logger.info("🔍 Starting Vision Parser Timeout Diagnosis")
    logger.info("=" * 60)
    
    # Step 1: Check environment variables
    logger.info("\n📋 Step 1: Environment Variables Check")
    required_vars = [
        "COIN_CONSUMER_ENDPOINT_URL",
        "COIN_CONSUMER_CLIENT_ID", 
        "COIN_CONSUMER_CLIENT_SECRET",
        "PROJECT_ID",
        "VERTEXAI_API_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"✅ {var}: {'*' * min(len(value), 20)}...")
        else:
            logger.error(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    # Step 2: Test Vision Model Creation
    logger.info("\n🏗️  Step 2: Vision Model Creation")
    try:
        start_time = time.time()
        vision_model = VertexVisionAI(model_name="gemini-1.5-pro-002")
        creation_time = time.time() - start_time
        logger.info(f"✅ Model created successfully in {creation_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Model creation failed: {str(e)}")
        return False
    
    # Step 3: Test Authentication Manager
    logger.info("\n🔐 Step 3: Authentication Manager Test")
    try:
        start_time = time.time()
        auth_health = vision_model.auth_manager.get_health_status()
        auth_time = time.time() - start_time
        logger.info(f"✅ Auth manager health check: {auth_health} ({auth_time:.2f}s)")
    except Exception as e:
        logger.error(f"❌ Auth manager failed: {str(e)}")
        return False
    
    # Step 4: Test Credential Acquisition (with timeout)
    logger.info("\n🎫 Step 4: Credential Acquisition Test")
    try:
        start_time = time.time()
        # Add timeout to credential acquisition
        credentials = await asyncio.wait_for(
            vision_model.auth_manager.get_credentials(),
            timeout=30.0  # 30 second timeout
        )
        cred_time = time.time() - start_time
        logger.info(f"✅ Credentials acquired in {cred_time:.2f}s")
        logger.info(f"   Credential type: {type(credentials)}")
    except asyncio.TimeoutError:
        logger.error("❌ Credential acquisition timed out after 30 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Credential acquisition failed: {str(e)}")
        return False
    
    # Step 5: Test Model Initialization
    logger.info("\n🚀 Step 5: Model Initialization Test")
    try:
        start_time = time.time()
        await asyncio.wait_for(
            vision_model._ensure_initialized(),
            timeout=60.0  # 60 second timeout
        )
        init_time = time.time() - start_time
        logger.info(f"✅ Model initialized successfully in {init_time:.2f}s")
    except asyncio.TimeoutError:
        logger.error("❌ Model initialization timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {str(e)}")
        return False
    
    # Step 6: Test Simple Vision Call (with sample base64 image)
    logger.info("\n👁️  Step 6: Simple Vision API Test")
    try:
        # Create a simple test image (1x1 pixel PNG in base64)
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        start_time = time.time()
        result = await asyncio.wait_for(
            vision_model.parse_text_from_image(
                base64_encoded=test_image_b64,
                prompt="Describe this image briefly.",
                timeout=30  # 30 second timeout for API call
            ),
            timeout=45.0  # 45 second overall timeout
        )
        api_time = time.time() - start_time
        logger.info(f"✅ Vision API call completed in {api_time:.2f}s")
        logger.info(f"   Response length: {len(result)} characters")
        logger.info(f"   Response preview: {result[:100]}...")
        
    except asyncio.TimeoutError:
        logger.error("❌ Vision API call timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Vision API call failed: {str(e)}")
        return False
    
    logger.info("\n🎉 All tests passed! Vision parser is working correctly.")
    return True

async def test_with_timeouts():
    """Test with various timeout configurations."""
    
    logger.info("\n⏱️  Testing Different Timeout Configurations")
    logger.info("-" * 50)
    
    timeout_configs = [
        {"name": "Short timeout (15s)", "timeout": 15},
        {"name": "Medium timeout (30s)", "timeout": 30}, 
        {"name": "Long timeout (60s)", "timeout": 60},
        {"name": "Very long timeout (120s)", "timeout": 120}
    ]
    
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    for config in timeout_configs:
        logger.info(f"\n🧪 Testing {config['name']}")
        
        try:
            vision_model = VertexVisionAI(model_name="gemini-1.5-pro-002")
            
            start_time = time.time()
            result = await vision_model.parse_text_from_image(
                base64_encoded=test_image_b64,
                prompt="Describe this image briefly.",
                timeout=config['timeout']
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ Success with {config['name']}: {elapsed_time:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout with {config['name']}")
        except Exception as e:
            logger.error(f"❌ Error with {config['name']}: {str(e)}")

async def main():
    """Main diagnostic function."""
    logger.info("🩺 Vision Parser Timeout Diagnostic Tool")
    logger.info("=" * 60)
    
    # Run step-by-step diagnosis
    success = await test_step_by_step()
    
    if success:
        # If basic test passes, try different timeout configurations
        await test_with_timeouts()
    else:
        logger.error("\n❌ Basic functionality test failed.")
        logger.error("Please fix the issues above before proceeding.")
    
    logger.info("\n📊 Diagnostic Summary:")
    logger.info("If you see timeouts consistently at a specific step:")
    logger.info("• Step 4 (Credentials): Check network connectivity and auth server")
    logger.info("• Step 5 (Initialization): Check Vertex AI endpoint and project ID")
    logger.info("• Step 6 (API Call): Check Vertex AI API quotas and model availability")

if __name__ == "__main__":
    asyncio.run(main())
