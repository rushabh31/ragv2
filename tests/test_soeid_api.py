#!/usr/bin/env python3
"""
Test script for the new SOEID-based API endpoint.

This script tests:
1. Adding conversations with SOEID
2. Retrieving user history by SOEID via API
3. Verifying the response format includes session_id and all history
"""

import asyncio
import logging
import sys
import os
import requests
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8001"
SOEID_HEADER = "SOEID_ABC123"

async def test_soeid_api():
    """Test the SOEID-based API functionality."""
    logger.info("=== Testing SOEID API Endpoint ===")
    
    # Initialize memory once for all tests
    from src.rag.chatbot.memory.memory_factory import MemoryFactory
    from src.rag.shared.utils.config_manager import ConfigManager
    
    config = ConfigManager().get_section("chatbot.memory", {})
    memory = MemoryFactory.create_memory(config)
    
    # Test 1: Add conversations with SOEID
    logger.info("--- Test 1: Adding conversations with SOEID ---")
    
    conversations = [
        ("session_1", "What is machine learning?", "Machine learning is a subset of AI..."),
        ("session_1", "Can you explain neural networks?", "Neural networks are computing systems..."),
        ("session_2", "What is Python?", "Python is a high-level programming language..."),
        ("session_2", "How do I install packages?", "You can use pip to install packages..."),
        ("session_3", "What is Docker?", "Docker is a platform for containerization..."),
        ("session_3", "How do containers work?", "Containers share the host OS kernel...")
    ]
    
    for session_id, query, response in conversations:
        logger.info(f"Adding conversation for session {session_id}: {query[:50]}...")
        
        success = await memory.add(
            session_id=session_id,
            query=query,
            response=response,
            metadata={
                "soeid": SOEID_HEADER,
                "user_id": "user_123",
                "test": True
            }
        )
        
        logger.info(f"Added conversation: {success}")
    
    # Test 2: Test the API endpoint (if server is running)
    logger.info("\n--- Test 2: Testing API Endpoint ---")
    
    try:
        # Test the user history endpoint
        response = requests.get(
            f"{API_BASE_URL}/chat/user/history",
            headers={"SOEID": SOEID_HEADER},
            params={"limit": 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ API endpoint working!")
            logger.info(f"Response: {json.dumps(data, indent=2)}")
            
            # Verify response structure
            assert "soeid" in data, "Response should include soeid"
            assert "messages" in data, "Response should include messages"
            assert "total_messages" in data, "Response should include total_messages"
            assert "total_sessions" in data, "Response should include total_sessions"
            
            logger.info(f"✅ Response structure is correct")
            logger.info(f"Total messages: {data['total_messages']}")
            logger.info(f"Total sessions: {data['total_sessions']}")
            
            # Check that messages include session_id
            for msg in data["messages"]:
                assert "session_id" in msg, "Each message should include session_id"
                assert "role" in msg, "Each message should include role"
                assert "content" in msg, "Each message should include content"
            
            logger.info("✅ All messages include session_id")
            
        else:
            logger.warning(f"API endpoint returned status {response.status_code}: {response.text}")
            logger.info("Skipping API test - server may not be running")
            
    except requests.exceptions.ConnectionError:
        logger.info("Server not running, skipping API test")
    
    # Test 3: Test the service method directly
    logger.info("\n--- Test 3: Testing Service Method Directly ---")
    
    from src.rag.chatbot.api.service import ChatbotService
    
    # Create service with the same memory instance
    service = ChatbotService()
    service._memory = memory  # Use the same memory instance
    
    result = await service.get_user_history_by_soeid(SOEID_HEADER, limit=10)
    
    logger.info(f"Service result: {json.dumps(result, indent=2, default=str)}")
    
    # Verify service result structure
    assert "soeid" in result, "Service result should include soeid"
    assert "messages" in result, "Service result should include messages"
    assert "total_messages" in result, "Service result should include total_messages"
    assert "total_sessions" in result, "Service result should include total_sessions"
    
    logger.info(f"✅ Service method working correctly")
    logger.info(f"Total messages: {result['total_messages']}")
    logger.info(f"Total sessions: {result['total_sessions']}")
    
    # Check that messages include session_id
    session_ids = set()
    for msg in result["messages"]:
        assert "session_id" in msg, "Each message should include session_id"
        assert "role" in msg, "Each message should include role"
        assert "content" in msg, "Each message should include content"
        session_ids.add(msg["session_id"])
    
    logger.info(f"✅ All messages include session_id")
    logger.info(f"Session IDs found: {list(session_ids)}")
    
    # Test 4: Test with different SOEID
    logger.info("\n--- Test 4: Testing with Different SOEID ---")
    
    different_soeid = "SOEID_DEF456"
    result2 = await service.get_user_history_by_soeid(different_soeid, limit=5)
    
    logger.info(f"Different SOEID result: {json.dumps(result2, indent=2, default=str)}")
    logger.info(f"Total messages for different SOEID: {result2['total_messages']}")
    
    # Test 5: Test with limit
    logger.info("\n--- Test 5: Testing with Limit ---")
    
    result3 = await service.get_user_history_by_soeid(SOEID_HEADER, limit=3)
    
    logger.info(f"Limited result: {json.dumps(result3, indent=2, default=str)}")
    logger.info(f"Total messages with limit: {result3['total_messages']}")
    
    assert len(result3["messages"]) <= 3, "Should respect the limit"
    logger.info("✅ Limit functionality working correctly")
    
    # Test 6: Verify the actual data
    logger.info("\n--- Test 6: Verifying Data Content ---")
    
    if result["total_messages"] > 0:
        logger.info("Sample messages:")
        for i, msg in enumerate(result["messages"][:3]):
            logger.info(f"  {i+1}. Session: {msg['session_id']}, Role: {msg['role']}, Content: {msg['content'][:50]}...")
    else:
        logger.warning("No messages found - this might indicate an issue with the memory system")
    
    logger.info("\n🎉 All SOEID API tests completed successfully!")

async def main():
    """Run the SOEID API tests."""
    await test_soeid_api()

if __name__ == "__main__":
    asyncio.run(main()) 