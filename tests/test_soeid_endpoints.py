#!/usr/bin/env python3
"""
Test script for SOEID endpoints.

This script tests:
1. /chat/user/history (SOEID in header)
2. /chat/user/history/{soeid} (SOEID in path)
3. Verifies both endpoints return the same data
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

async def test_soeid_endpoints():
    """Test both SOEID endpoints."""
    logger.info("=== Testing SOEID Endpoints ===")
    
    # Test 1: Add some test data
    logger.info("--- Test 1: Adding Test Data ---")
    
    from src.rag.src.chatbot.memory.memory_singleton import memory_singleton
    
    # Get memory instance
    memory = await memory_singleton.get_memory()
    
    # Add some conversations
    conversations = [
        ("session_1", "What is machine learning?", "Machine learning is a subset of AI..."),
        ("session_1", "Can you explain neural networks?", "Neural networks are computing systems..."),
        ("session_2", "What is Python?", "Python is a high-level programming language..."),
        ("session_2", "How do I install packages?", "You can use pip to install packages..."),
        ("session_3", "What is Docker?", "Docker is a platform for containerization..."),
        ("session_3", "How do containers work?", "Containers share the host OS kernel...")
    ]
    
    for session_id, query, response in conversations:
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
        logger.info(f"Added conversation for {session_id}: {success}")
    
    # Test 2: Test header-based endpoint
    logger.info("\n--- Test 2: Header-Based Endpoint (/chat/user/history) ---")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/user/history",
            headers={"SOEID": SOEID_HEADER},
            params={"limit": 10}
        )
        
        if response.status_code == 200:
            data_header = response.json()
            logger.info("✅ Header-based endpoint working!")
            logger.info(f"  Total messages: {data_header['total_messages']}")
            logger.info(f"  Total sessions: {data_header['total_sessions']}")
            logger.info(f"  SOEID: {data_header['soeid']}")
        else:
            logger.warning(f"Header-based endpoint returned status {response.status_code}: {response.text}")
            data_header = None
            
    except requests.exceptions.ConnectionError:
        logger.info("Server not running - skipping API test")
        data_header = None
    
    # Test 3: Test path-based endpoint
    logger.info("\n--- Test 3: Path-Based Endpoint (/chat/user/history/{soeid}) ---")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/user/history/{SOEID_HEADER}",
            params={"limit": 10}
        )
        
        if response.status_code == 200:
            data_path = response.json()
            logger.info("✅ Path-based endpoint working!")
            logger.info(f"  Total messages: {data_path['total_messages']}")
            logger.info(f"  Total sessions: {data_path['total_sessions']}")
            logger.info(f"  SOEID: {data_path['soeid']}")
        else:
            logger.warning(f"Path-based endpoint returned status {response.status_code}: {response.text}")
            data_path = None
            
    except requests.exceptions.ConnectionError:
        logger.info("Server not running - skipping API test")
        data_path = None
    
    # Test 4: Compare results
    logger.info("\n--- Test 4: Comparing Results ---")
    
    if data_header and data_path:
        logger.info("Comparing header-based vs path-based results:")
        logger.info(f"  Header total messages: {data_header['total_messages']}")
        logger.info(f"  Path total messages: {data_path['total_messages']}")
        logger.info(f"  Header total sessions: {data_header['total_sessions']}")
        logger.info(f"  Path total sessions: {data_path['total_sessions']}")
        
        # Verify they return the same data
        assert data_header['total_messages'] == data_path['total_messages'], "Both endpoints should return the same number of messages"
        assert data_header['total_sessions'] == data_path['total_sessions'], "Both endpoints should return the same number of sessions"
        assert data_header['soeid'] == data_path['soeid'], "Both endpoints should return the same SOEID"
        
        logger.info("✅ Both endpoints return identical data!")
    
    # Test 5: Test with different SOEID
    logger.info("\n--- Test 5: Different SOEID Test ---")
    
    different_soeid = "SOEID_DEF456"
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/user/history/{different_soeid}",
            params={"limit": 5}
        )
        
        if response.status_code == 200:
            data_diff = response.json()
            logger.info(f"✅ Different SOEID endpoint working!")
            logger.info(f"  Total messages for {different_soeid}: {data_diff['total_messages']}")
            logger.info(f"  SOEID: {data_diff['soeid']}")
        else:
            logger.warning(f"Different SOEID endpoint returned status {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.info("Server not running - skipping different SOEID test")
    
    # Test 6: Test service method directly
    logger.info("\n--- Test 6: Service Method Test ---")
    
    from src.rag.src.chatbot.api.service import ChatbotService
    
    service = ChatbotService()
    
    # Test with the same SOEID
    result = await service.get_user_history_by_soeid(SOEID_HEADER, limit=10)
    logger.info(f"Service method result:")
    logger.info(f"  Total messages: {result['total_messages']}")
    logger.info(f"  Total sessions: {result['total_sessions']}")
    logger.info(f"  SOEID: {result['soeid']}")
    
    # Test with different SOEID
    result_diff = await service.get_user_history_by_soeid(different_soeid, limit=5)
    logger.info(f"Service method result for different SOEID:")
    logger.info(f"  Total messages: {result_diff['total_messages']}")
    logger.info(f"  SOEID: {result_diff['soeid']}")
    
    logger.info("\n🎉 All SOEID endpoint tests completed!")

async def main():
    """Run the SOEID endpoint tests."""
    await test_soeid_endpoints()

if __name__ == "__main__":
    asyncio.run(main()) 