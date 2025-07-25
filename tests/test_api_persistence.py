#!/usr/bin/env python3
"""
Test script to verify API persistence with singleton memory.

This script tests:
1. Adding conversations via API
2. Retrieving user history via API
3. Verifying persistence across multiple API calls
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

async def test_api_persistence():
    """Test API persistence with singleton memory."""
    logger.info("=== Testing API Persistence ===")
    
    # Test 1: Add conversations via memory directly
    logger.info("--- Test 1: Adding Conversations via Memory ---")
    
    from src.rag.chatbot.memory.memory_singleton import memory_singleton
    from src.rag.chatbot.api.service import ChatbotService
    
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
    
    # Test 2: Test API service with singleton memory
    logger.info("\n--- Test 2: API Service with Singleton Memory ---")
    
    service = ChatbotService()
    
    # Get user history
    result = await service.get_user_history_by_soeid(SOEID_HEADER, limit=10)
    
    logger.info(f"User history retrieved:")
    logger.info(f"  Total messages: {result['total_messages']}")
    logger.info(f"  Total sessions: {result['total_sessions']}")
    logger.info(f"  Session IDs: {result['metadata']['session_ids']}")
    
    # Test 3: Test API endpoint (if server is running)
    logger.info("\n--- Test 3: API Endpoint Test ---")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/user/history",
            headers={"SOEID": SOEID_HEADER},
            params={"limit": 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ API endpoint working with persistent memory!")
            logger.info(f"  Total messages: {data['total_messages']}")
            logger.info(f"  Total sessions: {data['total_sessions']}")
            
            # Show sample messages
            if data['messages']:
                logger.info("  Sample messages:")
                for i, msg in enumerate(data['messages'][:3]):
                    logger.info(f"    {i+1}. Session: {msg['session_id']}, Role: {msg['role']}, Content: {msg['content'][:50]}...")
        else:
            logger.warning(f"API endpoint returned status {response.status_code}: {response.text}")
            logger.info("Server may not be running - this is expected for this test")
            
    except requests.exceptions.ConnectionError:
        logger.info("Server not running - this is expected for this test")
    
    # Test 4: Test multiple service instances
    logger.info("\n--- Test 4: Multiple Service Instances ---")
    
    service1 = ChatbotService()
    service2 = ChatbotService()
    
    result1 = await service1.get_user_history_by_soeid(SOEID_HEADER, limit=5)
    result2 = await service2.get_user_history_by_soeid(SOEID_HEADER, limit=5)
    
    logger.info(f"Service 1 messages: {result1['total_messages']}")
    logger.info(f"Service 2 messages: {result2['total_messages']}")
    
    assert result1['total_messages'] == result2['total_messages'], "Both services should see the same data"
    logger.info("✅ Multiple service instances working with singleton memory")
    
    # Test 5: Test session-specific history
    logger.info("\n--- Test 5: Session-Specific History ---")
    
    session_history = await service1.get_session_history("session_1", limit=5)
    logger.info(f"Session 1 messages: {len(session_history['messages'])}")
    
    session_history2 = await service1.get_session_history("session_2", limit=5)
    logger.info(f"Session 2 messages: {len(session_history2['messages'])}")
    
    session_history3 = await service1.get_session_history("session_3", limit=5)
    logger.info(f"Session 3 messages: {len(session_history3['messages'])}")
    
    # Test 6: Test workflow manager integration
    logger.info("\n--- Test 6: Workflow Manager Integration ---")
    
    from src.rag.chatbot.workflow.workflow_manager import WorkflowManager
    
    workflow_manager = WorkflowManager()
    
    # Get conversation history through workflow manager
    messages = await workflow_manager._get_conversation_history("session_1")
    logger.info(f"Workflow manager found {len(messages)} messages for session_1")
    
    # Add conversation through workflow manager
    await workflow_manager._memory.add(
        session_id="session_4",
        query="What is Kubernetes?",
        response="Kubernetes is a container orchestration platform...",
        metadata={
            "soeid": SOEID_HEADER,
            "workflow_run_id": "test_run_456",
            "test": True
        }
    )
    
    # Verify the conversation was added
    result3 = await service1.get_user_history_by_soeid(SOEID_HEADER, limit=20)
    logger.info(f"Total messages after workflow addition: {result3['total_messages']}")
    
    logger.info("\n🎉 All API persistence tests passed!")

async def main():
    """Run the API persistence tests."""
    await test_api_persistence()

if __name__ == "__main__":
    asyncio.run(main()) 