#!/usr/bin/env python3
"""
Test script for singleton memory functionality.

This script tests:
1. Singleton memory instance creation
2. Memory persistence across different service instances
3. API service using singleton memory
4. Workflow manager using singleton memory
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_singleton_memory():
    """Test singleton memory functionality."""
    logger.info("=== Testing Singleton Memory ===")
    
    from src.rag.src.chatbot.memory.memory_singleton import memory_singleton
    from src.rag.src.chatbot.api.service import ChatbotService
    from src.rag.src.chatbot.workflow.workflow_manager import WorkflowManager
    
    # Test 1: Get memory instance
    logger.info("--- Test 1: Getting Memory Instance ---")
    
    memory1 = await memory_singleton.get_memory()
    memory2 = await memory_singleton.get_memory()
    
    logger.info(f"Memory 1 ID: {id(memory1)}")
    logger.info(f"Memory 2 ID: {id(memory2)}")
    logger.info(f"Same instance: {memory1 is memory2}")
    
    assert memory1 is memory2, "Singleton memory instances should be the same"
    logger.info("✅ Singleton memory working correctly")
    
    # Test 2: Add conversations and verify persistence
    logger.info("\n--- Test 2: Memory Persistence ---")
    
    # Add conversations using the memory directly
    conversations = [
        ("session_1", "What is machine learning?", "Machine learning is a subset of AI..."),
        ("session_1", "Can you explain neural networks?", "Neural networks are computing systems..."),
        ("session_2", "What is Python?", "Python is a high-level programming language..."),
        ("session_2", "How do I install packages?", "You can use pip to install packages...")
    ]
    
    for session_id, query, response in conversations:
        success = await memory1.add(
            session_id=session_id,
            query=query,
            response=response,
            metadata={
                "soeid": "SOEID_ABC123",
                "user_id": "user_123",
                "test": True
            }
        )
        logger.info(f"Added conversation for {session_id}: {success}")
    
    # Test 3: Verify memory persistence across different instances
    logger.info("\n--- Test 3: Cross-Instance Persistence ---")
    
    # Create new service instance
    service1 = ChatbotService()
    service2 = ChatbotService()
    
    # Get user history from both services
    result1 = await service1.get_user_history_by_soeid("SOEID_ABC123", limit=10)
    result2 = await service2.get_user_history_by_soeid("SOEID_ABC123", limit=10)
    
    logger.info(f"Service 1 total messages: {result1['total_messages']}")
    logger.info(f"Service 2 total messages: {result2['total_messages']}")
    
    assert result1['total_messages'] == result2['total_messages'], "Both services should see the same data"
    logger.info("✅ Memory persistence across service instances working")
    
    # Test 4: Workflow manager using singleton memory
    logger.info("\n--- Test 4: Workflow Manager Integration ---")
    
    workflow_manager = WorkflowManager()
    
    # Get conversation history through workflow manager
    messages = await workflow_manager._get_conversation_history("session_1")
    logger.info(f"Workflow manager found {len(messages)} messages for session_1")
    
    # Test 5: Add conversation through workflow manager
    logger.info("\n--- Test 5: Workflow Manager Adding Conversation ---")
    
    # Simulate a workflow result
    await workflow_manager._memory.add(
        session_id="session_3",
        query="What is Docker?",
        response="Docker is a platform for containerization...",
        metadata={
            "soeid": "SOEID_ABC123",
            "workflow_run_id": "test_run_123",
            "test": True
        }
    )
    
    # Verify the conversation was added
    result3 = await service1.get_user_history_by_soeid("SOEID_ABC123", limit=15)
    logger.info(f"Total messages after workflow addition: {result3['total_messages']}")
    
    # Test 6: Verify session-specific history
    logger.info("\n--- Test 6: Session-Specific History ---")
    
    session_history = await service1.get_session_history("session_1", limit=5)
    logger.info(f"Session 1 messages: {len(session_history['messages'])}")
    
    session_history2 = await service1.get_session_history("session_2", limit=5)
    logger.info(f"Session 2 messages: {len(session_history2['messages'])}")
    
    session_history3 = await service1.get_session_history("session_3", limit=5)
    logger.info(f"Session 3 messages: {len(session_history3['messages'])}")
    
    # Test 7: Test memory reset functionality
    logger.info("\n--- Test 7: Memory Reset ---")
    
    await memory_singleton.reset_memory()
    
    # Reset service memory references
    await service1.reset_memory_reference()
    await service2.reset_memory_reference()
    
    # Get fresh memory instance
    memory3 = await memory_singleton.get_memory()
    logger.info(f"New memory instance ID: {id(memory3)}")
    logger.info(f"Different from original: {memory3 is not memory1}")
    
    # Verify memory is empty after reset
    result4 = await service1.get_user_history_by_soeid("SOEID_ABC123", limit=5)
    logger.info(f"Messages after reset: {result4['total_messages']}")
    
    assert result4['total_messages'] == 0, "Memory should be empty after reset"
    logger.info("✅ Memory reset functionality working")
    
    logger.info("\n🎉 All singleton memory tests passed!")

async def main():
    """Run the singleton memory tests."""
    await test_singleton_memory()

if __name__ == "__main__":
    asyncio.run(main()) 