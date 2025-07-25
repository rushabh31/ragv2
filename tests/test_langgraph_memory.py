#!/usr/bin/env python3
"""
Simple test script for LangGraph memory implementation.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_langgraph_memory():
    """Test the LangGraph memory implementation."""
    try:
        from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
        
        logger.info("Testing LangGraph memory implementation...")
        
        # Test configuration
        config = {
            "type": "langgraph",
            "store_type": "in_memory",
            "embedding_dimensions": 384,
            "max_history": 10
        }
        
        # Create memory instance
        memory = MemoryFactory.create_memory(config)
        logger.info("✓ Memory instance created successfully")
        
        # Test session ID
        session_id = "test_session_123"
        
        # Test adding conversation
        success = await memory.add(
            session_id=session_id,
            query="What is artificial intelligence?",
            response="Artificial intelligence is the simulation of human intelligence in machines.",
            metadata={"test": True, "topic": "AI"}
        )
        
        if success:
            logger.info("✓ Conversation added successfully")
        else:
            logger.error("✗ Failed to add conversation")
            return False
        
        # Test getting history
        history = await memory.get_history(session_id, limit=5)
        if len(history) > 0:
            logger.info(f"✓ Retrieved {len(history)} conversation interactions")
        else:
            logger.error("✗ No conversation history retrieved")
            return False
        
        # Test getting relevant history
        relevant_history = await memory.get_relevant_history(
            session_id=session_id,
            query="Tell me about AI",
            limit=3
        )
        logger.info(f"✓ Retrieved {len(relevant_history)} relevant interactions")
        
        # Test long-term memory
        user_data = {
            "name": "Test User",
            "preferences": ["AI", "machine learning"],
            "expertise_level": "beginner"
        }
        
        success = await memory.add_long_term_memory(
            namespace=(session_id, "user_profile"),
            key="user_info",
            data=user_data
        )
        
        if success:
            logger.info("✓ Long-term memory added successfully")
        else:
            logger.error("✗ Failed to add long-term memory")
            return False
        
        # Test retrieving long-term memory
        retrieved_data = await memory.get_long_term_memory(
            namespace=(session_id, "user_profile"),
            key="user_info"
        )
        
        if retrieved_data and retrieved_data.get("name") == "Test User":
            logger.info("✓ Long-term memory retrieved successfully")
        else:
            logger.error("✗ Failed to retrieve long-term memory")
            return False
        
        # Test clearing session
        success = await memory.clear_session(session_id)
        if success:
            logger.info("✓ Session cleared successfully")
        else:
            logger.error("✗ Failed to clear session")
            return False
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}", exc_info=True)
        return False

async def test_memory_factory():
    """Test the memory factory with different memory types."""
    try:
        from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
        
        logger.info("Testing memory factory...")
        
        # Test supported types
        supported_types = MemoryFactory.get_supported_types()
        logger.info(f"✓ Supported memory types: {supported_types}")
        
        # Test creating different memory types
        test_configs = [
            {"type": "simple", "max_history": 5},
            {"type": "langgraph", "store_type": "in_memory", "max_history": 5}
        ]
        
        for config in test_configs:
            try:
                memory = MemoryFactory.create_memory(config)
                logger.info(f"✓ Successfully created {config['type']} memory")
                
                # Test basic functionality
                session_id = f"test_{config['type']}"
                success = await memory.add(
                    session_id=session_id,
                    query="Test query",
                    response="Test response"
                )
                
                if success:
                    logger.info(f"✓ {config['type']} memory add operation successful")
                else:
                    logger.error(f"✗ {config['type']} memory add operation failed")
                    
            except Exception as e:
                logger.error(f"✗ Failed to test {config['type']} memory: {str(e)}")
        
        logger.info("✓ Memory factory tests completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory factory test failed: {str(e)}", exc_info=True)
        return False

async def main():
    """Run all tests."""
    logger.info("Starting LangGraph memory tests...")
    
    # Test 1: LangGraph memory functionality
    test1_passed = await test_langgraph_memory()
    
    # Test 2: Memory factory
    test2_passed = await test_memory_factory()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 