#!/usr/bin/env python3
"""
Example script demonstrating LangGraph memory usage in the RAG system.

This script shows how to:
1. Use in-memory LangGraph store for conversation history
2. Use PostgreSQL LangGraph store for persistent memory
3. Add and retrieve conversation history
4. Use semantic search for relevant conversation history
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_in_memory_memory():
    """Demonstrate in-memory LangGraph memory."""
    logger.info("=== In-Memory LangGraph Memory Demo ===")
    
    from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
    
    # Configure in-memory memory
    config = {
        "type": "langgraph",
        "store_type": "in_memory",
        "embedding_dimensions": 384,
        "max_history": 10
    }
    
    # Create memory instance
    memory = MemoryFactory.create_memory(config)
    
    # Session ID for this demo
    session_id = "demo_session_1"
    
    # Add some conversation interactions
    conversations = [
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."),
        ("Can you explain neural networks?", "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information."),
        ("What's the difference between supervised and unsupervised learning?", "Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds hidden patterns in unlabeled data."),
        ("How does deep learning work?", "Deep learning uses multiple layers of neural networks to automatically learn hierarchical representations of data.")
    ]
    
    # Add conversations to memory
    for query, response in conversations:
        success = await memory.add(
            session_id=session_id,
            query=query,
            response=response,
            metadata={"demo": True}
        )
        logger.info(f"Added conversation: {success}")
    
    # Get conversation history
    history = await memory.get_history(session_id, limit=5)
    logger.info(f"Retrieved {len(history)} conversation interactions")
    
    # Get relevant history for a specific query
    relevant_history = await memory.get_relevant_history(
        session_id=session_id,
        query="Tell me about neural networks",
        limit=3
    )
    logger.info(f"Found {len(relevant_history)} relevant interactions")
    
    # Add long-term memory
    user_preferences = {
        "preferred_topics": ["AI", "machine learning", "data science"],
        "communication_style": "technical but accessible",
        "expertise_level": "intermediate"
    }
    
    success = await memory.add_long_term_memory(
        namespace=(session_id, "preferences"),
        key="user_preferences",
        data=user_preferences
    )
    logger.info(f"Added long-term memory: {success}")
    
    # Retrieve long-term memory
    preferences = await memory.get_long_term_memory(
        namespace=(session_id, "preferences"),
        key="user_preferences"
    )
    logger.info(f"Retrieved preferences: {preferences}")
    
    # Search long-term memory
    search_results = await memory.search_long_term_memory(
        namespace=(session_id, "preferences"),
        query="technical communication",
        limit=5
    )
    logger.info(f"Search results: {len(search_results)} items found")
    
    return memory

async def demo_postgres_memory():
    """Demonstrate PostgreSQL LangGraph memory (requires PostgreSQL setup)."""
    logger.info("=== PostgreSQL LangGraph Memory Demo ===")
    
    from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
    
    # Configure PostgreSQL memory
    config = {
        "type": "langgraph",
        "store_type": "postgres",
        "embedding_dimensions": 384,
        "max_history": 10,
        "postgres": {
            "connection_string": "postgresql://user:password@localhost:5432/rag_memory"
        }
    }
    
    try:
        # Create memory instance
        memory = MemoryFactory.create_memory(config)
        
        # Session ID for this demo
        session_id = "demo_session_2"
        
        # Add conversation
        success = await memory.add(
            session_id=session_id,
            query="What is PostgreSQL?",
            response="PostgreSQL is a powerful, open-source object-relational database system with over 30 years of active development.",
            metadata={"demo": True, "topic": "databases"}
        )
        logger.info(f"Added conversation to PostgreSQL: {success}")
        
        # Get history
        history = await memory.get_history(session_id)
        logger.info(f"Retrieved {len(history)} interactions from PostgreSQL")
        
        return memory
        
    except Exception as e:
        logger.warning(f"PostgreSQL demo skipped (likely not configured): {str(e)}")
        return None

async def demo_memory_factory():
    """Demonstrate the memory factory with different memory types."""
    logger.info("=== Memory Factory Demo ===")
    
    from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
    
    # Show supported memory types
    supported_types = MemoryFactory.get_supported_types()
    logger.info(f"Supported memory types: {supported_types}")
    
    # Test creating different memory types
    memory_configs = [
        {"type": "simple", "max_history": 10},
        {"type": "langgraph", "store_type": "in_memory", "max_history": 10}
    ]
    
    for config in memory_configs:
        try:
            memory = MemoryFactory.create_memory(config)
            logger.info(f"Successfully created {config['type']} memory")
            
            # Test basic functionality
            session_id = f"test_session_{config['type']}"
            success = await memory.add(
                session_id=session_id,
                query="Test query",
                response="Test response"
            )
            logger.info(f"Test add operation: {success}")
            
        except Exception as e:
            logger.error(f"Failed to create {config['type']} memory: {str(e)}")

async def main():
    """Run all memory demos."""
    logger.info("Starting LangGraph Memory Demo")
    
    # Demo 1: In-memory memory
    await demo_in_memory_memory()
    
    # Demo 2: PostgreSQL memory (if configured)
    await demo_postgres_memory()
    
    # Demo 3: Memory factory
    await demo_memory_factory()
    
    logger.info("Demo completed!")

if __name__ == "__main__":
    asyncio.run(main()) 