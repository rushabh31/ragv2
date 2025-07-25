#!/usr/bin/env python3
"""
Example script demonstrating SOEID (Source of Entity ID) usage in LangGraph memory.

This script shows how to:
1. Store conversations with SOEID for user identification
2. Retrieve user-specific conversation history by SOEID
3. Use SOEID for cross-session memory retrieval
4. Manage user-specific long-term memory
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

async def demo_soeid_memory():
    """Demonstrate SOEID-based memory functionality."""
    logger.info("=== SOEID Memory Demo ===")
    
    from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
    
    # Configure LangGraph memory
    config = {
        "type": "langgraph",
        "store_type": "in_memory",
        "embedding_dimensions": 384,
        "max_history": 20
    }
    
    # Create memory instance
    memory = MemoryFactory.create_memory(config)
    
    # Define some test users with SOEIDs
    users = {
        "user_123": "SOEID_ABC123",
        "user_456": "SOEID_DEF456",
        "user_789": "SOEID_GHI789"
    }
    
    # Simulate conversations for different users across multiple sessions
    conversations = [
        # User 1 conversations
        ("session_1", "user_123", "SOEID_ABC123", "What is machine learning?", "Machine learning is a subset of AI..."),
        ("session_2", "user_123", "SOEID_ABC123", "Can you explain neural networks?", "Neural networks are computing systems..."),
        ("session_3", "user_123", "SOEID_ABC123", "What's the difference between supervised and unsupervised?", "Supervised learning uses labeled data..."),
        
        # User 2 conversations
        ("session_4", "user_456", "SOEID_DEF456", "What is Python?", "Python is a high-level programming language..."),
        ("session_5", "user_456", "SOEID_DEF456", "How do I install packages?", "You can use pip to install packages..."),
        ("session_6", "user_456", "SOEID_DEF456", "What are virtual environments?", "Virtual environments isolate Python dependencies..."),
        
        # User 3 conversations
        ("session_7", "user_789", "SOEID_GHI789", "What is Docker?", "Docker is a platform for containerization..."),
        ("session_8", "user_789", "SOEID_GHI789", "How do containers work?", "Containers share the host OS kernel..."),
        ("session_9", "user_789", "SOEID_GHI789", "What's the difference between Docker and VMs?", "VMs include the OS, containers share it...")
    ]
    
    # Add conversations to memory
    logger.info("Adding conversations with SOEID...")
    for session_id, user_id, soeid, query, response in conversations:
        success = await memory.add(
            session_id=session_id,
            query=query,
            response=response,
            metadata={
                "soeid": soeid,
                "user_id": user_id,
                "demo": True
            }
        )
        logger.info(f"Added conversation for {user_id} (SOEID: {soeid}) in session {session_id}: {success}")
    
    # Test 1: Get user history by SOEID
    logger.info("\n--- Test 1: Get User History by SOEID ---")
    for soeid in ["SOEID_ABC123", "SOEID_DEF456", "SOEID_GHI789"]:
        history = await memory.get_user_history_by_soeid(soeid, limit=5)
        logger.info(f"User {soeid} has {len(history)} conversation interactions")
        for i, msg in enumerate(history[-3:], 1):  # Show last 3 interactions
            logger.info(f"  {i}. [{msg['role']}] {msg['content'][:50]}...")
    
    # Test 2: Get relevant history by SOEID
    logger.info("\n--- Test 2: Get Relevant History by SOEID ---")
    test_queries = [
        ("SOEID_ABC123", "Tell me about neural networks"),
        ("SOEID_DEF456", "How do I install Python packages"),
        ("SOEID_GHI789", "What are Docker containers")
    ]
    
    for soeid, query in test_queries:
        relevant_history = await memory.get_user_relevant_history_by_soeid(soeid, query, limit=3)
        logger.info(f"Relevant history for {soeid} query '{query}': {len(relevant_history)} interactions")
        for i, msg in enumerate(relevant_history, 1):
            logger.info(f"  {i}. [{msg['role']}] {msg['content'][:50]}...")
    
    # Test 3: Add user-specific long-term memory
    logger.info("\n--- Test 3: User-Specific Long-Term Memory ---")
    user_preferences = {
        "SOEID_ABC123": {
            "preferred_topics": ["AI", "machine learning", "data science"],
            "expertise_level": "intermediate",
            "communication_style": "technical"
        },
        "SOEID_DEF456": {
            "preferred_topics": ["programming", "Python", "development"],
            "expertise_level": "beginner",
            "communication_style": "simple"
        },
        "SOEID_GHI789": {
            "preferred_topics": ["DevOps", "containers", "deployment"],
            "expertise_level": "advanced",
            "communication_style": "detailed"
        }
    }
    
    for soeid, preferences in user_preferences.items():
        success = await memory.add_long_term_memory(
            namespace=(soeid, "user_profile"),
            key="preferences",
            data=preferences
        )
        logger.info(f"Added preferences for {soeid}: {success}")
    
    # Test 4: Retrieve user-specific long-term memory
    logger.info("\n--- Test 4: Retrieve User-Specific Memory ---")
    for soeid in user_preferences.keys():
        preferences = await memory.get_long_term_memory(
            namespace=(soeid, "user_profile"),
            key="preferences"
        )
        if preferences:
            logger.info(f"User {soeid} preferences: {preferences.get('preferred_topics', [])}")
    
    # Test 5: Search user-specific memory
    logger.info("\n--- Test 5: Search User-Specific Memory ---")
    for soeid in user_preferences.keys():
        search_results = await memory.search_long_term_memory(
            namespace=(soeid, "user_profile"),
            query="technical communication",
            limit=5
        )
        logger.info(f"Search results for {soeid}: {len(search_results)} items")
    
    # Test 6: Clear user history
    logger.info("\n--- Test 6: Clear User History ---")
    test_soeid = "SOEID_ABC123"
    history_before = await memory.get_user_history_by_soeid(test_soeid)
    logger.info(f"User {test_soeid} has {len(history_before)} interactions before clearing")
    
    success = await memory.clear_user_history(test_soeid)
    logger.info(f"Cleared user history for {test_soeid}: {success}")
    
    history_after = await memory.get_user_history_by_soeid(test_soeid)
    logger.info(f"User {test_soeid} has {len(history_after)} interactions after clearing")
    
    return memory

async def demo_soeid_workflow_integration():
    """Demonstrate how SOEID integrates with the workflow."""
    logger.info("\n=== SOEID Workflow Integration Demo ===")
    
    # Simulate workflow state with SOEID
    workflow_state = {
        "query": "What is machine learning?",
        "session_id": "workflow_session_1",
        "soeid": "SOEID_ABC123",
        "user_id": "user_123",
        "messages": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "response": "Machine learning is a subset of artificial intelligence...",
        "metrics": {}
    }
    
    logger.info(f"Workflow state includes SOEID: {workflow_state.get('soeid')}")
    logger.info(f"Query: {workflow_state.get('query')}")
    logger.info(f"Session: {workflow_state.get('session_id')}")
    
    # Simulate memory update with SOEID
    from src.rag.src.chatbot.memory.memory_factory import MemoryFactory
    from src.rag.src.shared.utils.config_manager import ConfigManager
    
    config = ConfigManager().get_section("chatbot.memory", {})
    memory = MemoryFactory.create_memory(config)
    
    metadata = {
        "soeid": workflow_state.get("soeid"),
        "workflow_run_id": "run_123",
        "retrieved_documents_count": 3,
        "reranked_documents_count": 2
    }
    
    success = await memory.add(
        session_id=workflow_state.get("session_id"),
        query=workflow_state.get("query"),
        response=workflow_state.get("response"),
        metadata=metadata
    )
    
    logger.info(f"Added workflow interaction to memory: {success}")
    
    # Retrieve user history by SOEID
    user_history = await memory.get_user_history_by_soeid(
        soeid=workflow_state.get("soeid"),
        limit=5
    )
    
    logger.info(f"Retrieved {len(user_history)} interactions for user {workflow_state.get('soeid')}")

async def main():
    """Run SOEID memory demos."""
    logger.info("Starting SOEID Memory Demo")
    
    # Demo 1: Basic SOEID functionality
    await demo_soeid_memory()
    
    # Demo 2: Workflow integration
    await demo_soeid_workflow_integration()
    
    logger.info("SOEID Demo completed!")

if __name__ == "__main__":
    asyncio.run(main()) 