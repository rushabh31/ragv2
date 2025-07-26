#!/usr/bin/env python3
"""Debug script to test LangGraph checkpoint memory functionality."""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/rushabhsmacbook/Documents/controlsgenai')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_memory_functionality():
    """Test the LangGraph checkpoint memory functionality step by step."""
    
    print("🔍 Testing LangGraph Checkpoint Memory Functionality")
    print("=" * 60)
    
    try:
        # Import the memory class
        from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
        
        # Create memory instance with in-memory storage
        config = {
            "type": "langgraph_checkpoint",
            "store_type": "in_memory"
        }
        
        print("📝 Step 1: Creating memory instance...")
        memory = LangGraphCheckpointMemory(config)
        print(f"✅ Memory instance created: {type(memory).__name__}")
        print(f"   Store type: {memory._store_type}")
        print(f"   Checkpointer type: {type(memory._checkpointer).__name__}")
        
        # Test adding a message
        print("\n📝 Step 2: Adding test messages...")
        session_id = "test-session-123"
        soeid = "test-user-456"
        
        metadata = {"soeid": soeid, "test": "debug"}
        
        # Add user message
        result1 = await memory.add(
            session_id=session_id,
            query="Hello, this is a test message",
            response="Hi! This is a test response",
            metadata=metadata
        )
        print(f"✅ Added interaction: {result1}")
        
        # Add another message
        result2 = await memory.add(
            session_id=session_id,
            query="How are you?",
            response="I'm doing well, thank you!",
            metadata=metadata
        )
        print(f"✅ Added second interaction: {result2}")
        
        # Test retrieving messages by session
        print("\n📝 Step 3: Retrieving messages by session...")
        session_messages = await memory.get_history(session_id)
        print(f"📊 Retrieved {len(session_messages)} messages for session {session_id}")
        for i, msg in enumerate(session_messages):
            print(f"   Message {i+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:50]}...")
            print(f"              SOEID: {msg.get('soeid', 'missing')}")
        
        # Test retrieving messages by SOEID
        print("\n📝 Step 4: Retrieving messages by SOEID...")
        soeid_sessions = await memory.get_user_history_by_soeid(soeid)
        print(f"📊 Retrieved {len(soeid_sessions)} sessions for SOEID {soeid}")
        for i, session in enumerate(soeid_sessions):
            print(f"   Session {i+1}: {session.get('session_id', 'unknown')} with {session.get('message_count', 0)} messages")
            messages = session.get('messages', [])
            for j, msg in enumerate(messages):
                print(f"      Message {j+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:30]}...")
        
        # Test thread listing
        print("\n📝 Step 5: Testing thread listing...")
        thread_ids = await memory._list_thread_ids()
        print(f"📊 Found {len(thread_ids)} threads: {thread_ids}")
        
        # Test memory stats
        print("\n📝 Step 6: Testing memory statistics...")
        if hasattr(memory, 'get_memory_stats'):
            stats = await memory.get_memory_stats()
            print(f"📊 Memory stats: {stats}")
        
        # Test direct checkpoint access
        print("\n📝 Step 7: Testing direct checkpoint access...")
        config = await memory._get_thread_config(session_id)
        print(f"📊 Thread config: {config}")
        
        checkpoint = await memory._checkpointer.aget(config)
        print(f"📊 Checkpoint exists: {checkpoint is not None}")
        if checkpoint:
            print(f"   Checkpoint type: {type(checkpoint)}")
            print(f"   Has channel_values: {hasattr(checkpoint, 'channel_values')}")
            if hasattr(checkpoint, 'channel_values'):
                print(f"   Channel values: {checkpoint.channel_values}")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_functionality())
