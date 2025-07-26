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
        
        # Test adding messages with different SOEIDs
        print("\n📝 Step 2: Adding test messages with different SOEIDs...")
        session_id1 = "test-session-123"
        session_id2 = "test-session-456"
        soeid1 = "test-user-456"
        soeid2 = "test-user-789"
        
        metadata1 = {"soeid": soeid1, "test": "debug"}
        metadata2 = {"soeid": soeid2, "test": "debug"}
        
        # Add messages for first user
        result1 = await memory.add(
            session_id=session_id1,
            query="Hello, this is user 456",
            response="Hi user 456!",
            metadata=metadata1
        )
        print(f"✅ Added interaction for user 456: {result1}")
        
        result2 = await memory.add(
            session_id=session_id1,
            query="How are you?",
            response="I'm doing well, thank you!",
            metadata=metadata1
        )
        print(f"✅ Added second interaction for user 456: {result2}")
        
        # Add messages for second user
        result3 = await memory.add(
            session_id=session_id2,
            query="Hello, this is user 789",
            response="Hi user 789!",
            metadata=metadata2
        )
        print(f"✅ Added interaction for user 789: {result3}")
        
        # Test retrieving messages by session
        print("\n📝 Step 3: Retrieving messages by session...")
        session1_messages = await memory.get_history(session_id1)
        print(f"📊 Retrieved {len(session1_messages)} messages for session {session_id1}")
        for i, msg in enumerate(session1_messages):
            print(f"   Message {i+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:50]}...")
            print(f"              SOEID: {msg.get('soeid', 'missing')}")
        
        session2_messages = await memory.get_history(session_id2)
        print(f"📊 Retrieved {len(session2_messages)} messages for session {session_id2}")
        for i, msg in enumerate(session2_messages):
            print(f"   Message {i+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:50]}...")
            print(f"              SOEID: {msg.get('soeid', 'missing')}")
        
        # Test retrieving messages by SOEID - Test both users
        print("\n📝 Step 4: Retrieving messages by SOEID...")
        print(f"Testing SOEID filtering for user {soeid1}:")
        soeid1_sessions = await memory.get_user_history_by_soeid(soeid1)
        print(f"📊 Retrieved {len(soeid1_sessions)} sessions for SOEID {soeid1}")
        for i, session in enumerate(soeid1_sessions):
            print(f"   Session {i+1}: {session.get('session_id', 'unknown')} with {session.get('message_count', 0)} messages")
            messages = session.get('messages', [])
            for j, msg in enumerate(messages):
                print(f"      Message {j+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:30]}...")
        
        print(f"\nTesting SOEID filtering for user {soeid2}:")
        soeid2_sessions = await memory.get_user_history_by_soeid(soeid2)
        print(f"📊 Retrieved {len(soeid2_sessions)} sessions for SOEID {soeid2}")
        for i, session in enumerate(soeid2_sessions):
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
        try:
            # Test direct checkpoint access
            config = await memory._get_thread_config(session_id1)
            print(f"✅ Thread config retrieved: {config}")
            
            # Test checkpoint listing
            checkpoints = memory._checkpointer.alist(config)
            checkpoint_list = [cp async for cp in checkpoints]
            print(f"✅ Found {len(checkpoint_list)} checkpoints")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_functionality())
