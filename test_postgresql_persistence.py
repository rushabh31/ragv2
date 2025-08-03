#!/usr/bin/env python3
"""
Test script to verify PostgreSQL persistence in LangGraphCheckpointMemory.
This script tests that chat history persists across application restarts.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_postgresql_persistence():
    """Test PostgreSQL persistence across memory system restarts."""
    
    # Configuration for PostgreSQL
    config = {
        "store_type": "postgres",
        "postgres": {
            "connection_string": "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        }
    }
    
    test_session_id = "test_persistence_session_123"
    test_soeid = "test_user_persistence"
    
    print("🧪 Testing PostgreSQL Persistence for LangGraphCheckpointMemory")
    print("=" * 60)
    
    # Phase 1: Create memory system and add messages
    print("\n📝 Phase 1: Adding messages to PostgreSQL...")
    memory1 = LangGraphCheckpointMemory(config)
    
    # Add some test messages
    test_messages = [
        ("user", "Hello, this is my first message"),
        ("assistant", "Hello! I'm here to help you."),
        ("user", "Can you remember this conversation after restart?"),
        ("assistant", "Yes, I should be able to remember our conversation because it's stored in PostgreSQL."),
        ("user", "Great! Let's test this.")
    ]
    
    for i, (role, content) in enumerate(test_messages):
        metadata = {
            "soeid": test_soeid,
            "message_id": f"msg_{i+1}",
            "timestamp": datetime.now().isoformat()
        }
        
        success = await memory1.add(test_session_id, content, metadata)
        if success:
            print(f"✅ Added {role} message: {content[:50]}...")
        else:
            print(f"❌ Failed to add {role} message")
    
    # Retrieve messages from first memory instance
    messages_before = await memory1.get_history(test_session_id)
    print(f"\n📊 Messages stored in first instance: {len(messages_before)}")
    for i, msg in enumerate(messages_before):
        print(f"  {i+1}. [{msg.get('role', 'unknown')}]: {msg.get('content', '')[:50]}...")
    
    # Clean up first memory instance
    memory1.cleanup()
    del memory1
    print("\n🗑️  First memory instance cleaned up and deleted")
    
    # Phase 2: Create new memory system and retrieve messages
    print("\n🔄 Phase 2: Creating new memory instance (simulating app restart)...")
    memory2 = LangGraphCheckpointMemory(config)
    
    # Retrieve messages from second memory instance (after "restart")
    messages_after = await memory2.get_history(test_session_id)
    print(f"\n📊 Messages retrieved after restart: {len(messages_after)}")
    for i, msg in enumerate(messages_after):
        print(f"  {i+1}. [{msg.get('role', 'unknown')}]: {msg.get('content', '')[:50]}...")
    
    # Phase 3: Verify persistence
    print("\n🔍 Phase 3: Verifying persistence...")
    
    if len(messages_after) == len(messages_before):
        print("✅ PERSISTENCE TEST PASSED: Same number of messages retrieved after restart")
        
        # Verify content matches
        content_matches = True
        for i, (before, after) in enumerate(zip(messages_before, messages_after)):
            if before.get('content') != after.get('content'):
                print(f"❌ Content mismatch at message {i+1}")
                content_matches = False
        
        if content_matches:
            print("✅ CONTENT VERIFICATION PASSED: All message content matches")
        else:
            print("❌ CONTENT VERIFICATION FAILED: Some message content doesn't match")
    else:
        print(f"❌ PERSISTENCE TEST FAILED: Expected {len(messages_before)} messages, got {len(messages_after)}")
    
    # Phase 4: Test SOEID-based retrieval
    print("\n👤 Phase 4: Testing SOEID-based retrieval...")
    soeid_history = await memory2.get_user_history_by_soeid(test_soeid)
    print(f"📊 Messages retrieved by SOEID: {len(soeid_history)}")
    
    if len(soeid_history) > 0:
        print("✅ SOEID RETRIEVAL PASSED: Successfully retrieved messages by SOEID")
    else:
        print("❌ SOEID RETRIEVAL FAILED: No messages found by SOEID")
    
    # Clean up second memory instance
    memory2.cleanup()
    del memory2
    print("\n🗑️  Second memory instance cleaned up")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"   Messages before restart: {len(messages_before)}")
    print(f"   Messages after restart:  {len(messages_after)}")
    print(f"   SOEID messages:          {len(soeid_history)}")
    
    if len(messages_after) == len(messages_before) and len(messages_after) > 0:
        print("🎉 OVERALL RESULT: PostgreSQL persistence is WORKING! ✅")
        return True
    else:
        print("💥 OVERALL RESULT: PostgreSQL persistence is NOT WORKING! ❌")
        return False

async def main():
    """Main test function."""
    try:
        success = await test_postgresql_persistence()
        if success:
            print("\n🚀 All tests passed! PostgreSQL persistence is working correctly.")
            sys.exit(0)
        else:
            print("\n💥 Tests failed! PostgreSQL persistence needs debugging.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
