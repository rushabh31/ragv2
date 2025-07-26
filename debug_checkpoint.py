#!/usr/bin/env python3
"""Debug script to test LangGraph checkpoint API directly."""

import asyncio
import logging
import sys
import os
from datetime import datetime
import uuid

# Add the project root to the path
sys.path.insert(0, '/Users/rushabhsmacbook/Documents/controlsgenai')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_checkpoint_api():
    """Test the LangGraph checkpoint API directly."""
    
    print("🔍 Testing LangGraph Checkpoint API Directly")
    print("=" * 50)
    
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        
        # Create checkpointer
        checkpointer = InMemorySaver()
        print(f"✅ Created checkpointer: {type(checkpointer).__name__}")
        
        # Create config
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": None
            }
        }
        print(f"📋 Config: {config}")
        
        # Test 1: Try to store a simple checkpoint
        print("\n📝 Test 1: Storing simple checkpoint...")
        
        simple_checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now().isoformat(),
            "channel_values": {"test": "hello world"},
            "channel_versions": {"test": 1},
            "versions_seen": {}
        }
        
        print(f"📦 Storing checkpoint: {simple_checkpoint}")
        await checkpointer.aput(config, simple_checkpoint, {}, {})
        print("✅ Checkpoint stored successfully")
        
        # Test 2: Retrieve the checkpoint
        print("\n📝 Test 2: Retrieving checkpoint...")
        retrieved = await checkpointer.aget(config)
        print(f"📦 Retrieved: {retrieved}")
        print(f"   Type: {type(retrieved)}")
        print(f"   Has channel_values: {'channel_values' in retrieved if isinstance(retrieved, dict) else hasattr(retrieved, 'channel_values')}")
        
        if isinstance(retrieved, dict) and 'channel_values' in retrieved:
            print(f"   Channel values: {retrieved['channel_values']}")
        elif hasattr(retrieved, 'channel_values'):
            print(f"   Channel values: {retrieved.channel_values}")
        
        # Test 3: Try with messages format
        print("\n📝 Test 3: Storing checkpoint with messages...")
        
        messages = [
            {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "Hi there!", "timestamp": datetime.now().isoformat()}
        ]
        
        messages_checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now().isoformat(),
            "channel_values": {"messages": messages},
            "channel_versions": {"messages": len(messages)},
            "versions_seen": {}
        }
        
        print(f"📦 Storing messages checkpoint: {messages_checkpoint}")
        await checkpointer.aput(config, messages_checkpoint, {}, {})
        print("✅ Messages checkpoint stored successfully")
        
        # Test 4: Retrieve the messages checkpoint
        print("\n📝 Test 4: Retrieving messages checkpoint...")
        retrieved_messages = await checkpointer.aget(config)
        print(f"📦 Retrieved: {retrieved_messages}")
        
        if isinstance(retrieved_messages, dict) and 'channel_values' in retrieved_messages:
            channel_values = retrieved_messages['channel_values']
            print(f"   Channel values: {channel_values}")
            if 'messages' in channel_values:
                messages_list = channel_values['messages']
                print(f"   Found {len(messages_list)} messages:")
                for i, msg in enumerate(messages_list):
                    print(f"      {i+1}: {msg}")
        
        print("\n🎉 Checkpoint API tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_checkpoint_api())
