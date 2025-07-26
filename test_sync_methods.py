#!/usr/bin/env python3
"""
Test PostgreSQL Checkpointer Sync Methods

This script tests the sync methods of the PostgreSQL checkpointer.
"""

import asyncio
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import Checkpoint
from datetime import datetime
import uuid

async def test_sync_methods():
    """Test PostgreSQL checkpointer sync methods."""
    print("🔍 Testing PostgreSQL Checkpointer Sync Methods")
    print("=" * 60)
    
    connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
    
    try:
        # Create the context manager
        postgres_cm = PostgresSaver.from_conn_string(connection_string)
        
        # Enter the context manager
        with postgres_cm as saver:
            print(f"✅ Entered context: {type(saver)}")
            
            # Set up schema
            saver.setup()
            print("✅ Schema setup completed")
            
            # Test basic sync operations
            config = {"configurable": {"thread_id": "test_sync_thread"}}
            
            # Test sync get (should return None for non-existent)
            print("\n🧪 Testing sync get...")
            try:
                result = saver.get(config)
                print(f"   ✅ sync get result: {result}")
            except Exception as e:
                print(f"   ❌ sync get failed: {e}")
                return False
            
            # Create a simple checkpoint for testing
            print("\n🧪 Testing sync put...")
            try:
                # Create checkpoint data
                checkpoint_data = Checkpoint(
                    v=1,
                    id=str(uuid.uuid4()),
                    ts=datetime.now().isoformat(),
                    channel_values={
                        "messages": [
                            {"role": "user", "content": "Hello sync test!"},
                            {"role": "assistant", "content": "Hello from sync PostgreSQL!"}
                        ]
                    },
                    channel_versions={
                        "messages": 1
                    },
                    versions_seen={
                        "messages": {}
                    }
                )
                
                # Put checkpoint using sync method
                saver.put(config, checkpoint_data, {}, {})
                print("   ✅ sync put completed")
                
                # Retrieve it back
                retrieved = saver.get(config)
                if retrieved:
                    checkpoint, metadata = retrieved
                    print(f"   ✅ sync get after put: checkpoint ID {checkpoint.id}")
                    messages = checkpoint.channel_values.get('messages', [])
                    print(f"   ✅ Retrieved {len(messages)} messages")
                    for i, msg in enumerate(messages):
                        print(f"      {i+1}. [{msg['role']}] {msg['content']}")
                else:
                    print("   ❌ No checkpoint retrieved after put")
                    return False
                
            except Exception as e:
                print(f"   ❌ sync put/get failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            print("\n✅ Sync methods working correctly!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_sync_methods())
    if success:
        print("\n🎉 PostgreSQL sync methods work correctly!")
        print("The issue might be with async method implementations.")
    else:
        print("\n❌ PostgreSQL sync methods failed!")
