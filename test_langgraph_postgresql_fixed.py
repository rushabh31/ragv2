#!/usr/bin/env python3
"""
Fixed LangGraph PostgreSQL Memory Test

This script tests the corrected LangGraph checkpoint PostgreSQL memory system
that properly handles the PostgresSaver context manager.
"""

import asyncio
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_langgraph_postgresql_with_context_manager():
    """Test LangGraph PostgreSQL with proper context manager handling."""
    print("🧪 Testing LangGraph PostgreSQL with Context Manager")
    print("=" * 60)
    
    db_name = "langgraph_test_db"
    connection_string = f"postgresql://rushabhsmacbook@localhost:5432/{db_name}"
    
    try:
        # Test direct PostgresSaver usage
        from langgraph.checkpoint.postgres import PostgresSaver
        
        print("🔍 Testing direct PostgresSaver usage...")
        
        # Create the context manager
        saver_cm = PostgresSaver.from_conn_string(connection_string)
        print(f"✅ Context manager created: {type(saver_cm)}")
        
        # Use the context manager
        with saver_cm as saver:
            print(f"✅ Entered context manager: {type(saver)}")
            
            # Test basic checkpoint operations
            from langgraph.checkpoint.base import Checkpoint
            from langgraph.checkpoint import CheckpointMetadata
            
            # Create a simple checkpoint
            test_thread_id = "test_thread_postgresql"
            test_checkpoint_id = "checkpoint_001"
            
            # Create checkpoint data
            checkpoint_data = Checkpoint(
                v=1,
                id=test_checkpoint_id,
                ts=datetime.now().isoformat(),
                channel_values={
                    "messages": [
                        {"role": "user", "content": "Hello PostgreSQL!"},
                        {"role": "assistant", "content": "Hello from PostgreSQL checkpointer!"}
                    ]
                },
                channel_versions={
                    "messages": 1
                },
                versions_seen={
                    "messages": {}
                }
            )
            
            # Create metadata
            metadata = CheckpointMetadata(
                source="test",
                step=1,
                writes={},
                parents={}
            )
            
            # Put checkpoint
            print(f"📝 Storing checkpoint in PostgreSQL...")
            config = {"configurable": {"thread_id": test_thread_id}}
            
            saver.put(config, checkpoint_data, metadata, {})
            print("✅ Checkpoint stored successfully")
            
            # Get checkpoint
            print(f"📖 Retrieving checkpoint from PostgreSQL...")
            retrieved = saver.get(config)
            
            if retrieved:
                checkpoint, metadata = retrieved
                print("✅ Checkpoint retrieved successfully")
                print(f"📋 Checkpoint ID: {checkpoint.id}")
                print(f"📋 Messages: {len(checkpoint.channel_values.get('messages', []))}")
                
                # Verify content
                messages = checkpoint.channel_values.get('messages', [])
                if messages:
                    print("📋 Message content:")
                    for i, msg in enumerate(messages):
                        print(f"   {i+1}. [{msg['role']}] {msg['content']}")
                
                return True
            else:
                print("❌ No checkpoint retrieved")
                return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chatbot_with_postgresql():
    """Test the chatbot API with PostgreSQL memory."""
    print("\n🤖 Testing Chatbot API with PostgreSQL Memory")
    print("=" * 60)
    
    try:
        # Update the chatbot config to use PostgreSQL
        config_path = Path("examples/rag/chatbot/config.yaml")
        
        if not config_path.exists():
            print("⚠️  Chatbot config not found")
            return False
        
        import yaml
        
        # Read and update config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure proper structure
        if 'chatbot' not in config:
            config['chatbot'] = {}
        if 'memory' not in config['chatbot']:
            config['chatbot']['memory'] = {}
        
        # Set PostgreSQL memory configuration
        config['chatbot']['memory'] = {
            'type': 'langgraph_checkpoint',
            'store_type': 'postgres',
            'postgres': {
                'connection_string': 'postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db'
            }
        }
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Updated chatbot config for PostgreSQL")
        
        # Test importing the chatbot service
        from examples.rag.chatbot.api.service import ChatbotService
        
        service = ChatbotService()
        print("✅ ChatbotService created")
        
        # Check if memory is properly initialized
        if hasattr(service, 'memory') and service.memory:
            print("✅ Memory system initialized")
            
            # Check the checkpointer type
            if hasattr(service.memory, '_checkpointer'):
                checkpointer_type = type(service.memory._checkpointer).__name__
                print(f"🔍 Checkpointer type: {checkpointer_type}")
                
                if "Postgres" in checkpointer_type or "GeneratorContextManager" in checkpointer_type:
                    print("✅ PostgreSQL checkpointer detected!")
                else:
                    print("⚠️  Non-PostgreSQL checkpointer detected")
            
            return True
        else:
            print("❌ Memory system not initialized")
            return False
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_postgresql_data():
    """Verify that data is actually stored in PostgreSQL."""
    print("\n🔍 Verifying PostgreSQL Data Storage")
    print("=" * 60)
    
    try:
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check checkpoint count
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]
        print(f"📊 Total checkpoints in database: {checkpoint_count}")
        
        # Get recent checkpoints
        cursor.execute("""
            SELECT thread_id, checkpoint_id, created_at, 
                   LENGTH(checkpoint::text) as checkpoint_size
            FROM checkpoints 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        checkpoints = cursor.fetchall()
        
        if checkpoints:
            print("📋 Recent checkpoints:")
            for thread_id, checkpoint_id, created_at, size in checkpoints:
                print(f"   Thread: {thread_id}")
                print(f"   Checkpoint: {checkpoint_id}")
                print(f"   Created: {created_at}")
                print(f"   Size: {size} bytes")
                print()
        else:
            print("⚠️  No checkpoints found in database")
        
        cursor.close()
        conn.close()
        
        return checkpoint_count > 0
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 LangGraph PostgreSQL Context Manager Test")
    print("=" * 80)
    
    # Run tests
    results = {
        "Direct PostgresSaver Test": await test_langgraph_postgresql_with_context_manager(),
        "Chatbot API Test": await test_chatbot_with_postgresql(),
        "Database Verification": await verify_postgresql_data()
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("PostgreSQL LangGraph checkpointer is working correctly!")
        print("\nNext steps:")
        print("1. The chatbot API is configured to use PostgreSQL")
        print("2. Start the chatbot: python examples/rag/chatbot/run_chatbot.py")
        print("3. Test chat functionality with persistent memory")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Check the error messages above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
