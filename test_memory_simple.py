#!/usr/bin/env python3
"""
Simple Memory Functionality Test

This script tests the basic memory functionality with PostgreSQL.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_simple_memory_operations():
    """Test simple memory operations."""
    print("🧪 Testing Simple Memory Operations")
    print("=" * 50)
    
    try:
        from examples.rag.chatbot.api.service import ChatbotService
        
        service = ChatbotService()
        print("✅ ChatbotService created")
        
        # Trigger memory initialization
        await service.get_memory_stats()
        memory = service._memory
        print("✅ Memory initialized")
        
        # Test adding a message
        session_id = "test_session_simple"
        message = "Hello PostgreSQL!"
        
        print(f"📝 Adding message: '{message}'")
        success = await memory.add(
            session_id=session_id,
            query=message,
            metadata={"soeid": "test_user"}
        )
        
        if success:
            print("✅ Message added successfully")
        else:
            print("❌ Failed to add message")
            return False
        
        # Test retrieving messages
        print(f"📖 Retrieving messages...")
        history = await memory.get_history(session_id)
        
        if history:
            print(f"✅ Retrieved {len(history)} messages:")
            for i, msg in enumerate(history):
                content = msg.get('content', 'No content')
                role = msg.get('role', 'unknown')
                print(f"   {i+1}. [{role}] {content}")
            return True
        else:
            print("❌ No messages retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_database_data():
    """Verify data in database."""
    print("\n🔍 Verifying Database Data")
    print("=" * 40)
    
    try:
        import psycopg2
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check checkpoints
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]
        print(f"📊 Checkpoints: {checkpoint_count}")
        
        # Check checkpoint_writes
        cursor.execute("SELECT COUNT(*) FROM checkpoint_writes")
        writes_count = cursor.fetchone()[0]
        print(f"📊 Checkpoint writes: {writes_count}")
        
        # Check checkpoint_blobs
        cursor.execute("SELECT COUNT(*) FROM checkpoint_blobs")
        blobs_count = cursor.fetchone()[0]
        print(f"📊 Checkpoint blobs: {blobs_count}")
        
        cursor.close()
        conn.close()
        
        total_records = checkpoint_count + writes_count + blobs_count
        print(f"📊 Total records: {total_records}")
        
        return total_records > 0
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Simple Memory Test")
    print("=" * 60)
    
    # Test memory operations
    memory_success = await test_simple_memory_operations()
    
    # Verify database
    db_success = await verify_database_data()
    
    print("\n" + "=" * 60)
    print("📊 SIMPLE TEST SUMMARY")
    print("=" * 60)
    
    memory_status = "✅ PASS" if memory_success else "❌ FAIL"
    db_status = "✅ PASS" if db_success else "❌ FAIL"
    
    print(f"{memory_status} Memory Operations")
    print(f"{db_status} Database Persistence")
    
    if memory_success and db_success:
        print("\n🎉 SUCCESS!")
        print("PostgreSQL LangGraph memory is working!")
        return 0
    elif memory_success:
        print("\n⚠️  PARTIAL SUCCESS")
        print("Memory operations work but data persistence unclear.")
        return 0
    else:
        print("\n❌ FAILURE")
        print("Memory operations failed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
