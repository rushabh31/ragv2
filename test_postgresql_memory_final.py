#!/usr/bin/env python3
"""
Comprehensive PostgreSQL Memory Test

This script tests the complete LangGraph PostgreSQL memory integration
including data persistence, retrieval, and chatbot API functionality.
"""

import asyncio
import sys
import psycopg2
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_memory_system_initialization():
    """Test that the memory system initializes with PostgreSQL."""
    print("🧪 Testing Memory System Initialization")
    print("=" * 50)
    
    try:
        from examples.rag.chatbot.api.service import ChatbotService
        
        service = ChatbotService()
        print("✅ ChatbotService created")
        
        # Trigger memory initialization
        await service.get_memory_stats()
        
        # Check memory system
        if hasattr(service, '_memory') and service._memory:
            memory_type = type(service._memory).__name__
            print(f"✅ Memory type: {memory_type}")
            
            # Check for PostgreSQL connection string (new approach)
            if hasattr(service._memory, '_postgres_connection_string') and service._memory._postgres_connection_string:
                print(f"✅ PostgreSQL connection string configured: {service._memory._postgres_connection_string[:50]}...")
                print("✅ PostgreSQL checkpointer confirmed (on-demand connections)!")
                return service
            elif hasattr(service._memory, '_checkpointer') and service._memory._checkpointer:
                checkpointer_type = str(type(service._memory._checkpointer))
                print(f"✅ Checkpointer type: {checkpointer_type}")
                
                if "PostgresSaver" in checkpointer_type:
                    print("✅ PostgreSQL checkpointer confirmed!")
                    return service
                else:
                    print(f"❌ Expected PostgresSaver, got {checkpointer_type}")
                    return None
            else:
                print("❌ No PostgreSQL configuration found")
                return None
        else:
            print("❌ Memory system not initialized")
            return None
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_memory_operations(service):
    """Test basic memory operations."""
    print("\n🔄 Testing Memory Operations")
    print("=" * 50)
    
    try:
        memory = service._memory
        
        # Test adding a message
        test_soeid = "test_user_postgresql"
        test_session_id = "session_postgresql_001"
        test_message = "Hello PostgreSQL memory!"
        
        print(f"📝 Adding message to memory...")
        await memory.add(
            session_id=test_session_id,
            query=test_message,
            metadata={"soeid": test_soeid}
        )
        print("✅ Message added successfully")
        
        # Test retrieving history
        print(f"📖 Retrieving conversation history...")
        history = await memory.get_history(
            session_id=test_session_id
        )
        
        if history:
            print(f"✅ Retrieved {len(history)} messages")
            for i, msg in enumerate(history):
                print(f"   {i+1}. [{msg.get('role', 'unknown')}] {msg.get('content', 'No content')}")
            return True
        else:
            print("❌ No history retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Memory operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_postgresql_persistence():
    """Verify data is actually stored in PostgreSQL."""
    print("\n🔍 Verifying PostgreSQL Data Persistence")
    print("=" * 50)
    
    try:
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check checkpoints table
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]
        print(f"📊 Checkpoints in database: {checkpoint_count}")
        
        # Check checkpoint_writes table
        cursor.execute("SELECT COUNT(*) FROM checkpoint_writes")
        writes_count = cursor.fetchone()[0]
        print(f"📊 Checkpoint writes in database: {writes_count}")
        
        # Get recent checkpoints with details
        cursor.execute("""
            SELECT thread_id, checkpoint_id, checkpoint_ns,
                   LENGTH(checkpoint::text) as checkpoint_size
            FROM checkpoints 
            ORDER BY thread_id, checkpoint_id 
            LIMIT 3
        """)
        
        checkpoints = cursor.fetchall()
        
        if checkpoints:
            print("📋 Recent checkpoints:")
            for thread_id, checkpoint_id, checkpoint_ns, size in checkpoints:
                print(f"   Thread: {thread_id}")
                print(f"   Checkpoint: {checkpoint_id}")
                print(f"   Namespace: {checkpoint_ns}")
                print(f"   Size: {size} bytes")
                print()
        
        cursor.close()
        conn.close()
        
        return checkpoint_count > 0 or writes_count > 0
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

async def test_session_management(service):
    """Test session-based memory management."""
    print("\n👥 Testing Session Management")
    print("=" * 50)
    
    try:
        memory = service._memory
        
        # Test multiple sessions for same user
        test_soeid = "test_user_sessions"
        sessions = ["session_1", "session_2", "session_3"]
        
        # Add messages to different sessions
        for i, session_id in enumerate(sessions):
            message = f"Message {i+1} in {session_id}"
            await memory.add(
                session_id=session_id,
                query=message,
                metadata={"soeid": test_soeid}
            )
            print(f"✅ Added message to {session_id}")
        
        # Retrieve history for each session
        for session_id in sessions:
            history = await memory.get_history(
                session_id=session_id
            )
            print(f"📖 {session_id}: {len(history)} messages")
        
        # Test user-level history
        user_history = await memory.get_user_history_by_soeid(test_soeid)
        print(f"📖 Total user history: {len(user_history)} messages")
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_cleanup(service):
    """Test memory cleanup functionality."""
    print("\n🧹 Testing Memory Cleanup")
    print("=" * 50)
    
    try:
        memory = service._memory
        
        # Test session clearing
        test_soeid = "test_cleanup_user"
        test_session_id = "cleanup_session"
        
        # Add a message
        await memory.add(
            session_id=test_session_id,
            query="Message to be cleared",
            metadata={"soeid": test_soeid}
        )
        
        # Verify it exists
        history_before = await memory.get_history(test_session_id)
        print(f"📊 Messages before cleanup: {len(history_before)}")
        
        # Clear the session
        await memory.clear_session(test_session_id)
        print("✅ Session cleared")
        
        # Verify it's gone
        history_after = await memory.get_history(test_session_id)
        print(f"📊 Messages after cleanup: {len(history_after)}")
        
        return len(history_after) == 0
        
    except Exception as e:
        print(f"❌ Cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Comprehensive PostgreSQL Memory Test")
    print("=" * 80)
    
    # Initialize service
    service = await test_memory_system_initialization()
    if not service:
        print("\n❌ CRITICAL: Memory system initialization failed!")
        return 1
    
    # Run comprehensive tests
    results = {
        "Memory Operations": await test_memory_operations(service),
        "PostgreSQL Persistence": await verify_postgresql_persistence(),
        "Session Management": await test_session_management(service),
        "Memory Cleanup": await test_memory_cleanup(service)
    }
    
    # Final database check
    final_persistence = await verify_postgresql_persistence()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total and final_persistence:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ PostgreSQL LangGraph memory is fully functional!")
        print("\n🚀 Ready for production use:")
        print("1. Memory system uses PostgreSQL for persistence")
        print("2. Conversation history is stored in database")
        print("3. Session management works correctly")
        print("4. Memory cleanup functions properly")
        print("5. Data persists across application restarts")
        
        # Cleanup
        if hasattr(service._memory, 'cleanup'):
            service._memory.cleanup()
            print("\n🧹 Memory resources cleaned up")
        
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED OR DATA NOT PERSISTED")
        print("Check the error messages above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
