#!/usr/bin/env python3
"""
Focused PostgreSQL LangGraph Memory Test

This script specifically tests that PostgreSQL is being used for LangGraph checkpoints.
"""

import asyncio
import sys
from pathlib import Path
import psycopg2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_postgresql_checkpointer():
    """Test that PostgreSQL checkpointer is actually being used."""
    print("🧪 Testing PostgreSQL LangGraph Checkpointer")
    print("=" * 60)
    
    # Database connection
    db_name = "langgraph_test_db"
    connection_string = f"postgresql://rushabhsmacbook@localhost:5432/{db_name}"
    
    try:
        # Import and create memory with PostgreSQL
        from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
        
        memory_config = {
            "store_type": "postgres",
            "postgres": {
                "connection_string": connection_string
            }
        }
        
        print(f"📝 Creating LangGraphCheckpointMemory with PostgreSQL config...")
        memory = LangGraphCheckpointMemory(memory_config)
        print(f"✅ Memory created successfully")
        
        # Check what type of checkpointer was created
        checkpointer_type = type(memory._checkpointer).__name__
        print(f"🔍 Checkpointer type: {checkpointer_type}")
        
        if "Postgres" in checkpointer_type:
            print("✅ PostgreSQL checkpointer is being used!")
        else:
            print("❌ In-memory checkpointer is being used instead of PostgreSQL")
            return False
        
        # Clear any existing data
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints")
        cursor.execute("DELETE FROM checkpoint_writes")
        conn.commit()
        cursor.close()
        conn.close()
        print("🧹 Cleared existing checkpoint data")
        
        # Add a test interaction
        test_session = "postgres_test_session"
        test_query = "Hello PostgreSQL!"
        test_response = "Hello from PostgreSQL checkpointer!"
        
        print(f"📝 Adding test interaction to session: {test_session}")
        success = await memory.add(
            session_id=test_session,
            query=test_query,
            response=test_response,
            metadata={"soeid": "postgres_test_user", "test": "postgresql"}
        )
        
        if not success:
            print("❌ Failed to add interaction")
            return False
        
        print("✅ Interaction added successfully")
        
        # Check if data was written to PostgreSQL
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT thread_id, checkpoint_id FROM checkpoints LIMIT 3")
        checkpoints = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"📊 PostgreSQL checkpoint count: {checkpoint_count}")
        
        if checkpoint_count > 0:
            print("✅ Data was successfully written to PostgreSQL!")
            print("📋 Checkpoint details:")
            for thread_id, checkpoint_id in checkpoints:
                print(f"   Thread: {thread_id}, Checkpoint: {checkpoint_id}")
        else:
            print("❌ No data found in PostgreSQL checkpoints table")
            return False
        
        # Test retrieving the data
        print(f"📖 Retrieving conversation history...")
        history = await memory.get_history(test_session)
        
        print(f"✅ Retrieved {len(history)} messages from PostgreSQL")
        
        if history:
            print("📋 Retrieved messages:")
            for i, msg in enumerate(history):
                content = msg.get('content', 'No content')
                role = msg.get('role', 'unknown')
                print(f"   {i+1}. [{role}] {content}")
        
        # Verify the content matches what we stored
        found_query = any(test_query in msg.get('content', '') for msg in history)
        found_response = any(test_response in msg.get('content', '') for msg in history)
        
        if found_query and found_response:
            print("✅ Retrieved messages match stored data!")
            return True
        else:
            print("❌ Retrieved messages don't match stored data")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_postgresql_checkpointer()
    
    if success:
        print("\n🎉 PostgreSQL LangGraph checkpointer is working correctly!")
        print("✅ Data is being persisted to PostgreSQL")
        print("✅ Data can be retrieved from PostgreSQL")
        return 0
    else:
        print("\n❌ PostgreSQL LangGraph checkpointer test failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
