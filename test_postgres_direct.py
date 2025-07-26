#!/usr/bin/env python3
"""
Direct PostgreSQL checkpointer test to understand connection lifecycle.
"""

import asyncio
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph
from typing import TypedDict

# Test configuration
CONNECTION_STRING = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"

class TestState(TypedDict):
    messages: list

def test_node(state: TestState):
    """Simple test node."""
    return {"messages": state["messages"] + ["test"]}

async def test_postgres_direct():
    """Test PostgreSQL checkpointer directly."""
    print("🧪 Direct PostgreSQL Checkpointer Test")
    print("=" * 50)
    
    # Test 1: Context manager approach (current approach)
    print("\n1️⃣ Testing Context Manager Approach")
    try:
        postgres_cm = PostgresSaver.from_conn_string(CONNECTION_STRING)
        print(f"✅ Context manager created: {type(postgres_cm)}")
        
        # Enter context manager
        postgres_saver = postgres_cm.__enter__()
        print(f"✅ Entered context manager: {type(postgres_saver)}")
        
        # Setup schema
        postgres_saver.setup()
        print("✅ Schema setup completed")
        
        # Test basic operations
        config = {"configurable": {"thread_id": "test-thread-1"}}
        
        # Try to get (should return None for new thread)
        checkpoint = postgres_saver.get(config)
        print(f"✅ Get operation: {checkpoint is not None}")
        
        # Try to put a checkpoint
        test_data = {
            "v": 1,
            "ts": "2025-01-26T07:00:00Z",
            "id": "test-checkpoint-1",
            "channel_values": {"messages": ["Hello"]},
            "channel_versions": {"messages": 1},
            "versions_seen": {"test": {"messages": 1}},
            "pending_sends": []
        }
        
        postgres_saver.put(config, test_data, {}, {})
        print("✅ Put operation successful")
        
        # Try to get again
        checkpoint = postgres_saver.get(config)
        print(f"✅ Get after put: {checkpoint is not None}")
        
        # Clean exit
        postgres_cm.__exit__(None, None, None)
        print("✅ Context manager exited cleanly")
        
    except Exception as e:
        print(f"❌ Context manager approach failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Direct connection approach
    print("\n2️⃣ Testing Direct Connection Approach")
    try:
        import psycopg
        
        # Create direct connection
        conn = psycopg.connect(CONNECTION_STRING)
        print(f"✅ Direct connection created: {type(conn)}")
        
        # Create PostgresSaver with direct connection
        postgres_saver = PostgresSaver(conn)
        print(f"✅ PostgresSaver created with direct connection: {type(postgres_saver)}")
        
        # Setup schema
        postgres_saver.setup()
        print("✅ Schema setup completed")
        
        # Test operations
        config = {"configurable": {"thread_id": "test-thread-2"}}
        
        # Try to get
        checkpoint = postgres_saver.get(config)
        print(f"✅ Get operation: {checkpoint is not None}")
        
        # Try to put
        test_data = {
            "v": 1,
            "ts": "2025-01-26T07:00:00Z",
            "id": "test-checkpoint-2",
            "channel_values": {"messages": ["Hello Direct"]},
            "channel_versions": {"messages": 1},
            "versions_seen": {"test": {"messages": 1}},
            "pending_sends": []
        }
        
        postgres_saver.put(config, test_data, {}, {})
        print("✅ Put operation successful")
        
        # Try to get again
        checkpoint = postgres_saver.get(config)
        print(f"✅ Get after put: {checkpoint is not None}")
        
        # Close connection
        conn.close()
        print("✅ Connection closed cleanly")
        
    except Exception as e:
        print(f"❌ Direct connection approach failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: StateGraph with checkpointer
    print("\n3️⃣ Testing StateGraph with Checkpointer")
    try:
        # Create graph with checkpointer
        postgres_cm = PostgresSaver.from_conn_string(CONNECTION_STRING)
        
        with postgres_cm as checkpointer:
            print(f"✅ Using checkpointer in with statement: {type(checkpointer)}")
            
            # Setup schema
            checkpointer.setup()
            print("✅ Schema setup completed")
            
            # Create simple graph
            graph = StateGraph(TestState)
            graph.add_node("test", test_node)
            graph.set_entry_point("test")
            graph.set_finish_point("test")
            
            # Compile with checkpointer
            app = graph.compile(checkpointer=checkpointer)
            print("✅ Graph compiled with checkpointer")
            
            # Test graph execution
            config = {"configurable": {"thread_id": "test-thread-3"}}
            result = app.invoke({"messages": []}, config)
            print(f"✅ Graph execution: {result}")
            
            # Test checkpoint retrieval
            checkpoint = checkpointer.get(config)
            print(f"✅ Checkpoint after graph execution: {checkpoint is not None}")
            
        print("✅ With statement completed")
        
    except Exception as e:
        print(f"❌ StateGraph approach failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_postgres_direct())
