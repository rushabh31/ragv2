#!/usr/bin/env python3
"""
Simple PostgreSQL Checkpointer Test

This script tests the PostgreSQL checkpointer with a minimal example.
"""

import asyncio
from langgraph.checkpoint.postgres import PostgresSaver

async def test_simple_postgres():
    """Test PostgreSQL checkpointer with minimal operations."""
    print("🔍 Testing Simple PostgreSQL Checkpointer")
    print("=" * 60)
    
    connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
    
    try:
        # Create the context manager
        postgres_cm = PostgresSaver.from_conn_string(connection_string)
        
        # Enter the context manager
        with postgres_cm as saver:
            print(f"✅ Entered context: {type(saver)}")
            
            # Schema should already be set up, but let's ensure it
            try:
                saver.setup()
                print("✅ Schema setup completed")
            except Exception as e:
                print(f"⚠️  Schema setup warning: {e}")
            
            # Test with a very simple config
            config = {"configurable": {"thread_id": "simple_test"}}
            
            # Test get on empty thread (should return None)
            print("\n🧪 Testing get on empty thread...")
            try:
                result = saver.get(config)
                print(f"   ✅ get result: {result}")
                
                if result is None:
                    print("   ✅ Correctly returned None for empty thread")
                else:
                    print(f"   ⚠️  Unexpected result: {result}")
                
            except Exception as e:
                print(f"   ❌ get failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Test async get
            print("\n🧪 Testing async get...")
            try:
                result = await saver.aget(config)
                print(f"   ✅ async get result: {result}")
                
                if result is None:
                    print("   ✅ Correctly returned None for empty thread")
                else:
                    print(f"   ⚠️  Unexpected result: {result}")
                
            except Exception as e:
                print(f"   ❌ async get failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            print("\n✅ Simple PostgreSQL checkpointer test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_postgres())
    if success:
        print("\n🎉 PostgreSQL checkpointer is working correctly!")
        print("The issue might be in the LangGraph memory implementation.")
    else:
        print("\n❌ PostgreSQL checkpointer has issues!")
