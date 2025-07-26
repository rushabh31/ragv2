#!/usr/bin/env python3
"""
Test PostgreSQL Checkpointer Methods

This script tests what methods are available on the PostgreSQL checkpointer.
"""

import asyncio
from langgraph.checkpoint.postgres import PostgresSaver

async def test_postgres_methods():
    """Test PostgreSQL checkpointer methods."""
    print("🔍 Testing PostgreSQL Checkpointer Methods")
    print("=" * 60)
    
    connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
    
    try:
        # Create the context manager
        postgres_cm = PostgresSaver.from_conn_string(connection_string)
        print(f"✅ Context manager created: {type(postgres_cm)}")
        
        # Enter the context manager
        with postgres_cm as saver:
            print(f"✅ Entered context: {type(saver)}")
            
            # Check available methods
            methods = [method for method in dir(saver) if not method.startswith('_')]
            print(f"\n📋 Available methods ({len(methods)}):")
            for method in sorted(methods):
                print(f"   - {method}")
            
            # Check specific async methods
            async_methods = ['aget', 'aput', 'alist', 'aget_tuple', 'adelete_thread']
            print(f"\n🔍 Checking async methods:")
            for method in async_methods:
                if hasattr(saver, method):
                    method_obj = getattr(saver, method)
                    print(f"   ✅ {method}: {type(method_obj)}")
                else:
                    print(f"   ❌ {method}: Not found")
            
            # Check sync methods
            sync_methods = ['get', 'put', 'list', 'get_tuple', 'delete_thread']
            print(f"\n🔍 Checking sync methods:")
            for method in sync_methods:
                if hasattr(saver, method):
                    method_obj = getattr(saver, method)
                    print(f"   ✅ {method}: {type(method_obj)}")
                else:
                    print(f"   ❌ {method}: Not found")
            
            # Test a simple operation
            print(f"\n🧪 Testing basic operations:")
            
            # Try to create a simple config
            config = {"configurable": {"thread_id": "test_thread"}}
            
            # Test sync get (should return None for non-existent)
            try:
                result = saver.get(config)
                print(f"   ✅ sync get: {result}")
            except Exception as e:
                print(f"   ❌ sync get failed: {e}")
            
            # Test async get (should return None for non-existent)
            try:
                result = await saver.aget(config)
                print(f"   ✅ async get: {result}")
            except Exception as e:
                print(f"   ❌ async get failed: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_postgres_methods())
    if success:
        print("\n🎉 PostgreSQL checkpointer methods tested successfully!")
    else:
        print("\n❌ PostgreSQL checkpointer method test failed!")
