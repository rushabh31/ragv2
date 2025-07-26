#!/usr/bin/env python3
"""
Test PostgresSaver initialization
"""

import asyncio

async def test_postgres_saver():
    """Test PostgresSaver initialization."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        
        print("🔍 Testing PostgresSaver.from_conn_string()...")
        
        # Test the from_conn_string method
        saver = PostgresSaver.from_conn_string(connection_string)
        print(f"✅ PostgresSaver created: {type(saver)}")
        print(f"📋 Saver details: {saver}")
        
        # Check if it's a context manager
        if hasattr(saver, '__enter__') and hasattr(saver, '__exit__'):
            print("🔍 PostgresSaver is a context manager")
            
            # Try using it as a context manager
            with saver as actual_saver:
                print(f"✅ Context manager entered: {type(actual_saver)}")
                print(f"📋 Actual saver: {actual_saver}")
                
                # Test if it has the expected methods
                if hasattr(actual_saver, 'put') and hasattr(actual_saver, 'get'):
                    print("✅ Saver has expected methods (put, get)")
                else:
                    print("❌ Saver missing expected methods")
                    
        elif hasattr(saver, '__aenter__') and hasattr(saver, '__aexit__'):
            print("🔍 PostgresSaver is an async context manager")
            
            # Try using it as an async context manager
            async with saver as actual_saver:
                print(f"✅ Async context manager entered: {type(actual_saver)}")
                print(f"📋 Actual saver: {actual_saver}")
                
                # Test if it has the expected methods
                if hasattr(actual_saver, 'put') and hasattr(actual_saver, 'get'):
                    print("✅ Saver has expected methods (put, get)")
                else:
                    print("❌ Saver missing expected methods")
        else:
            print("🔍 PostgresSaver is not a context manager")
            
            # Test if it has the expected methods directly
            if hasattr(saver, 'put') and hasattr(saver, 'get'):
                print("✅ Saver has expected methods (put, get)")
            else:
                print("❌ Saver missing expected methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PostgresSaver: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_postgres_saver())
    exit(0 if success else 1)
