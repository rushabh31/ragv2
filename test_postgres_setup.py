#!/usr/bin/env python3
"""
Test PostgreSQL Checkpointer Setup

This script sets up the PostgreSQL database schema for LangGraph checkpointer.
"""

import asyncio
from langgraph.checkpoint.postgres import PostgresSaver

async def setup_postgres_schema():
    """Set up PostgreSQL schema for LangGraph checkpointer."""
    print("🔧 Setting up PostgreSQL Schema for LangGraph")
    print("=" * 60)
    
    connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
    
    try:
        # Create the context manager
        postgres_cm = PostgresSaver.from_conn_string(connection_string)
        print(f"✅ Context manager created")
        
        # Enter the context manager
        with postgres_cm as saver:
            print(f"✅ Entered context: {type(saver)}")
            
            # Check if setup method exists
            if hasattr(saver, 'setup'):
                print("🔧 Running setup method...")
                saver.setup()
                print("✅ Setup completed successfully")
            else:
                print("❌ No setup method found")
                return False
            
            # Test basic operations after setup
            print(f"\n🧪 Testing operations after setup:")
            
            config = {"configurable": {"thread_id": "test_setup_thread"}}
            
            # Test sync get (should return None for non-existent)
            try:
                result = saver.get(config)
                print(f"   ✅ sync get: {result}")
            except Exception as e:
                print(f"   ❌ sync get failed: {e}")
                return False
            
            # Test async get (should return None for non-existent)
            try:
                result = await saver.aget(config)
                print(f"   ✅ async get: {result}")
            except Exception as e:
                print(f"   ❌ async get failed: {e}")
                return False
            
            print("✅ PostgreSQL schema setup successful!")
            return True
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_tables():
    """Verify that the required tables exist."""
    print("\n🔍 Verifying Database Tables")
    print("=" * 40)
    
    try:
        import psycopg2
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check for required tables
        required_tables = ['checkpoints', 'checkpoint_writes', 'checkpoint_blobs']
        
        for table in required_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table,))
            
            exists = cursor.fetchone()[0]
            status = "✅" if exists else "❌"
            print(f"   {status} Table '{table}': {'exists' if exists else 'missing'}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Table verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PostgreSQL LangGraph Setup")
    print("=" * 80)
    
    success = asyncio.run(setup_postgres_schema())
    
    if success:
        asyncio.run(verify_tables())
        print("\n🎉 PostgreSQL LangGraph setup completed!")
        print("Ready to test memory operations.")
    else:
        print("\n❌ PostgreSQL LangGraph setup failed!")
        print("Check the error messages above.")
