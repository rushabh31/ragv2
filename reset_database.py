#!/usr/bin/env python3
"""
Reset Database for LangGraph PostgreSQL

This script drops and recreates the test database.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def reset_database():
    """Reset the test database."""
    print("🔄 Resetting Database")
    print("=" * 40)
    
    db_name = "langgraph_test_db"
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect("postgresql://rushabhsmacbook@localhost:5432/postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Drop database if exists
        print(f"🗑️  Dropping database {db_name}...")
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        print("✅ Database dropped")
        
        # Create database
        print(f"🆕 Creating database {db_name}...")
        cursor.execute(f"CREATE DATABASE {db_name}")
        print("✅ Database created")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False

if __name__ == "__main__":
    success = reset_database()
    if success:
        print("\n🎉 Database reset completed!")
        print("Ready to test PostgreSQL checkpointer.")
    else:
        print("\n❌ Database reset failed!")
