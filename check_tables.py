#!/usr/bin/env python3
"""
Check PostgreSQL Table Structure

This script checks what tables and columns exist in the database.
"""

import psycopg2

def check_database_structure():
    """Check the database structure."""
    print("🔍 Checking Database Structure")
    print("=" * 50)
    
    try:
        connection_string = "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"📋 Tables in database ({len(tables)}):")
        
        for (table_name,) in tables:
            print(f"\n   📄 Table: {table_name}")
            
            # Get columns for this table
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cursor.fetchall()
            for col_name, data_type, nullable, default in columns:
                null_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                print(f"      - {col_name}: {data_type} {null_str}{default_str}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_database_structure()
    if success:
        print("\n✅ Database structure check completed!")
    else:
        print("\n❌ Database structure check failed!")
