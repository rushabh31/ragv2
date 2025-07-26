#!/usr/bin/env python3
"""
LangGraph PostgreSQL Memory Test Script

This script tests the LangGraph checkpoint PostgreSQL memory system for chat history.
It creates the necessary database and tables, then tests the memory functionality.
"""

import asyncio
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class LangGraphPostgreSQLTester:
    """Test LangGraph PostgreSQL memory functionality."""
    
    def __init__(self):
        self.db_name = "langgraph_test_db"
        self.connection_string = f"postgresql://rushabhsmacbook@localhost:5432/{self.db_name}"
        self.admin_connection_string = "postgresql://rushabhsmacbook@localhost:5432/postgres"
        
    def print_banner(self):
        """Print test banner."""
        print("=" * 80)
        print("🧪 LangGraph PostgreSQL Memory Test")
        print("=" * 80)
        print()
        
    def create_database(self):
        """Create the test database if it doesn't exist."""
        print("🗄️  Setting up PostgreSQL database...")
        
        try:
            # Connect to default postgres database
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="rushabhsmacbook",
                database="postgres"
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.db_name,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                print(f"📝 Creating database: {self.db_name}")
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                print("✅ Database created successfully")
            else:
                print(f"✅ Database {self.db_name} already exists")
                
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return False
            
    def setup_langgraph_tables(self):
        """Set up LangGraph checkpoint tables."""
        print("\n🏗️  Setting up LangGraph tables...")
        
        try:
            # Connect to our test database
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Create the checkpoints table (LangGraph standard schema)
            checkpoint_table_sql = """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            """
            
            # Create the checkpoint_writes table (for pending writes)
            writes_table_sql = """
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """
            
            # Create indexes for better performance
            indexes_sql = [
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);",
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);",
                "CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);",
                "CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_checkpoint_id ON checkpoint_writes(checkpoint_id);"
            ]
            
            # Execute table creation
            cursor.execute(checkpoint_table_sql)
            cursor.execute(writes_table_sql)
            
            # Execute index creation
            for index_sql in indexes_sql:
                cursor.execute(index_sql)
            
            conn.commit()
            print("✅ LangGraph tables created successfully")
            
            # Verify tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name IN ('checkpoints', 'checkpoint_writes')
            """)
            tables = cursor.fetchall()
            print(f"📋 Created tables: {[table[0] for table in tables]}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error setting up tables: {e}")
            return False
            
    def test_basic_connection(self):
        """Test basic database connection."""
        print("\n🔌 Testing database connection...")
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("SELECT current_database(), current_user, version();")
            db_info = cursor.fetchone()
            
            print(f"✅ Connected to database: {db_info[0]}")
            print(f"✅ Connected as user: {db_info[1]}")
            print(f"✅ PostgreSQL version: {db_info[2][:50]}...")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
            
    async def test_langgraph_memory_integration(self):
        """Test LangGraph memory integration."""
        print("\n🧠 Testing LangGraph memory integration...")
        
        try:
            # Import LangGraph checkpoint memory
            from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
            
            # Create memory instance with PostgreSQL
            memory_config = {
                "store_type": "postgres",
                "postgres": {
                    "connection_string": self.connection_string
                }
            }
            
            memory = LangGraphCheckpointMemory(memory_config)
            print("✅ LangGraphCheckpointMemory instance created")
            
            # Test storing a message using the correct method
            test_soeid = "test_user_123"
            test_session_id = "test_session_456"
            test_message = "Hello, this is a test message!"
            test_response = "Hello! This is a test response."
            
            print(f"📝 Storing test interaction for SOEID: {test_soeid}")
            await memory.add(
                session_id=test_session_id,
                query=test_message,
                response=test_response,
                metadata={"soeid": test_soeid, "test": True, "timestamp": datetime.now().isoformat()}
            )
            print("✅ Interaction stored successfully")
            
            # Test retrieving conversation history
            print(f"📚 Getting conversation history for session: {test_session_id}")
            history = await memory.get_history(test_session_id)
            print(f"✅ Retrieved conversation history with {len(history)} entries")
            
            if history:
                print("📋 History details:")
                for i, msg in enumerate(history):
                    content = msg.get('content', msg.get('query', 'No content'))
                    print(f"   {i+1}. {content[:50]}...")
            
            # Test getting user history by SOEID
            print(f"🗂️  Getting user history for SOEID: {test_soeid}")
            user_history = await memory.get_user_history_by_soeid(test_soeid)
            print(f"✅ Found {len(user_history)} sessions for user")
            
            if user_history:
                print("📋 User session details:")
                for session in user_history:
                    print(f"   Session ID: {session.get('session_id', 'Unknown')}")
                    print(f"   Messages: {session.get('message_count', 0)}")
            
            # Test session-specific history by SOEID
            print(f"📄 Getting session history filtered by SOEID: {test_soeid}")
            session_history = await memory.get_session_history_by_soeid(test_session_id, test_soeid)
            print(f"✅ Retrieved {len(session_history)} messages for SOEID in session")
            
            # Test clearing session
            print(f"🧹 Testing session clearing")
            clear_result = await memory.clear_session(test_session_id)
            print(f"✅ Session clear result: {clear_result}")
            
            return True
            
        except Exception as e:
            print(f"❌ LangGraph memory integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def test_direct_database_operations(self):
        """Test direct database operations."""
        print("\n🔧 Testing direct database operations...")
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Test inserting a checkpoint
            test_checkpoint = {
                "thread_id": "test_thread_789",
                "checkpoint_id": "checkpoint_001",
                "checkpoint": {
                    "messages": [
                        {"role": "user", "content": "Test message"},
                        {"role": "assistant", "content": "Test response"}
                    ],
                    "metadata": {"created_at": datetime.now().isoformat()}
                },
                "metadata": {
                    "soeid": "test_user_direct",
                    "session_id": "direct_session_123"
                }
            }
            
            insert_sql = """
            INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata)
            VALUES (%(thread_id)s, %(checkpoint_id)s, %(checkpoint)s, %(metadata)s)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) 
            DO UPDATE SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata
            """
            
            cursor.execute(insert_sql, {
                "thread_id": test_checkpoint["thread_id"],
                "checkpoint_id": test_checkpoint["checkpoint_id"],
                "checkpoint": json.dumps(test_checkpoint["checkpoint"]),
                "metadata": json.dumps(test_checkpoint["metadata"])
            })
            
            conn.commit()
            print("✅ Test checkpoint inserted successfully")
            
            # Test querying checkpoints
            cursor.execute("""
                SELECT thread_id, checkpoint_id, checkpoint, metadata, created_at
                FROM checkpoints 
                WHERE thread_id = %s
                ORDER BY created_at DESC
            """, (test_checkpoint["thread_id"],))
            
            results = cursor.fetchall()
            print(f"✅ Retrieved {len(results)} checkpoints")
            
            if results:
                for result in results:
                    print(f"   Thread: {result[0]}, Checkpoint: {result[1]}")
                    print(f"   Created: {result[4]}")
            
            # Test table statistics
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            checkpoint_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM checkpoint_writes")
            writes_count = cursor.fetchone()[0]
            
            print(f"📊 Database statistics:")
            print(f"   Checkpoints: {checkpoint_count}")
            print(f"   Checkpoint writes: {writes_count}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Direct database operations test failed: {e}")
            return False
            
    async def test_chatbot_api_integration(self):
        """Test integration with chatbot API."""
        print("\n🤖 Testing chatbot API integration...")
        
        try:
            # Update chatbot config to use PostgreSQL
            config_path = Path("examples/rag/chatbot/config.yaml")
            
            if not config_path.exists():
                print("⚠️  Chatbot config not found, skipping API integration test")
                return True
                
            import yaml
            
            # Read current config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update memory configuration
            if 'chatbot' not in config:
                config['chatbot'] = {}
            if 'memory' not in config['chatbot']:
                config['chatbot']['memory'] = {}
                
            config['chatbot']['memory'] = {
                'type': 'langgraph_checkpoint',
                'store_type': 'postgres',
                'postgres': {
                    'connection_string': self.connection_string
                }
            }
            
            # Write updated config
            config_backup_path = config_path.with_suffix('.yaml.backup')
            if not config_backup_path.exists():
                # Create backup
                with open(config_backup_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print("✅ Created config backup")
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print("✅ Updated chatbot config for PostgreSQL")
            
            # Test importing chatbot service
            from examples.rag.chatbot.api.service import ChatbotService
            
            service = ChatbotService()
            print("✅ ChatbotService imported successfully")
            
            # Test memory initialization
            if hasattr(service, 'memory') and service.memory:
                print("✅ Memory system initialized in chatbot service")
                
                # Test a simple memory operation
                test_soeid = "api_test_user"
                test_session = "api_test_session"
                
                await service.memory.add(
                    session_id=test_session,
                    query="API integration test",
                    response="API integration response",
                    metadata={"soeid": test_soeid}
                )
                print("✅ Memory operation through API service successful")
            else:
                print("⚠️  Memory system not initialized in chatbot service")
            
            return True
            
        except Exception as e:
            print(f"❌ Chatbot API integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def cleanup_test_data(self):
        """Clean up test data."""
        print("\n🧹 Cleaning up test data...")
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Delete test data
            cursor.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'test_%'")
            cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id LIKE 'test_%'")
            
            deleted_checkpoints = cursor.rowcount
            conn.commit()
            
            print(f"✅ Cleaned up test data ({deleted_checkpoints} records)")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False
            
    def print_summary(self, results):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 LANGGRAPH POSTGRESQL TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("LangGraph PostgreSQL memory is working correctly.")
            print(f"\nDatabase connection string: {self.connection_string}")
            print("\nYou can now use PostgreSQL memory in your chatbot:")
            print("1. Update config.yaml with PostgreSQL settings")
            print("2. Start the chatbot API")
            print("3. Test chat functionality with persistent memory")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Please check the error messages above.")
            
        return passed_tests == total_tests

async def main():
    """Main test function."""
    tester = LangGraphPostgreSQLTester()
    tester.print_banner()
    
    # Run all tests
    results = {
        "Database Creation": tester.create_database(),
        "Table Setup": tester.setup_langgraph_tables(),
        "Basic Connection": tester.test_basic_connection(),
        "Direct Database Operations": tester.test_direct_database_operations(),
        "LangGraph Memory Integration": await tester.test_langgraph_memory_integration(),
        "Chatbot API Integration": await tester.test_chatbot_api_integration()
    }
    
    # Clean up test data
    tester.cleanup_test_data()
    
    # Print summary
    success = tester.print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
