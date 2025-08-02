#!/usr/bin/env python3
"""
Comprehensive Test Script for Advanced Implementations

This script tests the advanced environment manager, moved langgraph_utils,
and the fixed PostgreSQL checkpoint memory implementation.

Author: Expert Python Developer
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedImplementationTester:
    """
    Comprehensive tester for advanced implementations.
    
    Tests:
    - Environment variable manager
    - Moved langgraph_utils functionality
    - Advanced PostgreSQL checkpoint memory
    - Integration between components
    """
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.env_manager = None
        self.memory_instance = None
        
    async def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all tests and return results.
        
        Returns:
            Dictionary of test results
        """
        logger.info("🚀 Starting comprehensive advanced implementation tests")
        
        # Test environment manager
        await self.test_environment_manager()
        
        # Test moved langgraph_utils
        await self.test_langgraph_utils()
        
        # Test advanced checkpoint memory
        await self.test_advanced_checkpoint_memory()
        
        # Test PostgreSQL connection
        await self.test_postgresql_connection()
        
        # Test integration
        await self.test_integration()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    async def test_environment_manager(self):
        """Test the environment variable manager."""
        logger.info("🔧 Testing Environment Manager")
        
        try:
            from src.utils.env_manager import env, EnvironmentManager, EnvVarType, EnvVarConfig
            
            # Test 1: Basic functionality
            test_value = env.get_string("PATH")
            assert test_value is not None, "PATH environment variable should exist"
            
            # Test 2: Default values
            test_default = env.get_string("NON_EXISTENT_VAR", "default_value")
            assert test_default == "default_value", "Default value should be returned"
            
            # Test 3: Type conversion
            port = env.get_int("POSTGRES_PORT", 5432)
            assert isinstance(port, int), "Port should be converted to integer"
            
            # Test 4: Boolean conversion
            debug_mode = env.get_bool("DEBUG_MODE", False)
            assert isinstance(debug_mode, bool), "Debug mode should be boolean"
            
            # Test 5: Path conversion
            ssl_cert = env.get_path("SSL_CERT_FILE", "config/certs.pem")
            assert isinstance(ssl_cert, Path), "SSL cert path should be Path object"
            
            # Test 6: Custom environment manager
            custom_env = EnvironmentManager()
            custom_env.register_var(EnvVarConfig(
                name="TEST_VAR",
                var_type=EnvVarType.STRING,
                default="test_value",
                description="Test variable"
            ))
            
            test_val = custom_env.get("TEST_VAR")
            assert test_val == "test_value", "Custom variable should return default"
            
            # Test 7: Environment summary
            summary = env.get_summary()
            assert isinstance(summary, dict), "Summary should be a dictionary"
            assert len(summary) > 0, "Summary should contain registered variables"
            
            self.env_manager = env
            self.test_results["environment_manager"] = True
            logger.info("✅ Environment Manager tests passed")
            
        except Exception as e:
            logger.error(f"❌ Environment Manager test failed: {str(e)}")
            self.test_results["environment_manager"] = False
    
    async def test_langgraph_utils(self):
        """Test the moved langgraph_utils functionality."""
        logger.info("🔗 Testing Moved LangGraph Utils")
        
        try:
            # Test import from new location
            from examples.rag.chatbot.api.langgraph_utils import (
                WorkflowManager,
                RAGWorkflowState,
                RAGDecision,
                MessageRole,
                Message,
                RetrievedDocument
            )
            
            # Test 1: WorkflowManager creation
            manager = WorkflowManager()
            assert manager is not None, "WorkflowManager should be created"
            assert manager.workflow is not None, "Workflow should be initialized"
            
            # Test 2: State types
            test_state: RAGWorkflowState = {
                "query": "test query",
                "session_id": "test_session",
                "user_id": "test_user",
                "soeid": "test_soeid",
                "messages": [],
                "use_chat_history": False,
                "chat_history_days": 7,
                "retrieved_documents": [],
                "reranked_documents": [],
                "generation_parameters": {},
                "response": "",
                "error": None,
                "metrics": {}
            }
            
            # Test 3: Message creation
            message = Message(
                role=MessageRole.USER,
                content="Test message",
                metadata={"test": True}
            )
            assert message.role == MessageRole.USER, "Message role should be USER"
            assert message.content == "Test message", "Message content should match"
            
            # Test 4: RetrievedDocument creation
            doc = RetrievedDocument(
                content="Test document content",
                metadata={"source": "test"},
                score=0.95
            )
            assert doc.content == "Test document content", "Document content should match"
            assert doc.score == 0.95, "Document score should match"
            
            # Test 5: Decision enum
            assert RAGDecision.RETRIEVE == "retrieve", "Decision enum should work"
            assert RAGDecision.END == "end", "Decision enum should work"
            
            self.test_results["langgraph_utils"] = True
            logger.info("✅ LangGraph Utils tests passed")
            
        except Exception as e:
            logger.error(f"❌ LangGraph Utils test failed: {str(e)}")
            self.test_results["langgraph_utils"] = False
    
    async def test_advanced_checkpoint_memory(self):
        """Test the advanced checkpoint memory implementation."""
        logger.info("🧠 Testing Advanced Checkpoint Memory")
        
        try:
            from src.rag.chatbot.memory.advanced_langgraph_checkpoint_memory import AdvancedLangGraphCheckpointMemory
            
            # Test 1: In-memory configuration
            memory_config = {
                "store_type": "in_memory",
                "cache_ttl": 300,
                "batch_size": 100
            }
            
            memory = AdvancedLangGraphCheckpointMemory(memory_config)
            assert memory is not None, "Memory should be created"
            assert memory._store_type == "in_memory", "Store type should be in_memory"
            
            # Test 2: Add messages
            session_id = "test_session_123"
            success = await memory.add(
                session_id=session_id,
                query="What is machine learning?",
                response="Machine learning is a subset of AI...",
                metadata={"test": True, "soeid": "test_user"}
            )
            assert success, "Adding message should succeed"
            
            # Test 3: Get history
            history = await memory.get_history(session_id)
            assert len(history) >= 2, "History should contain user and assistant messages"
            assert any(msg.get("role") == "user" for msg in history), "Should contain user message"
            assert any(msg.get("role") == "assistant" for msg in history), "Should contain assistant message"
            
            # Test 4: Cache functionality
            cached_history = await memory.get_history(session_id)
            assert len(cached_history) == len(history), "Cached history should match"
            
            # Test 5: Clear session
            clear_success = await memory.clear_session(session_id)
            assert clear_success, "Clearing session should succeed"
            
            # Test 6: Verify session cleared
            cleared_history = await memory.get_history(session_id)
            assert len(cleared_history) == 0, "History should be empty after clearing"
            
            # Test 7: Stats
            stats = await memory.get_stats()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            assert "store_type" in stats, "Stats should contain store_type"
            
            self.memory_instance = memory
            self.test_results["advanced_checkpoint_memory"] = True
            logger.info("✅ Advanced Checkpoint Memory tests passed")
            
        except Exception as e:
            logger.error(f"❌ Advanced Checkpoint Memory test failed: {str(e)}")
            self.test_results["advanced_checkpoint_memory"] = False
    
    async def test_postgresql_connection(self):
        """Test PostgreSQL connection and configuration."""
        logger.info("🐘 Testing PostgreSQL Connection")
        
        try:
            from src.rag.chatbot.memory.advanced_langgraph_checkpoint_memory import AdvancedLangGraphCheckpointMemory
            
            # Test 1: Connection string building
            postgres_config = {
                "store_type": "postgres",
                "postgres": {
                    "connection_string": "postgresql://test@localhost:5432/test_db"
                }
            }
            
            # This will test connection string building without actually connecting
            memory = AdvancedLangGraphCheckpointMemory(postgres_config)
            assert memory._connection_string is not None, "Connection string should be built"
            
            # Test 2: Environment variable integration
            if self.env_manager:
                # Test that environment variables are properly read
                postgres_host = self.env_manager.get_string("POSTGRES_HOST", "localhost")
                postgres_port = self.env_manager.get_int("POSTGRES_PORT", 5432)
                assert postgres_host is not None, "Postgres host should be available"
                assert isinstance(postgres_port, int), "Postgres port should be integer"
            
            # Note: We don't actually test the PostgreSQL connection here
            # as it requires a running PostgreSQL instance
            
            self.test_results["postgresql_connection"] = True
            logger.info("✅ PostgreSQL Connection configuration tests passed")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL Connection test failed: {str(e)}")
            self.test_results["postgresql_connection"] = False
    
    async def test_integration(self):
        """Test integration between components."""
        logger.info("🔗 Testing Component Integration")
        
        try:
            # Test 1: Environment manager with memory
            if self.env_manager and self.memory_instance:
                # Test that memory can use environment variables
                cache_ttl = self.env_manager.get_int("CACHE_TTL", 3600)
                assert isinstance(cache_ttl, int), "Cache TTL should be integer"
                
                # Test memory stats include environment info
                stats = await self.memory_instance.get_stats()
                assert "store_type" in stats, "Stats should include store type"
            
            # Test 2: LangGraph utils with environment manager
            from examples.rag.chatbot.api.langgraph_utils import WorkflowManager
            from src.utils.env_manager import env
            
            # Test that workflow manager can access environment variables
            max_workers = env.get_int("MAX_WORKERS", 4)
            assert isinstance(max_workers, int), "Max workers should be integer"
            
            manager = WorkflowManager({"max_retries": 3})
            assert manager.max_retries == 3, "Config should be passed to workflow manager"
            
            # Test 3: Memory factory integration
            from src.rag.chatbot.memory.memory_factory import MemoryFactory
            
            # Test that factory can create advanced memory
            supported_types = MemoryFactory.get_supported_types()
            assert "advanced_langgraph_checkpoint" in supported_types, "Advanced memory should be supported"
            
            # Test factory creation
            memory_config = {"type": "advanced_langgraph_checkpoint", "store_type": "in_memory"}
            factory_memory = MemoryFactory.create_memory(memory_config)
            assert factory_memory is not None, "Factory should create memory instance"
            
            self.test_results["integration"] = True
            logger.info("✅ Integration tests passed")
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {str(e)}")
            self.test_results["integration"] = False
    
    def print_test_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "="*60)
        logger.info("🎯 ADVANCED IMPLEMENTATION TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name.replace('_', ' ').title()}")
        
        logger.info("-"*60)
        logger.info(f"📊 Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Advanced implementations are working correctly.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed. Please review the errors above.")
        
        logger.info("="*60)


async def main():
    """Main test execution function."""
    tester = AdvancedImplementationTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Return appropriate exit code
        all_passed = all(results.values())
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
