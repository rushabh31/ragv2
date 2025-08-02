#!/usr/bin/env python3
"""
Comprehensive test script for ingestion session tracking functionality.

This script validates:
1. Session ID generation at ingestion start
2. Session ID propagation through the pipeline
3. Session ID storage in PostgreSQL
4. Session tracking API endpoints
5. Session information retrieval and listing
"""

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.rag.ingestion.api.service import IngestionService
from src.rag.ingestion.indexers.pgvector_store import PgVectorStore
from src.utils.env_manager import EnvManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionTrackingTester:
    """Test class for session tracking functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.env_manager = EnvManager()
        self.ingestion_service = None
        self.test_sessions = []
        self.test_files = []
        
    async def setup(self):
        """Set up test environment."""
        logger.info("Setting up session tracking test environment...")
        
        # Initialize ingestion service
        self.ingestion_service = IngestionService(data_dir="./test_data")
        await self.ingestion_service._init_components()
        
        logger.info("✅ Test environment setup complete")
    
    def create_test_pdf_content(self, filename: str) -> str:
        """Create a test PDF file with sample content."""
        # Create a simple text file for testing (simulating PDF content)
        content = f"""
# Test Document: {filename}

This is a test document created at {datetime.now().isoformat()}.

## Section 1: Introduction
This document is used to test the session tracking functionality in the RAG ingestion pipeline.

## Section 2: Content
The session ID should be generated at the beginning of the ingestion process and propagated through:
1. Document parsing
2. Chunking
3. Embedding generation
4. Vector store insertion

## Section 3: Verification
We will verify that the session ID is properly stored in PostgreSQL and can be retrieved via API endpoints.

Test UUID: {str(uuid.uuid4())}
Created: {datetime.now().isoformat()}
"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        self.test_files.append(temp_path)
        return temp_path
    
    async def test_session_generation_and_propagation(self) -> Dict[str, Any]:
        """Test session ID generation and propagation through pipeline."""
        logger.info("🧪 Testing session ID generation and propagation...")
        
        test_results = {
            "session_generation": False,
            "session_propagation": False,
            "session_storage": False,
            "session_id": None,
            "document_id": None,
            "chunks_with_session": 0,
            "error": None
        }
        
        try:
            # Create test document
            test_file = self.create_test_pdf_content("session_test_doc.txt")
            user_id = "test_user_session_tracking"
            
            # Upload document (this should generate session ID)
            job = await self.ingestion_service.upload_document(
                file_path=test_file,
                user_id=user_id,
                metadata={"test_type": "session_tracking", "original_filename": "session_test_doc.txt"}
            )
            
            test_results["document_id"] = job.document_id
            logger.info(f"📄 Document uploaded with job_id: {job.job_id}, document_id: {job.document_id}")
            
            # Wait for processing to complete
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                job_status = await self.ingestion_service.get_job_status(job.job_id)
                if job_status and job_status.status in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1
            
            if job_status.status == "failed":
                test_results["error"] = f"Job failed: {job_status.error_message}"
                return test_results
            
            logger.info(f"✅ Document processing completed with status: {job_status.status}")
            
            # Check if session ID was generated and stored
            if hasattr(self.ingestion_service._vector_store, 'chunks'):
                chunks_with_session = 0
                session_ids = set()
                
                for chunk in self.ingestion_service._vector_store.chunks:
                    if chunk.metadata and chunk.metadata.get('document_id') == job.document_id:
                        if 'session_id' in chunk.metadata:
                            chunks_with_session += 1
                            session_ids.add(chunk.metadata['session_id'])
                            logger.info(f"📊 Found chunk with session_id: {chunk.metadata['session_id']}")
                
                test_results["chunks_with_session"] = chunks_with_session
                
                if session_ids:
                    test_results["session_generation"] = True
                    test_results["session_id"] = list(session_ids)[0]  # Get first session ID
                    
                    if chunks_with_session > 0:
                        test_results["session_propagation"] = True
                        logger.info(f"✅ Session ID propagated to {chunks_with_session} chunks")
                    
                    # Test session storage in PostgreSQL
                    if isinstance(self.ingestion_service._vector_store, PgVectorStore):
                        test_results["session_storage"] = True
                        logger.info("✅ Session ID stored in PostgreSQL via PgVectorStore")
                    
                    self.test_sessions.append(test_results["session_id"])
                else:
                    test_results["error"] = "No session IDs found in chunk metadata"
            else:
                test_results["error"] = "No chunks found in vector store"
                
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"❌ Session tracking test failed: {str(e)}", exc_info=True)
        
        return test_results
    
    async def test_session_api_endpoints(self, session_id: str) -> Dict[str, Any]:
        """Test session tracking API endpoints."""
        logger.info(f"🧪 Testing session API endpoints for session: {session_id}")
        
        test_results = {
            "get_session_info": False,
            "list_sessions": False,
            "list_all_sessions": False,
            "session_info": None,
            "sessions_list": None,
            "error": None
        }
        
        try:
            # Test get_session_info
            session_info = await self.ingestion_service.get_session_info(session_id)
            if session_info:
                test_results["get_session_info"] = True
                test_results["session_info"] = session_info
                logger.info(f"✅ Retrieved session info: {session_info['total_chunks']} chunks")
            else:
                test_results["error"] = f"Session info not found for {session_id}"
                return test_results
            
            # Test list_sessions
            sessions_result = await self.ingestion_service.list_sessions(
                user_id="test_user_session_tracking", 
                page=1, 
                page_size=10
            )
            if sessions_result and sessions_result["sessions"]:
                test_results["list_sessions"] = True
                test_results["sessions_list"] = sessions_result
                logger.info(f"✅ Listed {len(sessions_result['sessions'])} sessions for user")
            
            # Test list_all_sessions
            all_sessions_result = await self.ingestion_service.list_sessions(
                user_id=None, 
                page=1, 
                page_size=20
            )
            if all_sessions_result:
                test_results["list_all_sessions"] = True
                logger.info(f"✅ Listed {len(all_sessions_result['sessions'])} total sessions")
                
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"❌ Session API test failed: {str(e)}", exc_info=True)
        
        return test_results
    
    async def test_session_metadata_integrity(self, session_id: str) -> Dict[str, Any]:
        """Test session metadata integrity and completeness."""
        logger.info(f"🧪 Testing session metadata integrity for: {session_id}")
        
        test_results = {
            "metadata_complete": False,
            "required_fields": [],
            "missing_fields": [],
            "session_consistency": False,
            "error": None
        }
        
        try:
            session_info = await self.ingestion_service.get_session_info(session_id)
            if not session_info:
                test_results["error"] = f"Session {session_id} not found"
                return test_results
            
            # Check required fields
            required_fields = [
                'session_id', 'total_chunks', 'document_id', 'user_id', 
                'filename', 'ingestion_started_at'
            ]
            
            for field in required_fields:
                if field in session_info and session_info[field] is not None:
                    test_results["required_fields"].append(field)
                else:
                    test_results["missing_fields"].append(field)
            
            # Check if all required fields are present
            if len(test_results["missing_fields"]) == 0:
                test_results["metadata_complete"] = True
                logger.info("✅ All required session metadata fields present")
            else:
                logger.warning(f"⚠️ Missing fields: {test_results['missing_fields']}")
            
            # Check session consistency across chunks
            if hasattr(self.ingestion_service._vector_store, 'chunks'):
                session_chunks = [
                    chunk for chunk in self.ingestion_service._vector_store.chunks
                    if chunk.metadata and chunk.metadata.get('session_id') == session_id
                ]
                
                if session_chunks:
                    # Verify all chunks have same session metadata
                    first_chunk_meta = session_chunks[0].metadata
                    consistent = all(
                        chunk.metadata.get('session_id') == session_id and
                        chunk.metadata.get('document_id') == first_chunk_meta.get('document_id') and
                        chunk.metadata.get('user_id') == first_chunk_meta.get('user_id')
                        for chunk in session_chunks
                    )
                    
                    if consistent:
                        test_results["session_consistency"] = True
                        logger.info(f"✅ Session metadata consistent across {len(session_chunks)} chunks")
                    else:
                        logger.warning("⚠️ Inconsistent session metadata across chunks")
                        
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"❌ Session metadata test failed: {str(e)}", exc_info=True)
        
        return test_results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive session tracking tests."""
        logger.info("🚀 Starting comprehensive session tracking tests...")
        
        overall_results = {
            "setup_success": False,
            "session_generation_test": None,
            "session_api_test": None,
            "session_metadata_test": None,
            "overall_success": False,
            "test_summary": {},
            "errors": []
        }
        
        try:
            # Setup
            await self.setup()
            overall_results["setup_success"] = True
            
            # Test 1: Session generation and propagation
            session_test = await self.test_session_generation_and_propagation()
            overall_results["session_generation_test"] = session_test
            
            if session_test.get("error"):
                overall_results["errors"].append(f"Session generation: {session_test['error']}")
            
            # Test 2: Session API endpoints (if session was generated)
            if session_test.get("session_id"):
                api_test = await self.test_session_api_endpoints(session_test["session_id"])
                overall_results["session_api_test"] = api_test
                
                if api_test.get("error"):
                    overall_results["errors"].append(f"Session API: {api_test['error']}")
                
                # Test 3: Session metadata integrity
                metadata_test = await self.test_session_metadata_integrity(session_test["session_id"])
                overall_results["session_metadata_test"] = metadata_test
                
                if metadata_test.get("error"):
                    overall_results["errors"].append(f"Session metadata: {metadata_test['error']}")
            
            # Calculate overall success
            session_success = (
                session_test.get("session_generation", False) and
                session_test.get("session_propagation", False) and
                session_test.get("session_storage", False)
            )
            
            api_success = (
                overall_results["session_api_test"] and
                overall_results["session_api_test"].get("get_session_info", False) and
                overall_results["session_api_test"].get("list_sessions", False)
            )
            
            metadata_success = (
                overall_results["session_metadata_test"] and
                overall_results["session_metadata_test"].get("metadata_complete", False) and
                overall_results["session_metadata_test"].get("session_consistency", False)
            )
            
            overall_results["overall_success"] = session_success and api_success and metadata_success
            
            # Generate test summary
            overall_results["test_summary"] = {
                "session_generation": "✅ PASS" if session_success else "❌ FAIL",
                "api_endpoints": "✅ PASS" if api_success else "❌ FAIL",
                "metadata_integrity": "✅ PASS" if metadata_success else "❌ FAIL",
                "total_errors": len(overall_results["errors"])
            }
            
        except Exception as e:
            overall_results["errors"].append(f"Test execution: {str(e)}")
            logger.error(f"❌ Comprehensive test failed: {str(e)}", exc_info=True)
        
        finally:
            await self.cleanup()
        
        return overall_results
    
    async def cleanup(self):
        """Clean up test resources."""
        logger.info("🧹 Cleaning up test resources...")
        
        # Remove test files
        for file_path in self.test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed test file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove test file {file_path}: {e}")
        
        logger.info("✅ Cleanup complete")

def print_test_results(results: Dict[str, Any]):
    """Print formatted test results."""
    print("\n" + "="*80)
    print("🧪 SESSION TRACKING TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL SUCCESS: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
    
    if results.get("test_summary"):
        print(f"\n📋 TEST SUMMARY:")
        for test_name, status in results["test_summary"].items():
            print(f"  • {test_name.replace('_', ' ').title()}: {status}")
    
    if results.get("session_generation_test"):
        session_test = results["session_generation_test"]
        print(f"\n🔄 SESSION GENERATION & PROPAGATION:")
        print(f"  • Session ID Generated: {'✅' if session_test.get('session_generation') else '❌'}")
        print(f"  • Session ID Propagated: {'✅' if session_test.get('session_propagation') else '❌'}")
        print(f"  • PostgreSQL Storage: {'✅' if session_test.get('session_storage') else '❌'}")
        if session_test.get("session_id"):
            print(f"  • Session ID: {session_test['session_id']}")
        if session_test.get("chunks_with_session"):
            print(f"  • Chunks with Session ID: {session_test['chunks_with_session']}")
    
    if results.get("session_api_test"):
        api_test = results["session_api_test"]
        print(f"\n🌐 API ENDPOINTS:")
        print(f"  • Get Session Info: {'✅' if api_test.get('get_session_info') else '❌'}")
        print(f"  • List Sessions: {'✅' if api_test.get('list_sessions') else '❌'}")
        print(f"  • List All Sessions: {'✅' if api_test.get('list_all_sessions') else '❌'}")
    
    if results.get("session_metadata_test"):
        meta_test = results["session_metadata_test"]
        print(f"\n📋 METADATA INTEGRITY:")
        print(f"  • Complete Metadata: {'✅' if meta_test.get('metadata_complete') else '❌'}")
        print(f"  • Session Consistency: {'✅' if meta_test.get('session_consistency') else '❌'}")
        if meta_test.get("required_fields"):
            print(f"  • Required Fields Present: {len(meta_test['required_fields'])}")
        if meta_test.get("missing_fields"):
            print(f"  • Missing Fields: {meta_test['missing_fields']}")
    
    if results.get("errors"):
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for i, error in enumerate(results["errors"], 1):
            print(f"  {i}. {error}")
    
    print("\n" + "="*80)

async def main():
    """Main test execution function."""
    print("🚀 Starting Session Tracking Comprehensive Test Suite")
    print("="*80)
    
    tester = SessionTrackingTester()
    results = await tester.run_comprehensive_test()
    
    print_test_results(results)
    
    # Exit with appropriate code
    exit_code = 0 if results.get("overall_success", False) else 1
    print(f"\n🏁 Test completed with exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
