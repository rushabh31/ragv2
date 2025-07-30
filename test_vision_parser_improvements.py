#!/usr/bin/env python3
"""
Test script to validate vision parser improvements:
1. Retry logic for vision extraction failures
2. SSL certificate handling
3. Filename extraction and storage in PostgreSQL and FAISS
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.indexers.pgvector_store import PgVectorStore
from src.rag.ingestion.indexers.faiss_vector_store import FAISSVectorStore
from src.rag.core.interfaces.base import Chunk, Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestVisionParserImprovements:
    """Test class for vision parser improvements."""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_retry_logic(self):
        """Test retry logic for vision extraction failures."""
        logger.info("🔄 Testing retry logic for vision extraction failures...")
        
        try:
            # Create parser with custom retry configuration
            config = {
                "max_pages": 5,
                "max_concurrent_pages": 2,
                "max_retries": 2,
                "retry_delay": 0.5,  # Shorter delay for testing
                "retry_backoff_multiplier": 2.0
            }
            
            parser = VisionParser(config)
            
            # Mock the vision model to simulate failures and eventual success
            mock_vision_model = AsyncMock()
            
            # First two calls fail, third succeeds
            mock_vision_model.parse_text_from_image.side_effect = [
                Exception("Network timeout"),  # First attempt fails
                Exception("API rate limit"),   # Second attempt fails
                "# Test Document\n\nThis is extracted text."  # Third attempt succeeds
            ]
            
            parser.vision_model = mock_vision_model
            
            # Test the retry logic
            result = await parser._extract_text_with_vision("fake_base64_image", page_num=0)
            
            # Verify that retries worked
            assert mock_vision_model.parse_text_from_image.call_count == 3
            assert result == "# Test Document\n\nThis is extracted text."
            
            self.test_results["retry_logic"] = "✅ PASS - Retry logic works correctly"
            logger.info("✅ Retry logic test passed")
            
        except Exception as e:
            self.test_results["retry_logic"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ Retry logic test failed: {str(e)}")
    
    async def test_ssl_certificate_handling(self):
        """Test SSL certificate handling."""
        logger.info("🔐 Testing SSL certificate handling...")
        
        try:
            # Create a temporary certificate file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as cert_file:
                cert_file.write("-----BEGIN CERTIFICATE-----\nFAKE_CERT_DATA\n-----END CERTIFICATE-----")
                cert_path = cert_file.name
            
            try:
                # Set environment variable
                os.environ["SSL_CERT_FILE"] = cert_path
                
                # Create parser - should detect and use the certificate
                config = {"max_pages": 1}
                parser = VisionParser(config)
                
                # Verify SSL certificate path is set
                assert parser.ssl_cert_path == cert_path
                assert os.environ.get("SSL_CERT_FILE") == cert_path
                
                self.test_results["ssl_certificate"] = "✅ PASS - SSL certificate handling works"
                logger.info("✅ SSL certificate test passed")
                
            finally:
                # Clean up
                os.unlink(cert_path)
                if "SSL_CERT_FILE" in os.environ:
                    del os.environ["SSL_CERT_FILE"]
                    
        except Exception as e:
            self.test_results["ssl_certificate"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ SSL certificate test failed: {str(e)}")
    
    async def test_filename_extraction(self):
        """Test filename extraction from file path."""
        logger.info("📁 Testing filename extraction...")
        
        try:
            config = {"max_pages": 1}
            parser = VisionParser(config)
            
            # Mock the vision model and PDF processing
            mock_vision_model = AsyncMock()
            mock_vision_model.parse_text_from_image.return_value = "Test content"
            mock_vision_model.validate_authentication.return_value = True
            parser.vision_model = mock_vision_model
            
            # Create a temporary PDF file for testing
            test_file_path = "/tmp/test_document.pdf"
            
            # Mock PyMuPDF document
            with patch('fitz.open') as mock_fitz:
                mock_doc = MagicMock()
                mock_doc.metadata = {
                    "title": "Test Document",
                    "author": "Test Author",
                    "creationDate": "D:20240101120000"
                }
                mock_doc.__len__.return_value = 1
                mock_doc.__getitem__.return_value.get_pixmap.return_value.tobytes.return_value = b"fake_image_data"
                mock_fitz.return_value = mock_doc
                
                # Parse the file
                documents = await parser._parse_file(test_file_path, {})
                
                # Verify filename is extracted and stored
                assert len(documents) == 1
                doc = documents[0]
                assert doc.metadata.get("file_name") == "test_document.pdf"
                assert doc.metadata.get("source_file") == "test_document.pdf"
                
                self.test_results["filename_extraction"] = "✅ PASS - Filename extraction works"
                logger.info("✅ Filename extraction test passed")
                
        except Exception as e:
            self.test_results["filename_extraction"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ Filename extraction test failed: {str(e)}")
    
    async def test_pgvector_filename_storage(self):
        """Test filename storage in PostgreSQL vector store."""
        logger.info("🐘 Testing filename storage in PostgreSQL...")
        
        try:
            # Create mock configuration for PgVector
            config = {
                "dimension": 768,
                "connection_string": "postgresql://test@localhost:5432/test_db",
                "table_name": "test_embeddings",
                "schema_name": "public",
                "index_method": "hnsw"
            }
            
            # Create PgVector store (won't actually connect in test)
            pgvector_store = PgVectorStore(config)
            
            # Create test chunk with filename metadata
            test_chunk = Chunk(
                content="Test content",
                metadata={
                    "file_name": "test_document.pdf",
                    "source_file": "test_document.pdf",
                    "document_id": "test-doc-123",
                    "chunk_index": 0,
                    "page_number": 1
                },
                embedding=[0.1] * 768  # Mock embedding
            )
            
            # Mock the database session and verify SQL query includes filename
            with patch('sqlalchemy.ext.asyncio.create_async_engine'), \
                 patch('sqlalchemy.ext.asyncio.async_sessionmaker'), \
                 patch('sqlalchemy.ext.asyncio.AsyncSession') as mock_session_class:
                
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                # Mock the execute method to capture the SQL
                executed_queries = []
                
                async def capture_execute(query, params=None):
                    executed_queries.append((str(query), params))
                    return MagicMock()
                
                mock_session.execute.side_effect = capture_execute
                mock_session.commit.return_value = None
                
                # Initialize and add vectors
                await pgvector_store.initialize()
                await pgvector_store._add_vectors([test_chunk])
                
                # Verify that filename fields are included in the SQL
                assert len(executed_queries) > 0
                
                # Check if any query includes file_name field
                filename_included = any(
                    "file_name" in str(query).lower() 
                    for query, params in executed_queries
                )
                
                assert filename_included, "file_name field not found in SQL queries"
                
                self.test_results["pgvector_filename"] = "✅ PASS - PostgreSQL filename storage works"
                logger.info("✅ PostgreSQL filename storage test passed")
                
        except Exception as e:
            self.test_results["pgvector_filename"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ PostgreSQL filename storage test failed: {str(e)}")
    
    async def test_faiss_filename_storage(self):
        """Test filename storage in FAISS vector store."""
        logger.info("🔍 Testing filename storage in FAISS...")
        
        try:
            # Create temporary paths for FAISS index
            with tempfile.TemporaryDirectory() as temp_dir:
                config = {
                    "dimension": 768,
                    "index_type": "Flat",
                    "index_path": os.path.join(temp_dir, "test.index"),
                    "metadata_path": os.path.join(temp_dir, "test.pkl")
                }
                
                faiss_store = FAISSVectorStore(config)
                
                # Create test chunk with filename metadata
                test_chunk = Chunk(
                    content="Test content",
                    metadata={
                        "file_name": "test_document.pdf",
                        "source_file": "test_document.pdf",
                        "document_id": "test-doc-123",
                        "chunk_index": 0,
                        "page_number": 1
                    },
                    embedding=[0.1] * 768  # Mock embedding
                )
                
                # Initialize and add vectors
                await faiss_store.initialize()
                await faiss_store._add_vectors([test_chunk])
                
                # Verify chunk is stored with filename metadata
                assert len(faiss_store.chunks) == 1
                stored_chunk = faiss_store.chunks[0]
                assert stored_chunk.metadata.get("file_name") == "test_document.pdf"
                assert stored_chunk.metadata.get("source_file") == "test_document.pdf"
                
                # Test search to ensure metadata is preserved
                search_results = await faiss_store._search_vectors([0.1] * 768, top_k=1)
                assert len(search_results) == 1
                result_chunk = search_results[0].chunk
                assert result_chunk.metadata.get("file_name") == "test_document.pdf"
                
                self.test_results["faiss_filename"] = "✅ PASS - FAISS filename storage works"
                logger.info("✅ FAISS filename storage test passed")
                
        except Exception as e:
            self.test_results["faiss_filename"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ FAISS filename storage test failed: {str(e)}")
    
    async def test_fallback_behavior(self):
        """Test fallback behavior when vision extraction fails completely."""
        logger.info("🔄 Testing fallback behavior...")
        
        try:
            config = {
                "max_pages": 1,
                "max_retries": 1,  # Only 1 retry for faster testing
                "retry_delay": 0.1
            }
            
            parser = VisionParser(config)
            
            # Mock vision model to always fail
            mock_vision_model = AsyncMock()
            mock_vision_model.parse_text_from_image.side_effect = Exception("Always fails")
            mock_vision_model.validate_authentication.return_value = True
            parser.vision_model = mock_vision_model
            
            # Mock PyMuPDF for fallback text extraction
            with patch('fitz.open') as mock_fitz:
                mock_doc = MagicMock()
                mock_doc.metadata = {"title": "Test"}
                mock_doc.__len__.return_value = 1
                
                # Mock page for fallback extraction
                mock_page = MagicMock()
                mock_page.get_text.return_value = "Fallback text content"
                mock_page.get_pixmap.return_value.tobytes.return_value = b"fake_image"
                mock_doc.__getitem__.return_value = mock_page
                mock_fitz.return_value = mock_doc
                
                # Parse file - should use fallback
                documents = await parser._parse_file("/tmp/test.pdf", {})
                
                # Verify fallback was used
                assert len(documents) == 1
                content = documents[0].content
                assert "fallback extraction" in content.lower()
                assert "Fallback text content" in content
                
                self.test_results["fallback_behavior"] = "✅ PASS - Fallback behavior works"
                logger.info("✅ Fallback behavior test passed")
                
        except Exception as e:
            self.test_results["fallback_behavior"] = f"❌ FAIL - {str(e)}"
            logger.error(f"❌ Fallback behavior test failed: {str(e)}")
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("🚀 Starting vision parser improvements tests...")
        
        # Run all tests
        await self.test_retry_logic()
        await self.test_ssl_certificate_handling()
        await self.test_filename_extraction()
        await self.test_pgvector_filename_storage()
        await self.test_faiss_filename_storage()
        await self.test_fallback_behavior()
        
        # Report results
        logger.info("\n" + "="*60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            logger.info(f"{test_name}: {result}")
            if result.startswith("✅"):
                passed += 1
            else:
                failed += 1
        
        logger.info("="*60)
        logger.info(f"Total Tests: {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        
        if failed == 0:
            logger.info("🎉 All tests passed! Vision parser improvements are working correctly.")
        else:
            logger.warning(f"⚠️  {failed} test(s) failed. Please review the issues above.")
        
        return failed == 0

async def main():
    """Main test function."""
    tester = TestVisionParserImprovements()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
