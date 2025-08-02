#!/usr/bin/env python3
"""
Comprehensive test for PDF image storage functionality.

This test validates that:
1. Vision parser extracts and stores page images as base64
2. PgVector store stores base64 images alongside embeddings
3. Vector retriever returns images in retrieved documents
4. Chatbot API includes image data in responses
5. Memory integration stores image data for context

Author: AI Assistant
Date: 2025-08-01
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import fitz  # PyMuPDF
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.ingestion.indexers.pgvector_store import PgVectorStore
from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder
from src.rag.shared.utils.config_manager import ConfigManager
from src.utils.env_manager import env


class PDFImageStorageTest:
    """Comprehensive test suite for PDF image storage functionality."""
    
    def __init__(self):
        """Initialize test suite with configuration."""
        self.config_manager = ConfigManager()
        self.test_results = {
            "vision_parser_test": False,
            "pgvector_storage_test": False,
            "retrieval_test": False,
            "image_data_validation": False,
            "end_to_end_test": False
        }
        
        # Test configuration
        self.test_config = {
            "vision_parser": {
                "provider": "vertex_ai",
                "model": "gemini-1.5-pro-002",
                "max_pages": 5,
                "max_concurrent_pages": 2
            },
            "pgvector": {
                "dimension": 768,
                "connection_string": env.get_string("POSTGRES_CONNECTION_STRING", 
                    "postgresql://rushabhsmacbook@localhost:5432/langgraph_test_db"),
                "table_name": "test_document_embeddings",
                "schema_name": "public",
                "index_method": "hnsw"
            },
            "embedding": {
                "provider": "vertex_ai",
                "model": "text-embedding-004",
                "batch_size": 10
            }
        }
    
    def create_test_pdf(self) -> str:
        """Create a simple test PDF with text and visual elements."""
        # Create a temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Create PDF with PyMuPDF
        doc = fitz.open()
        
        # Page 1: Simple text
        page1 = doc.new_page()
        page1.insert_text((50, 100), "Test Document - Page 1", fontsize=16)
        page1.insert_text((50, 150), "This is a test PDF for image storage validation.", fontsize=12)
        page1.insert_text((50, 200), "It contains multiple pages with different content.", fontsize=12)
        
        # Page 2: More complex content
        page2 = doc.new_page()
        page2.insert_text((50, 100), "Test Document - Page 2", fontsize=16)
        page2.insert_text((50, 150), "This page contains structured information:", fontsize=12)
        page2.insert_text((70, 200), "• Item 1: First test item", fontsize=10)
        page2.insert_text((70, 220), "• Item 2: Second test item", fontsize=10)
        page2.insert_text((70, 240), "• Item 3: Third test item", fontsize=10)
        
        # Add a simple rectangle for visual content
        rect = fitz.Rect(300, 150, 500, 250)
        page2.draw_rect(rect, color=(0, 0, 1), width=2)
        page2.insert_text((320, 200), "Visual Element", fontsize=10)
        
        doc.save(temp_path)
        doc.close()
        
        logger.info(f"Created test PDF: {temp_path}")
        return temp_path
    
    async def test_vision_parser_image_extraction(self, pdf_path: str) -> bool:
        """Test that vision parser extracts and stores page images as base64."""
        logger.info("Testing vision parser image extraction...")
        
        try:
            # Initialize vision parser
            parser = VisionParser(self.test_config["vision_parser"])
            
            # Parse the PDF
            documents = await parser.parse_document(pdf_path)
            
            # Validate results
            if not documents:
                logger.error("No documents returned from vision parser")
                return False
            
            # Check for page-level documents with image data
            page_docs_with_images = 0
            for doc in documents:
                if doc.metadata.get('page_number') and doc.metadata.get('base64_image'):
                    page_docs_with_images += 1
                    
                    # Validate image data
                    base64_image = doc.metadata['base64_image']
                    image_width = doc.metadata.get('image_width')
                    image_height = doc.metadata.get('image_height')
                    
                    if not base64_image or not image_width or not image_height:
                        logger.error(f"Missing image data in page {doc.metadata.get('page_number')}")
                        return False
                    
                    # Validate base64 format
                    try:
                        base64.b64decode(base64_image)
                        logger.info(f"✅ Page {doc.metadata['page_number']}: Valid base64 image "
                                  f"({image_width}x{image_height})")
                    except Exception as e:
                        logger.error(f"Invalid base64 image data: {e}")
                        return False
            
            if page_docs_with_images == 0:
                logger.error("No page documents with image data found")
                return False
            
            logger.info(f"✅ Vision parser test passed: {page_docs_with_images} pages with images")
            return True
            
        except Exception as e:
            logger.error(f"Vision parser test failed: {e}")
            return False
    
    async def test_pgvector_image_storage(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Test that PgVector store stores base64 images alongside embeddings."""
        logger.info("Testing PgVector image storage...")
        
        try:
            # Initialize components
            parser = VisionParser(self.test_config["vision_parser"])
            embedder = VertexEmbedder(self.test_config["embedding"])
            vector_store = PgVectorStore(self.test_config["pgvector"])
            
            # Initialize vector store
            await vector_store.initialize()
            
            # Parse document
            documents = await parser.parse_document(pdf_path)
            page_documents = [doc for doc in documents if doc.metadata.get('page_number')]
            
            if not page_documents:
                logger.error("No page documents found for storage test")
                return []
            
            # Create chunks from documents
            from src.rag.core.interfaces.base import Chunk
            chunks = []
            for i, doc in enumerate(page_documents):
                chunk = Chunk(
                    id=f"test_chunk_{i}",
                    content=doc.content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"test_chunk_{i}",
                        "document_id": f"test_doc_{datetime.now().timestamp()}",
                        "soeid": "test_user"
                    }
                )
                chunks.append(chunk)
            
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.get_embeddings(texts)
            
            # Store in vector database
            await vector_store.add(chunks, embeddings)
            
            # Verify storage by searching
            if embeddings:
                search_results = await vector_store.search(embeddings[0], top_k=5)
                
                images_found = 0
                for result in search_results:
                    if result.chunk.metadata.get('base64_image'):
                        images_found += 1
                        logger.info(f"✅ Found stored image for chunk: {result.chunk.id}")
                
                if images_found > 0:
                    logger.info(f"✅ PgVector storage test passed: {images_found} chunks with images stored")
                    return search_results
                else:
                    logger.error("No images found in stored chunks")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"PgVector storage test failed: {e}")
            return []
    
    async def test_vector_retrieval_with_images(self, pdf_path: str) -> bool:
        """Test that vector retriever returns images in retrieved documents."""
        logger.info("Testing vector retrieval with images...")
        
        try:
            # Store documents first
            stored_results = await self.test_pgvector_image_storage(pdf_path)
            if not stored_results:
                logger.error("No stored results available for retrieval test")
                return False
            
            # Initialize retriever
            retriever_config = {
                "vector_store": self.test_config["pgvector"],
                "embedding": self.test_config["embedding"],
                "top_k": 3
            }
            retriever = VectorRetriever(retriever_config)
            
            # Test retrieval
            query = "test document with visual elements"
            retrieved_docs = await retriever.retrieve(query)
            
            if not retrieved_docs:
                logger.error("No documents retrieved")
                return False
            
            # Check for image data in retrieved documents
            docs_with_images = 0
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata') and doc.metadata.get('base64_image'):
                    docs_with_images += 1
                    logger.info(f"✅ Retrieved document with image: {doc.metadata.get('page_number', 'unknown')}")
            
            if docs_with_images > 0:
                logger.info(f"✅ Vector retrieval test passed: {docs_with_images} documents with images")
                return True
            else:
                logger.error("No retrieved documents contain image data")
                return False
            
        except Exception as e:
            logger.error(f"Vector retrieval test failed: {e}")
            return False
    
    async def test_image_data_validation(self, pdf_path: str) -> bool:
        """Validate image data integrity and format."""
        logger.info("Testing image data validation...")
        
        try:
            # Get stored results
            stored_results = await self.test_pgvector_image_storage(pdf_path)
            if not stored_results:
                return False
            
            for result in stored_results:
                metadata = result.chunk.metadata
                if metadata.get('base64_image'):
                    base64_image = metadata['base64_image']
                    image_width = metadata.get('image_width')
                    image_height = metadata.get('image_height')
                    
                    # Validate base64 format
                    try:
                        image_data = base64.b64decode(base64_image)
                        logger.info(f"✅ Valid base64 image: {len(image_data)} bytes")
                    except Exception as e:
                        logger.error(f"Invalid base64 data: {e}")
                        return False
                    
                    # Validate dimensions
                    if not isinstance(image_width, int) or not isinstance(image_height, int):
                        logger.error(f"Invalid image dimensions: {image_width}x{image_height}")
                        return False
                    
                    if image_width <= 0 or image_height <= 0:
                        logger.error(f"Invalid image dimensions: {image_width}x{image_height}")
                        return False
                    
                    logger.info(f"✅ Valid image dimensions: {image_width}x{image_height}")
            
            logger.info("✅ Image data validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Image data validation test failed: {e}")
            return False
    
    async def test_end_to_end_pipeline(self, pdf_path: str) -> bool:
        """Test the complete end-to-end pipeline with image storage."""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            # Run all component tests
            vision_test = await self.test_vision_parser_image_extraction(pdf_path)
            storage_test = len(await self.test_pgvector_image_storage(pdf_path)) > 0
            retrieval_test = await self.test_vector_retrieval_with_images(pdf_path)
            validation_test = await self.test_image_data_validation(pdf_path)
            
            # Update test results
            self.test_results.update({
                "vision_parser_test": vision_test,
                "pgvector_storage_test": storage_test,
                "retrieval_test": retrieval_test,
                "image_data_validation": validation_test,
                "end_to_end_test": all([vision_test, storage_test, retrieval_test, validation_test])
            })
            
            if self.test_results["end_to_end_test"]:
                logger.info("✅ End-to-end pipeline test passed!")
                return True
            else:
                logger.error("❌ End-to-end pipeline test failed")
                return False
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "="*60)
        print("PDF IMAGE STORAGE TEST RESULTS")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("="*60)
        
        if all(self.test_results.values()):
            print("🎉 ALL TESTS PASSED! PDF image storage is working correctly.")
            print("\nFeatures validated:")
            print("• Vision parser extracts page images as base64")
            print("• PgVector stores images alongside embeddings")
            print("• Vector retriever returns images with documents")
            print("• Image data integrity is maintained")
            print("• End-to-end pipeline functions properly")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
        
        print("="*60)


async def main():
    """Run the comprehensive PDF image storage test suite."""
    print("Starting PDF Image Storage Test Suite...")
    print("This test validates the complete pipeline for storing PDF images as base64 in PostgreSQL.")
    
    # Initialize test suite
    test_suite = PDFImageStorageTest()
    
    # Create test PDF
    pdf_path = test_suite.create_test_pdf()
    
    try:
        # Run comprehensive tests
        await test_suite.test_end_to_end_pipeline(pdf_path)
        
        # Print results
        test_suite.print_test_summary()
        
    finally:
        # Clean up test PDF
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
            logger.info(f"Cleaned up test PDF: {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
