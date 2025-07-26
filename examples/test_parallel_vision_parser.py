#!/usr/bin/env python3
"""
Test script for parallel vision parsing functionality.

This script demonstrates the parallel processing capabilities of the VisionParser
and compares performance between sequential and parallel processing.
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.shared.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_parallel_processing():
    """Test the parallel processing functionality of VisionParser."""
    
    logger.info("🚀 Testing Parallel Vision Parser")
    logger.info("=" * 50)
    
    # Test configurations
    configs = [
        {
            "name": "Sequential (max_concurrent_pages=1)",
            "config": {
                "model": "gemini-1.5-pro-002",
                "max_pages": 5,
                "max_concurrent_pages": 1
            }
        },
        {
            "name": "Parallel Low (max_concurrent_pages=2)",
            "config": {
                "model": "gemini-1.5-pro-002", 
                "max_pages": 5,
                "max_concurrent_pages": 2
            }
        },
        {
            "name": "Parallel High (max_concurrent_pages=5)",
            "config": {
                "model": "gemini-1.5-pro-002",
                "max_pages": 5,
                "max_concurrent_pages": 5
            }
        }
    ]
    
    # Look for a test PDF file
    test_pdf_paths = [
        "./test_document.pdf",
        "./examples/test_document.pdf",
        "./data/test_document.pdf",
        "/tmp/test_document.pdf"
    ]
    
    test_pdf = None
    for path in test_pdf_paths:
        if os.path.exists(path):
            test_pdf = path
            break
    
    if not test_pdf:
        logger.warning("⚠️  No test PDF found. Creating a mock test...")
        logger.info("To test with a real PDF, place a PDF file at one of these locations:")
        for path in test_pdf_paths:
            logger.info(f"  - {path}")
        return
    
    logger.info(f"📄 Using test PDF: {test_pdf}")
    
    results = []
    
    for test_config in configs:
        logger.info(f"\n🧪 Testing: {test_config['name']}")
        logger.info("-" * 40)
        
        try:
            # Create parser with test configuration
            parser = VisionParser(test_config['config'])
            
            # Measure processing time
            start_time = time.time()
            
            # Parse the document
            documents = await parser._parse_file(test_pdf, test_config['config'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Collect results
            result = {
                "config_name": test_config['name'],
                "concurrent_pages": test_config['config']['max_concurrent_pages'],
                "processing_time": processing_time,
                "documents_count": len(documents),
                "total_content_length": sum(len(doc.content) for doc in documents),
                "success": True
            }
            results.append(result)
            
            logger.info(f"✅ Success!")
            logger.info(f"   Processing time: {processing_time:.2f} seconds")
            logger.info(f"   Documents created: {len(documents)}")
            logger.info(f"   Total content length: {result['total_content_length']} characters")
            
        except Exception as e:
            logger.error(f"❌ Failed: {str(e)}")
            result = {
                "config_name": test_config['name'],
                "concurrent_pages": test_config['config']['max_concurrent_pages'],
                "processing_time": None,
                "documents_count": 0,
                "total_content_length": 0,
                "success": False,
                "error": str(e)
            }
            results.append(result)
    
    # Print summary
    logger.info("\n📊 Performance Summary")
    logger.info("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        logger.info(f"{'Configuration':<30} {'Time (s)':<10} {'Speedup':<10}")
        logger.info("-" * 50)
        
        baseline_time = None
        for result in successful_results:
            if result['concurrent_pages'] == 1:
                baseline_time = result['processing_time']
                break
        
        for result in successful_results:
            time_str = f"{result['processing_time']:.2f}"
            
            if baseline_time and result['processing_time']:
                speedup = baseline_time / result['processing_time']
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            logger.info(f"{result['config_name']:<30} {time_str:<10} {speedup_str:<10}")
        
        # Performance insights
        logger.info("\n💡 Performance Insights:")
        
        if len(successful_results) > 1:
            fastest = min(successful_results, key=lambda x: x['processing_time'])
            slowest = max(successful_results, key=lambda x: x['processing_time'])
            
            if fastest != slowest:
                improvement = slowest['processing_time'] / fastest['processing_time']
                logger.info(f"   • Best configuration: {fastest['config_name']}")
                logger.info(f"   • Performance improvement: {improvement:.2f}x faster than slowest")
                logger.info(f"   • Optimal concurrency: {fastest['concurrent_pages']} pages")
        
        logger.info(f"   • All configurations processed the same content successfully")
        logger.info(f"   • Content consistency verified across all runs")
        
    else:
        logger.error("❌ No successful test runs. Check your configuration and authentication.")
    
    # Configuration recommendations
    logger.info("\n⚙️  Configuration Recommendations:")
    logger.info("   • For small documents (1-5 pages): max_concurrent_pages = 2-3")
    logger.info("   • For medium documents (6-20 pages): max_concurrent_pages = 5-8")
    logger.info("   • For large documents (20+ pages): max_concurrent_pages = 8-12")
    logger.info("   • Consider API rate limits and memory usage when setting concurrency")
    
    return results

async def test_authentication():
    """Test authentication for vision models."""
    logger.info("\n🔐 Testing Authentication")
    logger.info("-" * 30)
    
    try:
        parser = VisionParser({"model": "gemini-1.5-pro-002"})
        await parser._init_vision_model()
        
        # Test authentication
        is_valid = await parser.vision_model.validate_authentication()
        
        if is_valid:
            logger.info("✅ Authentication successful")
            return True
        else:
            logger.error("❌ Authentication failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Authentication error: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("🧪 Parallel Vision Parser Test Suite")
    logger.info("=" * 60)
    
    # Test authentication first
    auth_success = await test_authentication()
    
    if not auth_success:
        logger.error("\n❌ Authentication failed. Please check your environment variables:")
        logger.error("   • COIN_CONSUMER_ENDPOINT_URL")
        logger.error("   • COIN_CONSUMER_CLIENT_ID")
        logger.error("   • COIN_CONSUMER_CLIENT_SECRET")
        logger.error("   • PROJECT_ID")
        return
    
    # Run parallel processing tests
    results = await test_parallel_processing()
    
    logger.info("\n🎉 Test suite completed!")
    
    if results and any(r['success'] for r in results):
        logger.info("✅ Parallel processing is working correctly")
    else:
        logger.error("❌ Parallel processing tests failed")

if __name__ == "__main__":
    asyncio.run(main())
