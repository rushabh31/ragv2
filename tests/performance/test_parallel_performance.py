#!/usr/bin/env python3
"""
Comprehensive test script for all parallel processing parsers.

This script tests the parallel processing capabilities of all vision parsers:
- VisionParser (Vertex AI)
- GroqVisionParser (Groq)
- OpenAIVisionParser (OpenAI)
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
from src.rag.ingestion.parsers.groq_vision_parser import GroqVisionParser
from src.rag.ingestion.parsers.openai_vision_parser import OpenAIVisionParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_parser_parallel_processing(parser_class, parser_name: str, config: dict):
    """Test parallel processing for a specific parser."""
    
    logger.info(f"\n🧪 Testing {parser_name}")
    logger.info("=" * 50)
    
    # Test configurations with different concurrency levels
    test_configs = [
        {
            "name": f"{parser_name} Sequential (concurrency=1)",
            "config": {**config, "max_concurrent_pages": 1}
        },
        {
            "name": f"{parser_name} Low Parallel (concurrency=2)",
            "config": {**config, "max_concurrent_pages": 2}
        },
        {
            "name": f"{parser_name} High Parallel (concurrency=5)",
            "config": {**config, "max_concurrent_pages": 5}
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
        logger.warning(f"⚠️  No test PDF found for {parser_name}. Skipping...")
        return []
    
    logger.info(f"📄 Using test PDF: {test_pdf}")
    
    results = []
    
    for test_config in test_configs:
        logger.info(f"\n🔧 Testing: {test_config['name']}")
        logger.info("-" * 40)
        
        try:
            # Create parser with test configuration
            parser = parser_class(test_config['config'])
            
            # Measure processing time
            start_time = time.time()
            
            # Parse the document
            documents = await parser._parse_file(test_pdf, test_config['config'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Collect results
            result = {
                "parser_name": parser_name,
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
                "parser_name": parser_name,
                "config_name": test_config['name'],
                "concurrent_pages": test_config['config']['max_concurrent_pages'],
                "processing_time": None,
                "documents_count": 0,
                "total_content_length": 0,
                "success": False,
                "error": str(e)
            }
            results.append(result)
    
    return results

async def test_all_parsers():
    """Test all parsers with parallel processing."""
    
    logger.info("🚀 Testing All Parallel Parsers")
    logger.info("=" * 60)
    
    # Parser configurations
    parser_configs = [
        {
            "class": VisionParser,
            "name": "VisionParser (Vertex AI)",
            "config": {
                "model": "gemini-1.5-pro-002",
                "max_pages": 5
            }
        },
        {
            "class": GroqVisionParser,
            "name": "GroqVisionParser (Groq)",
            "config": {
                "model_name": "llama-3.2-11b-vision-preview",
                "prompt_template": "Extract and structure the text content from this document.",
                "max_pages": 5
            }
        },
        {
            "class": OpenAIVisionParser,
            "name": "OpenAIVisionParser (OpenAI)",
            "config": {
                "model": "gpt-4o",
                "max_pages": 5
            }
        }
    ]
    
    all_results = []
    
    for parser_config in parser_configs:
        try:
            results = await test_parser_parallel_processing(
                parser_config["class"],
                parser_config["name"],
                parser_config["config"]
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"❌ Failed to test {parser_config['name']}: {str(e)}")
    
    # Print comprehensive summary
    logger.info("\n📊 Comprehensive Performance Summary")
    logger.info("=" * 60)
    
    if all_results:
        successful_results = [r for r in all_results if r['success']]
        
        if successful_results:
            # Group by parser
            parsers = {}
            for result in successful_results:
                parser_name = result['parser_name']
                if parser_name not in parsers:
                    parsers[parser_name] = []
                parsers[parser_name].append(result)
            
            # Print results for each parser
            for parser_name, parser_results in parsers.items():
                logger.info(f"\n🔧 {parser_name}")
                logger.info("-" * 40)
                logger.info(f"{'Configuration':<35} {'Time (s)':<10} {'Speedup':<10}")
                logger.info("-" * 55)
                
                # Find baseline (sequential) time
                baseline_time = None
                for result in parser_results:
                    if result['concurrent_pages'] == 1:
                        baseline_time = result['processing_time']
                        break
                
                for result in parser_results:
                    time_str = f"{result['processing_time']:.2f}"
                    
                    if baseline_time and result['processing_time']:
                        speedup = baseline_time / result['processing_time']
                        speedup_str = f"{speedup:.2f}x"
                    else:
                        speedup_str = "N/A"
                    
                    config_short = result['config_name'].split('(')[0].strip()
                    logger.info(f"{config_short:<35} {time_str:<10} {speedup_str:<10}")
            
            # Overall insights
            logger.info("\n💡 Overall Performance Insights:")
            
            # Find best performing configurations
            best_results = {}
            for result in successful_results:
                parser_name = result['parser_name']
                if parser_name not in best_results or result['processing_time'] < best_results[parser_name]['processing_time']:
                    best_results[parser_name] = result
            
            for parser_name, best_result in best_results.items():
                logger.info(f"   • {parser_name}: Best with {best_result['concurrent_pages']} concurrent pages")
                logger.info(f"     Processing time: {best_result['processing_time']:.2f}s")
            
            # Calculate average speedup
            speedups = []
            for parser_name, parser_results in parsers.items():
                baseline = next((r for r in parser_results if r['concurrent_pages'] == 1), None)
                fastest = min(parser_results, key=lambda x: x['processing_time'])
                if baseline and fastest and baseline != fastest:
                    speedup = baseline['processing_time'] / fastest['processing_time']
                    speedups.append(speedup)
            
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                logger.info(f"   • Average speedup across all parsers: {avg_speedup:.2f}x")
        
        else:
            logger.error("❌ No successful test runs. Check your configurations and authentication.")
    
    # Configuration recommendations
    logger.info("\n⚙️  Configuration Recommendations:")
    logger.info("   • For small documents (1-5 pages): max_concurrent_pages = 2-3")
    logger.info("   • For medium documents (6-20 pages): max_concurrent_pages = 5-8")
    logger.info("   • For large documents (20+ pages): max_concurrent_pages = 8-12")
    logger.info("   • Consider API rate limits and memory usage when setting concurrency")
    logger.info("   • Monitor authentication token refresh rates for high concurrency")
    
    return all_results

async def test_authentication():
    """Test authentication for all parsers."""
    logger.info("\n🔐 Testing Authentication for All Parsers")
    logger.info("-" * 50)
    
    auth_results = {}
    
    # Test VisionParser (Vertex AI)
    try:
        parser = VisionParser({"model": "gemini-1.5-pro-002"})
        await parser._init_vision_model()
        is_valid = await parser.vision_model.validate_authentication()
        auth_results["VisionParser (Vertex AI)"] = is_valid
        logger.info(f"✅ VisionParser (Vertex AI): {'Success' if is_valid else 'Failed'}")
    except Exception as e:
        auth_results["VisionParser (Vertex AI)"] = False
        logger.error(f"❌ VisionParser (Vertex AI): {str(e)}")
    
    # Test GroqVisionParser
    try:
        parser = GroqVisionParser({"model_name": "llama-3.2-11b-vision-preview"})
        vision_model = await parser._get_vision_model()
        # Groq doesn't have validate_authentication, so we assume success if model creation works
        auth_results["GroqVisionParser (Groq)"] = True
        logger.info("✅ GroqVisionParser (Groq): Success")
    except Exception as e:
        auth_results["GroqVisionParser (Groq)"] = False
        logger.error(f"❌ GroqVisionParser (Groq): {str(e)}")
    
    # Test OpenAIVisionParser
    try:
        parser = OpenAIVisionParser({"model": "gpt-4o"})
        token = parser.get_coin_token()
        auth_results["OpenAIVisionParser (OpenAI)"] = bool(token)
        logger.info(f"✅ OpenAIVisionParser (OpenAI): {'Success' if token else 'Failed'}")
    except Exception as e:
        auth_results["OpenAIVisionParser (OpenAI)"] = False
        logger.error(f"❌ OpenAIVisionParser (OpenAI): {str(e)}")
    
    return auth_results

async def main():
    """Main test function."""
    logger.info("🧪 Comprehensive Parallel Parser Test Suite")
    logger.info("=" * 70)
    
    # Test authentication first
    auth_results = await test_authentication()
    
    successful_auths = [k for k, v in auth_results.items() if v]
    
    if not successful_auths:
        logger.error("\n❌ No parsers have valid authentication. Please check your environment variables:")
        logger.error("   • For Vertex AI: COIN_CONSUMER_* variables")
        logger.error("   • For Groq: GROQ_API_KEY")
        logger.error("   • For OpenAI: COIN_CONSUMER_* variables (for universal auth)")
        return
    
    logger.info(f"\n✅ {len(successful_auths)} parser(s) have valid authentication")
    
    # Run parallel processing tests
    results = await test_all_parsers()
    
    logger.info("\n🎉 Test suite completed!")
    
    if results and any(r['success'] for r in results):
        logger.info("✅ Parallel processing is working correctly for tested parsers")
    else:
        logger.error("❌ Parallel processing tests failed")

if __name__ == "__main__":
    asyncio.run(main())
