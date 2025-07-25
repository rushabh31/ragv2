#!/usr/bin/env python3
"""
Comprehensive test script to validate the factory-based multi-provider system.

This script tests:
1. Generation model factory with all providers
2. Embedding model factory with all providers  
3. Vision model factory with all providers
4. RAG component integration with factories
5. Configuration-driven model selection

Run this script to validate the complete factory system after setting environment variables.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.generation.model_factory import GenerationModelFactory
from src.models.embedding.embedding_factory import EmbeddingModelFactory
from src.models.vision.vision_factory import VisionModelFactory
from src.rag.ingestion.parsers.vision_parser import VisionParser
from src.rag.chatbot.generators.vertex_generator import VertexGenerator
from src.rag.ingestion.embedders.vertex_embedder import VertexEmbedder
from src.rag.ingestion.chunkers.semantic_chunker import SemanticChunker
from src.rag.shared.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FactorySystemTester:
    """Comprehensive tester for the factory-based multi-provider system."""
    
    def __init__(self):
        self.results = {}
        self.config_manager = ConfigManager()
        
    async def test_generation_models(self) -> Dict[str, Any]:
        """Test all generation model providers."""
        logger.info("🚀 Testing Generation Models...")
        results = {}
        
        providers = ["vertex", "anthropic_vertex", "openai", "azure_openai", "groq"]
        
        for provider in providers:
            try:
                logger.info(f"  Testing {provider} generation model...")
                
                # Create model using factory
                model = GenerationModelFactory.create_model(provider=provider)
                
                # Test authentication
                auth_valid = await model.validate_authentication()
                results[provider] = {
                    "model_created": True,
                    "auth_valid": auth_valid,
                    "error": None
                }
                
                if auth_valid:
                    # Test basic generation
                    try:
                        response = await model.generate_content("Hello, how are you?")
                        results[provider]["generation_works"] = bool(response)
                        results[provider]["response_preview"] = response[:100] if response else None
                    except Exception as e:
                        results[provider]["generation_works"] = False
                        results[provider]["generation_error"] = str(e)
                
                logger.info(f"    ✅ {provider}: Created={results[provider]['model_created']}, Auth={results[provider]['auth_valid']}")
                
            except Exception as e:
                results[provider] = {
                    "model_created": False,
                    "auth_valid": False,
                    "error": str(e)
                }
                logger.error(f"    ❌ {provider}: {str(e)}")
        
        return results
    
    async def test_embedding_models(self) -> Dict[str, Any]:
        """Test all embedding model providers."""
        logger.info("🔗 Testing Embedding Models...")
        results = {}
        
        providers = ["vertex_ai", "openai_universal", "azure_openai", "sentence_transformer"]
        
        for provider in providers:
            try:
                logger.info(f"  Testing {provider} embedding model...")
                
                # Create model using factory
                model = EmbeddingModelFactory.create_model(provider=provider)
                
                # Test authentication
                auth_valid = await model.validate_authentication()
                results[provider] = {
                    "model_created": True,
                    "auth_valid": auth_valid,
                    "error": None
                }
                
                if auth_valid:
                    # Test basic embedding
                    try:
                        embeddings = await model.get_embeddings(["Hello world", "Test embedding"])
                        results[provider]["embedding_works"] = bool(embeddings and len(embeddings) == 2)
                        results[provider]["embedding_dimension"] = len(embeddings[0]) if embeddings else 0
                    except Exception as e:
                        results[provider]["embedding_works"] = False
                        results[provider]["embedding_error"] = str(e)
                
                logger.info(f"    ✅ {provider}: Created={results[provider]['model_created']}, Auth={results[provider]['auth_valid']}")
                
            except Exception as e:
                results[provider] = {
                    "model_created": False,
                    "auth_valid": False,
                    "error": str(e)
                }
                logger.error(f"    ❌ {provider}: {str(e)}")
        
        return results
    
    async def test_vision_models(self) -> Dict[str, Any]:
        """Test all vision model providers."""
        logger.info("👁️ Testing Vision Models...")
        results = {}
        
        providers = ["vertex_ai", "groq"]  # More can be added later
        
        for provider in providers:
            try:
                logger.info(f"  Testing {provider} vision model...")
                
                # Create model using factory
                model = VisionModelFactory.create_model(provider=provider)
                
                # Test authentication
                auth_valid = await model.validate_authentication()
                results[provider] = {
                    "model_created": True,
                    "auth_valid": auth_valid,
                    "error": None
                }
                
                logger.info(f"    ✅ {provider}: Created={results[provider]['model_created']}, Auth={results[provider]['auth_valid']}")
                
            except Exception as e:
                results[provider] = {
                    "model_created": False,
                    "auth_valid": False,
                    "error": str(e)
                }
                logger.error(f"    ❌ {provider}: {str(e)}")
        
        return results
    
    async def test_rag_components(self) -> Dict[str, Any]:
        """Test RAG components with factory-based models."""
        logger.info("🧩 Testing RAG Components...")
        results = {}
        
        # Test Vision Parser
        try:
            logger.info("  Testing VisionParser with factory...")
            parser = VisionParser({"model": "gemini-1.5-pro-002"})
            results["vision_parser"] = {
                "created": True,
                "provider": getattr(parser, 'vision_provider', 'unknown'),
                "error": None
            }
            logger.info(f"    ✅ VisionParser: Provider={results['vision_parser']['provider']}")
        except Exception as e:
            results["vision_parser"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ VisionParser: {str(e)}")
        
        # Test VertexGenerator
        try:
            logger.info("  Testing VertexGenerator with factory...")
            generator = VertexGenerator({"model_name": "gemini-1.5-pro-002"})
            results["vertex_generator"] = {
                "created": True,
                "provider": getattr(generator, 'generation_provider', 'unknown'),
                "error": None
            }
            logger.info(f"    ✅ VertexGenerator: Provider={results['vertex_generator']['provider']}")
        except Exception as e:
            results["vertex_generator"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ VertexGenerator: {str(e)}")
        
        # Test VertexEmbedder
        try:
            logger.info("  Testing VertexEmbedder with factory...")
            embedder = VertexEmbedder({"model": "text-embedding-004"})
            results["vertex_embedder"] = {
                "created": True,
                "provider": getattr(embedder, 'embedding_provider', 'unknown'),
                "error": None
            }
            logger.info(f"    ✅ VertexEmbedder: Provider={results['vertex_embedder']['provider']}")
        except Exception as e:
            results["vertex_embedder"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ VertexEmbedder: {str(e)}")
        
        # Test SemanticChunker
        try:
            logger.info("  Testing SemanticChunker with factory...")
            chunker = SemanticChunker({"use_llm_boundary": True})
            results["semantic_chunker"] = {
                "created": True,
                "provider": getattr(chunker, 'generation_provider', 'unknown'),
                "error": None
            }
            logger.info(f"    ✅ SemanticChunker: Provider={results['semantic_chunker']['provider']}")
        except Exception as e:
            results["semantic_chunker"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ SemanticChunker: {str(e)}")
        
        return results
    
    async def test_configuration_loading(self) -> Dict[str, Any]:
        """Test configuration loading for all model types."""
        logger.info("⚙️ Testing Configuration Loading...")
        results = {}
        
        # Test vision config
        try:
            vision_config = self.config_manager.get_config("vision")
            results["vision_config"] = {
                "loaded": vision_config is not None,
                "provider": vision_config.get("provider") if vision_config else None,
                "config_keys": list(vision_config.get("config", {}).keys()) if vision_config else []
            }
            logger.info(f"    ✅ Vision Config: Provider={results['vision_config']['provider']}")
        except Exception as e:
            results["vision_config"] = {"loaded": False, "error": str(e)}
            logger.error(f"    ❌ Vision Config: {str(e)}")
        
        # Test generation config
        try:
            generation_config = self.config_manager.get_config("generation")
            results["generation_config"] = {
                "loaded": generation_config is not None,
                "provider": generation_config.get("provider") if generation_config else None,
                "config_keys": list(generation_config.get("config", {}).keys()) if generation_config else []
            }
            logger.info(f"    ✅ Generation Config: Provider={results['generation_config']['provider']}")
        except Exception as e:
            results["generation_config"] = {"loaded": False, "error": str(e)}
            logger.error(f"    ❌ Generation Config: {str(e)}")
        
        # Test embedding config
        try:
            embedding_config = self.config_manager.get_config("embedding")
            results["embedding_config"] = {
                "loaded": embedding_config is not None,
                "provider": embedding_config.get("provider") if embedding_config else None,
                "config_keys": list(embedding_config.get("config", {}).keys()) if embedding_config else []
            }
            logger.info(f"    ✅ Embedding Config: Provider={results['embedding_config']['provider']}")
        except Exception as e:
            results["embedding_config"] = {"loaded": False, "error": str(e)}
            logger.error(f"    ❌ Embedding Config: {str(e)}")
        
        return results
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables."""
        logger.info("🌍 Checking Environment Variables...")
        
        required_vars = [
            "COIN_CONSUMER_ENDPOINT_URL",
            "COIN_CONSUMER_CLIENT_ID", 
            "COIN_CONSUMER_CLIENT_SECRET",
            "COIN_CONSUMER_SCOPE",
            "PROJECT_ID",
            "VERTEXAI_API_ENDPOINT",
            "GROQ_API_KEY"
        ]
        
        results = {}
        for var in required_vars:
            value = os.getenv(var)
            results[var] = {
                "set": value is not None,
                "value_preview": value[:20] + "..." if value and len(value) > 20 else value
            }
            status = "✅" if value else "❌"
            logger.info(f"    {status} {var}: {'Set' if value else 'Not set'}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🧪 Starting Factory System Tests...")
        
        # Check environment first
        env_results = self.check_environment_variables()
        
        # Run all tests
        generation_results = await self.test_generation_models()
        embedding_results = await self.test_embedding_models()
        vision_results = await self.test_vision_models()
        rag_results = await self.test_rag_components()
        config_results = await self.test_configuration_loading()
        
        # Compile final results
        final_results = {
            "environment": env_results,
            "generation_models": generation_results,
            "embedding_models": embedding_results,
            "vision_models": vision_results,
            "rag_components": rag_results,
            "configuration": config_results,
            "summary": self._generate_summary(
                env_results, generation_results, embedding_results, 
                vision_results, rag_results, config_results
            )
        }
        
        return final_results
    
    def _generate_summary(self, env_results, gen_results, emb_results, vis_results, rag_results, config_results) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        
        # Count successes
        env_success = sum(1 for r in env_results.values() if r["set"])
        gen_success = sum(1 for r in gen_results.values() if r.get("model_created", False))
        emb_success = sum(1 for r in emb_results.values() if r.get("model_created", False))
        vis_success = sum(1 for r in vis_results.values() if r.get("model_created", False))
        rag_success = sum(1 for r in rag_results.values() if r.get("created", False))
        config_success = sum(1 for r in config_results.values() if r.get("loaded", False))
        
        # Count totals
        env_total = len(env_results)
        gen_total = len(gen_results)
        emb_total = len(emb_results)
        vis_total = len(vis_results)
        rag_total = len(rag_results)
        config_total = len(config_results)
        
        return {
            "environment_vars": f"{env_success}/{env_total}",
            "generation_models": f"{gen_success}/{gen_total}",
            "embedding_models": f"{emb_success}/{emb_total}",
            "vision_models": f"{vis_success}/{vis_total}",
            "rag_components": f"{rag_success}/{rag_total}",
            "configurations": f"{config_success}/{config_total}",
            "overall_health": "Good" if all([
                env_success >= env_total * 0.8,
                gen_success > 0,
                emb_success > 0,
                vis_success > 0,
                rag_success >= rag_total * 0.8,
                config_success >= config_total * 0.8
            ]) else "Needs Attention"
        }

async def main():
    """Main test execution."""
    print("=" * 80)
    print("🏭 FACTORY-BASED MULTI-PROVIDER SYSTEM TEST")
    print("=" * 80)
    
    tester = FactorySystemTester()
    results = await tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    summary = results["summary"]
    
    print(f"Environment Variables: {summary['environment_vars']}")
    print(f"Generation Models:     {summary['generation_models']}")
    print(f"Embedding Models:      {summary['embedding_models']}")
    print(f"Vision Models:         {summary['vision_models']}")
    print(f"RAG Components:        {summary['rag_components']}")
    print(f"Configurations:        {summary['configurations']}")
    print(f"Overall Health:        {summary['overall_health']}")
    
    print("\n" + "=" * 80)
    print("✅ Factory system test completed!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
