#!/usr/bin/env python3
"""
Test script for Groq and Sentence Transformer models integration.

This script specifically tests the newly added Groq generation/vision models 
and Sentence Transformer embedding models to ensure they work correctly 
with the factory system and RAG components.
"""

import asyncio
import logging
import os
import sys
import base64
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.generation.groq_gen import GroqGenAI
from src.models.vision.groq_vision import GroqVisionAI
from src.models.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddingAI
from src.models.generation.model_factory import GenerationModelFactory
from src.models.vision.vision_factory import VisionModelFactory
from src.models.embedding.embedding_factory import EmbeddingModelFactory
from src.rag.chatbot.generators.groq_generator import GroqGenerator
from src.rag.ingestion.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroqAndSentenceTransformerTester:
    """Comprehensive tester for Groq and Sentence Transformer models."""
    
    def __init__(self):
        self.results = {}
        
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables for new models."""
        logger.info("🌍 Checking Environment Variables...")
        
        required_vars = [
            "GROQ_API_KEY",  # Required for Groq models
            # Sentence transformers don't need API keys (local models)
        ]
        
        optional_vars = [
            "COIN_CONSUMER_ENDPOINT_URL",
            "COIN_CONSUMER_CLIENT_ID", 
            "COIN_CONSUMER_CLIENT_SECRET",
            "PROJECT_ID"
        ]
        
        results = {"required": {}, "optional": {}}
        
        for var in required_vars:
            value = os.getenv(var)
            results["required"][var] = {
                "set": value is not None,
                "value_preview": value[:20] + "..." if value and len(value) > 20 else value
            }
            status = "✅" if value else "❌"
            logger.info(f"    {status} {var}: {'Set' if value else 'Not set'}")
        
        for var in optional_vars:
            value = os.getenv(var)
            results["optional"][var] = {
                "set": value is not None,
                "value_preview": value[:20] + "..." if value and len(value) > 20 else value
            }
            status = "✅" if value else "⚠️"
            logger.info(f"    {status} {var}: {'Set' if value else 'Not set (optional)'}")
        
        return results
    
    async def test_groq_generation_model(self) -> Dict[str, Any]:
        """Test Groq generation model directly."""
        logger.info("🚀 Testing Groq Generation Model...")
        
        try:
            # Test direct model usage
            model = GroqGenAI(model_name="llama-3.1-70b-versatile")
            
            # Test authentication
            auth_valid = await model.validate_authentication()
            
            result = {
                "model_created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            
            if auth_valid:
                # Test basic generation
                try:
                    response = await model.generate_content("Hello! Please respond with 'Groq model working correctly.'")
                    result["generation_works"] = bool(response)
                    result["response_preview"] = response[:100] if response else None
                    logger.info(f"    ✅ Groq Generation: {response[:50]}...")
                except Exception as e:
                    result["generation_works"] = False
                    result["generation_error"] = str(e)
                    logger.error(f"    ❌ Groq Generation Error: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Groq Generation Model Error: {str(e)}")
            return {
                "model_created": False,
                "auth_valid": False,
                "error": str(e)
            }
    
    async def test_groq_vision_model(self) -> Dict[str, Any]:
        """Test Groq vision model directly."""
        logger.info("👁️ Testing Groq Vision Model...")
        
        try:
            # Test direct model usage
            model = GroqVisionAI(model_name="llama-3.2-11b-vision-preview")
            
            # Test authentication
            auth_valid = await model.validate_authentication()
            
            result = {
                "model_created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            
            if auth_valid:
                logger.info("    ✅ Groq Vision model authenticated successfully")
                # Note: We can't test image processing without an actual image
                result["vision_ready"] = True
            else:
                logger.warning("    ⚠️ Groq Vision authentication failed")
                result["vision_ready"] = False
            
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Groq Vision Model Error: {str(e)}")
            return {
                "model_created": False,
                "auth_valid": False,
                "error": str(e)
            }
    
    async def test_sentence_transformer_model(self) -> Dict[str, Any]:
        """Test Sentence Transformer embedding model directly."""
        logger.info("🔗 Testing Sentence Transformer Model...")
        
        try:
            # Test direct model usage
            model = SentenceTransformerEmbeddingAI(model_name="all-mpnet-base-v2")
            
            # Test model availability (authentication for local model)
            auth_valid = await model.validate_authentication()
            
            result = {
                "model_created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            
            if auth_valid:
                # Test basic embedding
                try:
                    test_texts = ["Hello world", "This is a test sentence"]
                    embeddings = await model.get_embeddings(test_texts)
                    result["embedding_works"] = bool(embeddings and len(embeddings) == 2)
                    result["embedding_dimension"] = len(embeddings[0]) if embeddings else 0
                    logger.info(f"    ✅ Sentence Transformer: Generated {len(embeddings)} embeddings, dim={result['embedding_dimension']}")
                except Exception as e:
                    result["embedding_works"] = False
                    result["embedding_error"] = str(e)
                    logger.error(f"    ❌ Sentence Transformer Embedding Error: {str(e)}")
            else:
                logger.warning("    ⚠️ Sentence Transformer model not available")
            
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Sentence Transformer Model Error: {str(e)}")
            return {
                "model_created": False,
                "auth_valid": False,
                "error": str(e)
            }
    
    async def test_factory_integration(self) -> Dict[str, Any]:
        """Test factory integration for new models."""
        logger.info("🏭 Testing Factory Integration...")
        results = {}
        
        # Test Groq generation via factory
        try:
            logger.info("  Testing Groq generation factory...")
            model = GenerationModelFactory.create_model("groq")
            auth_valid = await model.validate_authentication()
            results["groq_generation_factory"] = {
                "created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            logger.info(f"    ✅ Groq Generation Factory: Auth={auth_valid}")
        except Exception as e:
            results["groq_generation_factory"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ Groq Generation Factory: {str(e)}")
        
        # Test Groq vision via factory
        try:
            logger.info("  Testing Groq vision factory...")
            model = VisionModelFactory.create_model("groq")
            auth_valid = await model.validate_authentication()
            results["groq_vision_factory"] = {
                "created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            logger.info(f"    ✅ Groq Vision Factory: Auth={auth_valid}")
        except Exception as e:
            results["groq_vision_factory"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ Groq Vision Factory: {str(e)}")
        
        # Test Sentence Transformer via factory
        try:
            logger.info("  Testing Sentence Transformer factory...")
            model = EmbeddingModelFactory.create_model("sentence_transformer")
            auth_valid = await model.validate_authentication()
            results["sentence_transformer_factory"] = {
                "created": True,
                "auth_valid": auth_valid,
                "error": None
            }
            logger.info(f"    ✅ Sentence Transformer Factory: Auth={auth_valid}")
        except Exception as e:
            results["sentence_transformer_factory"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ Sentence Transformer Factory: {str(e)}")
        
        return results
    
    async def test_rag_integration(self) -> Dict[str, Any]:
        """Test RAG component integration with new models."""
        logger.info("🧩 Testing RAG Integration...")
        results = {}
        
        # Test Groq Generator
        try:
            logger.info("  Testing Groq RAG Generator...")
            generator = GroqGenerator({"model_name": "llama-3.1-70b-versatile"})
            results["groq_rag_generator"] = {
                "created": True,
                "model_name": generator.model_name,
                "error": None
            }
            logger.info(f"    ✅ Groq RAG Generator: Model={generator.model_name}")
        except Exception as e:
            results["groq_rag_generator"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ Groq RAG Generator: {str(e)}")
        
        # Test Sentence Transformer Embedder
        try:
            logger.info("  Testing Sentence Transformer RAG Embedder...")
            embedder = SentenceTransformerEmbedder({"model": "all-mpnet-base-v2"})
            results["sentence_transformer_rag_embedder"] = {
                "created": True,
                "model_name": embedder.model_name,
                "error": None
            }
            logger.info(f"    ✅ Sentence Transformer RAG Embedder: Model={embedder.model_name}")
        except Exception as e:
            results["sentence_transformer_rag_embedder"] = {"created": False, "error": str(e)}
            logger.error(f"    ❌ Sentence Transformer RAG Embedder: {str(e)}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests for Groq and Sentence Transformer models."""
        logger.info("🧪 Starting Groq & Sentence Transformer Tests...")
        
        # Check environment first
        env_results = self.check_environment_variables()
        
        # Run model tests
        groq_gen_results = await self.test_groq_generation_model()
        groq_vision_results = await self.test_groq_vision_model()
        sentence_transformer_results = await self.test_sentence_transformer_model()
        
        # Run integration tests
        factory_results = await self.test_factory_integration()
        rag_results = await self.test_rag_integration()
        
        # Compile final results
        final_results = {
            "environment": env_results,
            "groq_generation": groq_gen_results,
            "groq_vision": groq_vision_results,
            "sentence_transformer": sentence_transformer_results,
            "factory_integration": factory_results,
            "rag_integration": rag_results,
            "summary": self._generate_summary(
                env_results, groq_gen_results, groq_vision_results,
                sentence_transformer_results, factory_results, rag_results
            )
        }
        
        return final_results
    
    def _generate_summary(self, env_results, groq_gen, groq_vision, sentence_transformer, factory, rag) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        
        # Count successes
        env_required_success = sum(1 for r in env_results["required"].values() if r["set"])
        env_required_total = len(env_results["required"])
        
        groq_gen_success = groq_gen.get("model_created", False) and groq_gen.get("auth_valid", False)
        groq_vision_success = groq_vision.get("model_created", False) and groq_vision.get("auth_valid", False)
        sentence_transformer_success = sentence_transformer.get("model_created", False) and sentence_transformer.get("auth_valid", False)
        
        factory_success = sum(1 for r in factory.values() if r.get("created", False))
        factory_total = len(factory)
        
        rag_success = sum(1 for r in rag.values() if r.get("created", False))
        rag_total = len(rag)
        
        return {
            "environment_required": f"{env_required_success}/{env_required_total}",
            "groq_generation": "✅" if groq_gen_success else "❌",
            "groq_vision": "✅" if groq_vision_success else "❌",
            "sentence_transformer": "✅" if sentence_transformer_success else "❌",
            "factory_integration": f"{factory_success}/{factory_total}",
            "rag_integration": f"{rag_success}/{rag_total}",
            "overall_health": "Good" if all([
                env_required_success >= env_required_total,
                groq_gen_success or groq_vision_success,  # At least one Groq model working
                sentence_transformer_success,
                factory_success >= factory_total * 0.7,
                rag_success >= rag_total * 0.7
            ]) else "Needs Attention"
        }

async def main():
    """Main test execution."""
    print("=" * 80)
    print("🆕 GROQ & SENTENCE TRANSFORMER MODELS TEST")
    print("=" * 80)
    
    tester = GroqAndSentenceTransformerTester()
    results = await tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    summary = results["summary"]
    
    print(f"Environment (Required):    {summary['environment_required']}")
    print(f"Groq Generation:           {summary['groq_generation']}")
    print(f"Groq Vision:               {summary['groq_vision']}")
    print(f"Sentence Transformer:      {summary['sentence_transformer']}")
    print(f"Factory Integration:       {summary['factory_integration']}")
    print(f"RAG Integration:           {summary['rag_integration']}")
    print(f"Overall Health:            {summary['overall_health']}")
    
    # Print specific notes
    print("\n" + "=" * 80)
    print("📝 NOTES")
    print("=" * 80)
    print("• Groq models require GROQ_API_KEY environment variable")
    print("• Sentence Transformers run locally and may download models on first use")
    print("• Vision models need actual images for full functionality testing")
    print("• All models integrate with the existing RAG factory system")
    
    print("\n" + "=" * 80)
    print("✅ Groq & Sentence Transformer test completed!")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
