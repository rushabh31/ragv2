#!/usr/bin/env python3
"""
Debug script to check the vector store state and diagnose retrieval issues.
"""

import asyncio
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_vector_store():
    """Debug the vector store configuration and state."""
    
    print("🔍 Vector Store Diagnostic Tool")
    print("=" * 50)
    
    try:
        # Import required modules
        from src.rag.shared.utils.config_manager import ConfigManager
        from src.rag.ingestion.indexers.faiss_vector_store import FAISSVectorStore
        from src.models.embedding.embedding_factory import EmbeddingModelFactory
        
        # 1. Check configuration
        print("\n📋 1. Checking Configuration...")
        config_manager = ConfigManager()
        
        # Check chatbot retrieval config
        chatbot_config = config_manager.get_section("chatbot", {})
        retrieval_config = chatbot_config.get("retrieval", {})
        vector_store_config = retrieval_config.get("vector_store", {})
        
        print(f"   Chatbot config found: {bool(chatbot_config)}")
        print(f"   Retrieval config found: {bool(retrieval_config)}")
        print(f"   Vector store config: {vector_store_config}")
        
        # Check embedding config
        embedding_config = config_manager.get_section("embedding", {})
        print(f"   Embedding config: {embedding_config}")
        
        # 2. Check data directory and index files
        print("\n📁 2. Checking Data Directory and Index Files...")
        data_dir = "./data"
        index_dir = f"{data_dir}/indices"
        index_path = f"{index_dir}/faiss.index"
        metadata_path = f"{index_dir}/metadata.pickle"
        
        print(f"   Data directory: {data_dir}")
        print(f"   Index directory: {index_dir}")
        print(f"   Index path: {index_path}")
        print(f"   Metadata path: {metadata_path}")
        
        print(f"   Data dir exists: {os.path.exists(data_dir)}")
        print(f"   Index dir exists: {os.path.exists(index_dir)}")
        print(f"   Index file exists: {os.path.exists(index_path)}")
        print(f"   Metadata file exists: {os.path.exists(metadata_path)}")
        
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path)
            print(f"   Index file size: {index_size} bytes")
        
        if os.path.exists(metadata_path):
            metadata_size = os.path.getsize(metadata_path)
            print(f"   Metadata file size: {metadata_size} bytes")
        
        # 3. Test embedding model
        print("\n🧠 3. Testing Embedding Model...")
        try:
            embedder = EmbeddingModelFactory.create_model()
            print(f"   Embedder created: {type(embedder).__name__}")
            
            # Test embedding generation
            test_text = "Hello world"
            embedding = await embedder.embed_single(test_text)
            print(f"   Test embedding generated: dimension={len(embedding)}")
            
        except Exception as e:
            print(f"   ❌ Embedder error: {str(e)}")
        
        # 4. Test vector store initialization
        print("\n🗄️ 4. Testing Vector Store...")
        try:
            # Configure vector store with proper paths
            vs_config = {
                "type": "faiss",
                "dimension": 768,  # Default for text-embedding-004
                "index_type": "HNSW",
                "index_path": index_path,
                "metadata_path": metadata_path
            }
            
            vector_store = FAISSVectorStore(vs_config)
            await vector_store.initialize()
            
            print(f"   Vector store initialized: {vector_store._initialized}")
            print(f"   Index total vectors: {vector_store.index.ntotal if vector_store.index else 'No index'}")
            print(f"   Chunks stored: {len(vector_store.chunks)}")
            
            # Show some chunk info if available
            if vector_store.chunks:
                print(f"   First chunk preview: {vector_store.chunks[0].content[:100]}...")
                print(f"   First chunk metadata: {vector_store.chunks[0].metadata}")
            
            # Test search with the vector store
            if vector_store.index and vector_store.index.ntotal > 0:
                print("\n🔍 5. Testing Vector Search...")
                test_query = "What is machine learning?"
                query_embedding = await embedder.embed_single(test_query)
                
                search_results = await vector_store._search_vectors(query_embedding, top_k=3)
                print(f"   Search results: {len(search_results)}")
                
                for i, result in enumerate(search_results):
                    print(f"   Result {i+1}: score={result.score:.4f}, content={result.chunk.content[:100]}...")
            else:
                print("\n⚠️ 5. Vector Store is Empty")
                print("   No vectors in the index. You need to ingest documents first.")
                print("   Use the ingestion API to upload and process documents.")
        
        except Exception as e:
            print(f"   ❌ Vector store error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 6. Recommendations
        print("\n💡 6. Recommendations:")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("   • No index files found - you need to ingest documents first")
            print("   • Use the ingestion API to upload documents: POST /ingest/upload")
            print("   • Check the ingestion logs for any processing errors")
        
        elif os.path.exists(index_path) and os.path.getsize(index_path) == 0:
            print("   • Index file exists but is empty")
            print("   • Check ingestion logs for processing errors")
            print("   • Verify document parsing and embedding generation worked")
        
        else:
            print("   • Index files exist - check vector store initialization")
            print("   • Verify embedding model compatibility")
            print("   • Check configuration paths match between ingestion and chatbot")
        
        print("\n✅ Diagnostic complete!")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_vector_store())
