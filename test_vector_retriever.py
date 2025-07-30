#!/usr/bin/env python3
"""
Test script to debug vector retriever initialization issues.
"""

import asyncio
import logging
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def test_vector_retriever():
    """Test the vector retriever initialization step by step."""
    
    print("🔍 Vector Retriever Debug Test")
    print("=" * 50)
    
    try:
        # Import required modules
        from src.rag.chatbot.retrievers.vector_retriever import VectorRetriever
        
        print("\n📋 1. Creating VectorRetriever...")
        
        # Create vector retriever with minimal config
        config = {
            "top_k": 5,
            "data_dir": "./data"
        }
        
        retriever = VectorRetriever(config)
        print(f"   ✅ VectorRetriever created: {retriever}")
        print(f"   Initial _vector_store: {retriever._vector_store}")
        print(f"   Initial _embedder: {retriever._embedder}")
        
        print("\n🔧 2. Testing _init_components()...")
        
        # Call the initialization method directly
        await retriever._init_components()
        
        print(f"   After init - _vector_store: {retriever._vector_store}")
        print(f"   After init - _embedder: {retriever._embedder}")
        
        if retriever._vector_store:
            print(f"   Vector store type: {type(retriever._vector_store).__name__}")
            print(f"   Vector store _initialized: {getattr(retriever._vector_store, '_initialized', 'MISSING')}")
            
            if hasattr(retriever._vector_store, 'index'):
                print(f"   Vector store index: {retriever._vector_store.index}")
                if retriever._vector_store.index:
                    print(f"   Index ntotal: {retriever._vector_store.index.ntotal}")
            
            if hasattr(retriever._vector_store, 'chunks'):
                print(f"   Vector store chunks: {len(retriever._vector_store.chunks) if retriever._vector_store.chunks else 0}")
        
        print("\n🧪 3. Testing retrieval with empty store...")
        
        # Test retrieval
        test_query = "What is machine learning?"
        test_config = {"top_k": 3}
        
        try:
            results = await retriever._retrieve_documents(test_query, test_config)
            print(f"   ✅ Retrieval completed successfully")
            print(f"   Results returned: {len(results)}")
            
            for i, doc in enumerate(results):
                print(f"   Result {i+1}: {doc.content[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Retrieval failed: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
        
        print("\n💡 4. Diagnosis:")
        
        # Check if the issue is initialization or empty store
        if not retriever._vector_store:
            print("   🔴 Vector store was not created")
        elif not getattr(retriever._vector_store, '_initialized', False):
            print("   🔴 Vector store was created but not initialized")
        elif not hasattr(retriever._vector_store, 'index') or not retriever._vector_store.index:
            print("   🔴 Vector store initialized but no index created")
        elif retriever._vector_store.index.ntotal == 0:
            print("   🟡 Vector store initialized but empty (no documents ingested)")
            print("   💡 Solution: Use ingestion API to upload documents first")
        else:
            print("   🟢 Vector store appears to be working correctly")
        
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vector_retriever())
