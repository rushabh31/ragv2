#!/usr/bin/env python3
"""
Simple Memory Initialization Test

This script tests the memory initialization in the chatbot service.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_memory_initialization():
    """Test memory initialization in chatbot service."""
    print("🧪 Testing Memory Initialization")
    print("=" * 50)
    
    try:
        # Import and create chatbot service
        from examples.rag.chatbot.api.service import ChatbotService
        
        service = ChatbotService()
        print("✅ ChatbotService created")
        
        # Call a method that triggers memory initialization
        print("🔄 Triggering memory initialization...")
        stats = await service.get_memory_stats()
        print("✅ Memory stats retrieved")
        
        # Check if memory is now initialized
        if hasattr(service, '_memory') and service._memory:
            print("✅ Memory system initialized")
            
            # Check the memory type
            memory_type = type(service._memory).__name__
            print(f"🔍 Memory type: {memory_type}")
            
            # Check if it has a checkpointer
            if hasattr(service._memory, '_checkpointer'):
                checkpointer = service._memory._checkpointer
                checkpointer_type = type(checkpointer).__name__
                print(f"🔍 Checkpointer type: {checkpointer_type}")
                
                # Check if it's a context manager (PostgreSQL)
                if hasattr(checkpointer, '__enter__') and hasattr(checkpointer, '__exit__'):
                    print("✅ Checkpointer is a context manager (likely PostgreSQL)")
                    return True
                else:
                    print("⚠️  Checkpointer is not a context manager (likely in-memory)")
                    return False
            else:
                print("⚠️  No checkpointer found")
                return False
        else:
            print("❌ Memory system not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Memory Initialization Test")
    print("=" * 60)
    
    success = await test_memory_initialization()
    
    if success:
        print("\n🎉 Memory initialization successful!")
        print("PostgreSQL context manager detected.")
    else:
        print("\n⚠️  Memory initialization issues detected.")
        print("May need to fix context manager handling.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
