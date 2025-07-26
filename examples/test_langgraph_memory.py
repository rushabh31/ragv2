#!/usr/bin/env python3
"""
Test script for LangGraph memory implementation.

This script demonstrates the new LangGraph checkpoint memory functionality
including SOEID-based user tracking and all the new API endpoints.
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8001"
TEST_SOEIDS = ["user123", "user456", "user789"]
API_KEY = "test-api-key"

class LangGraphMemoryTester:
    """Test class for LangGraph memory functionality."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
        self.session_ids = {}  # Track session IDs for each user
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def send_message(self, soeid: str, query: str, session_id: str = None) -> Dict[str, Any]:
        """Send a chat message."""
        print(f"📤 Sending message for {soeid}: {query}")
        
        response = await self.client.post(
            f"{self.base_url}/chat/message/json",
            headers={
                "soeid": soeid,
                "X-API-Key": API_KEY
            },
            json={
                "query": query,
                "session_id": session_id,
                "use_retrieval": True,
                "use_history": True,
                "metadata": {"test": True}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {result['response'][:100]}...")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_soeid_history(self, soeid: str) -> Dict[str, Any]:
        """Get all history for a SOEID."""
        print(f"📚 Getting history for SOEID: {soeid}")
        
        response = await self.client.get(
            f"{self.base_url}/chat/history/{soeid}",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_sessions']} sessions, {result['total_messages']} messages")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_sessions_for_soeid(self, soeid: str) -> Dict[str, Any]:
        """Get session metadata for a SOEID."""
        print(f"📋 Getting sessions for SOEID: {soeid}")
        
        response = await self.client.get(
            f"{self.base_url}/chat/sessions/{soeid}",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_sessions']} sessions")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get history for a specific session."""
        print(f"📖 Getting history for session: {session_id}")
        
        response = await self.client.get(
            f"{self.base_url}/chat/history/session/{session_id}",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {len(result['messages'])} messages")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_all_threads(self) -> Dict[str, Any]:
        """Get all threads in the system."""
        print(f"🧵 Getting all threads")
        
        response = await self.client.get(
            f"{self.base_url}/chat/threads",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_threads']} threads")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_thread_history(self, thread_id: str, limit: int = None) -> Dict[str, Any]:
        """Get history for a specific thread."""
        print(f"🧵 Getting history for thread: {thread_id}")
        
        url = f"{self.base_url}/chat/threads/{thread_id}/history"
        if limit:
            url += f"?limit={limit}"
        
        response = await self.client.get(
            url,
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {len(result['messages'])} messages")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        print(f"📊 Getting memory statistics")
        
        response = await self.client.get(
            f"{self.base_url}/chat/memory/stats",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Memory: {result['memory_type']} ({result['store_type']})")
            print(f"   Sessions: {result['total_sessions']}, Messages: {result['total_messages']}")
            print(f"   Unique SOEIDs: {result['unique_soeids']}")
            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
    
    async def delete_soeid_history(self, soeid: str) -> bool:
        """Delete all history for a SOEID."""
        print(f"🗑️ Deleting history for SOEID: {soeid}")
        
        response = await self.client.delete(
            f"{self.base_url}/chat/history/{soeid}",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 204:
            print(f"✅ Deleted history for {soeid}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    
    async def run_comprehensive_test(self):
        """Run a comprehensive test of all LangGraph memory functionality."""
        print("🚀 Starting LangGraph Memory Comprehensive Test")
        print("=" * 60)
        
        # Test 1: Send messages for multiple users
        print("\n📝 Test 1: Sending messages for multiple users")
        print("-" * 40)
        
        test_conversations = {
            "user123": [
                "What is machine learning?",
                "Can you explain neural networks?",
                "How does deep learning work?"
            ],
            "user456": [
                "What is artificial intelligence?",
                "Tell me about natural language processing"
            ],
            "user789": [
                "Explain computer vision",
                "What are transformers in AI?",
                "How does attention mechanism work?"
            ]
        }
        
        for soeid, queries in test_conversations.items():
            session_id = None
            for query in queries:
                result = await self.send_message(soeid, query, session_id)
                if result:
                    session_id = result.get("session_id")
                    self.session_ids[soeid] = session_id
                await asyncio.sleep(0.5)  # Small delay between messages
        
        # Test 2: Get memory statistics
        print("\n📊 Test 2: Memory Statistics")
        print("-" * 40)
        stats = await self.get_memory_stats()
        
        # Test 3: Get all threads
        print("\n🧵 Test 3: All Threads")
        print("-" * 40)
        threads = await self.get_all_threads()
        
        # Test 4: Test SOEID-based retrieval
        print("\n👤 Test 4: SOEID-based History Retrieval")
        print("-" * 40)
        
        for soeid in TEST_SOEIDS:
            # Get sessions for SOEID
            sessions = await self.get_sessions_for_soeid(soeid)
            
            # Get full history for SOEID
            history = await self.get_soeid_history(soeid)
            
            # Get session history if we have a session ID
            if soeid in self.session_ids:
                session_history = await self.get_session_history(self.session_ids[soeid])
        
        # Test 5: Thread-based retrieval
        print("\n🧵 Test 5: Thread-based History Retrieval")
        print("-" * 40)
        
        if threads and threads.get("threads"):
            # Test with the first thread
            first_thread = threads["threads"][0]
            thread_id = first_thread["thread_id"]
            
            # Get full thread history
            await self.get_thread_history(thread_id)
            
            # Get limited thread history
            await self.get_thread_history(thread_id, limit=2)
        
        # Test 6: Alternative endpoints
        print("\n🔄 Test 6: Alternative Endpoints")
        print("-" * 40)
        
        for soeid in TEST_SOEIDS[:2]:  # Test with first 2 users
            response = await self.client.get(
                f"{self.base_url}/chat/sessions/{soeid}/history?limit=5",
                headers={"X-API-Key": API_KEY}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Alternative endpoint for {soeid}: {result['total_sessions']} sessions")
        
        # Test 7: Cleanup (optional)
        print("\n🧹 Test 7: Cleanup Test (deleting one user)")
        print("-" * 40)
        
        # Delete history for one test user
        await self.delete_soeid_history("user789")
        
        # Verify deletion
        history_after_delete = await self.get_soeid_history("user789")
        if history_after_delete.get("total_sessions", 0) == 0:
            print("✅ Deletion successful")
        else:
            print("❌ Deletion may have failed")
        
        # Final statistics
        print("\n📊 Final Statistics")
        print("-" * 40)
        final_stats = await self.get_memory_stats()
        
        print("\n🎉 Comprehensive test completed!")
        print("=" * 60)


async def main():
    """Main test function."""
    print("LangGraph Memory Test Script")
    print("=" * 40)
    print(f"Testing against: {BASE_URL}")
    print(f"Test SOEIDs: {', '.join(TEST_SOEIDS)}")
    print()
    
    try:
        async with LangGraphMemoryTester() as tester:
            await tester.run_comprehensive_test()
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if server is running
    print("🔍 Checking if chatbot server is running...")
    try:
        import requests
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print()
            asyncio.run(main())
        else:
            print(f"❌ Server returned status {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start the chatbot server first:")
        print("   cd examples/rag/chatbot")
        print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8001")
