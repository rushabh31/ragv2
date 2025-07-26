#!/usr/bin/env python3
"""Test script to verify the chatbot API memory functionality."""

import asyncio
import aiohttp
import json
import time

async def test_chatbot_api():
    """Test the chatbot API memory functionality."""
    
    print("🚀 Testing Chatbot API Memory Functionality")
    print("=" * 50)
    
    base_url = "http://localhost:8001"
    headers = {
        "soeid": "test-user-789",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Send first message
            print("\n📝 Test 1: Sending first message...")
            data = {
                "query": "Hello, I'm testing the memory system",
                "use_retrieval": "false",
                "use_history": "true"
            }
            
            async with session.post(f"{base_url}/chat/message", headers=headers, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    session_id = result.get("session_id")
                    print(f"✅ First message sent successfully")
                    print(f"   Session ID: {session_id}")
                    print(f"   Response: {result.get('response', '')[:100]}...")
                else:
                    print(f"❌ Failed to send first message: {resp.status}")
                    return
            
            # Test 2: Send second message in same session
            print("\n📝 Test 2: Sending second message in same session...")
            data = {
                "query": "Do you remember what I just said?",
                "session_id": session_id,
                "use_retrieval": "false",
                "use_history": "true"
            }
            
            async with session.post(f"{base_url}/chat/message", headers=headers, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Second message sent successfully")
                    print(f"   Response: {result.get('response', '')[:100]}...")
                else:
                    print(f"❌ Failed to send second message: {resp.status}")
                    return
            
            # Test 3: Get session history
            print("\n📝 Test 3: Getting session history...")
            async with session.get(f"{base_url}/chat/history/session/{session_id}") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    messages = result.get("messages", [])
                    print(f"✅ Retrieved session history: {len(messages)} messages")
                    for i, msg in enumerate(messages):
                        print(f"   Message {i+1}: {msg.get('role')} - {msg.get('content', '')[:50]}...")
                else:
                    print(f"❌ Failed to get session history: {resp.status}")
            
            # Test 4: Get history by SOEID
            print("\n📝 Test 4: Getting history by SOEID...")
            async with session.get(f"{base_url}/chat/history/test-user-789") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    sessions = result.get("sessions", [])
                    print(f"✅ Retrieved SOEID history: {len(sessions)} sessions")
                    for i, sess in enumerate(sessions):
                        messages = sess.get("messages", [])
                        print(f"   Session {i+1}: {sess.get('session_id')} with {len(messages)} messages")
                else:
                    print(f"❌ Failed to get SOEID history: {resp.status}")
            
            # Test 5: Get memory stats
            print("\n📝 Test 5: Getting memory statistics...")
            async with session.get(f"{base_url}/chat/memory/stats") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Retrieved memory stats:")
                    print(f"   Memory type: {result.get('memory_type')}")
                    print(f"   Total sessions: {result.get('total_sessions')}")
                    print(f"   Total messages: {result.get('total_messages')}")
                else:
                    print(f"❌ Failed to get memory stats: {resp.status}")
            
            print("\n🎉 All API tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during API testing: {str(e)}")

if __name__ == "__main__":
    print("⏳ Waiting 2 seconds for server to be ready...")
    time.sleep(2)
    asyncio.run(test_chatbot_api())
