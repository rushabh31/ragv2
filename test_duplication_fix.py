#!/usr/bin/env python3
"""
Test script to verify that message duplication is fixed.
"""

import requests
import json
import time

def test_duplication_fix():
    """Test that messages are not duplicated in memory."""
    base_url = "http://localhost:8001"
    
    # Test data
    test_soeid = "test-user-duplication"
    test_session = "test-session-duplication"
    test_query = "Hello, this is a test message to check for duplication"
    
    print("🔍 Testing Message Duplication Fix")
    print("=" * 50)
    
    # Step 1: Send a message
    print(f"\n📝 Step 1: Sending test message...")
    print(f"   SOEID: {test_soeid}")
    print(f"   Session: {test_session}")
    print(f"   Query: {test_query}")
    
    chat_response = requests.post(
        f"{base_url}/chat/message/json",
        headers={
            "Content-Type": "application/json",
            "soeid": test_soeid
        },
        json={
            "query": test_query,
            "session_id": test_session,
            "use_retrieval": False  # Disable retrieval for faster testing
        }
    )
    
    if chat_response.status_code != 200:
        print(f"❌ Failed to send message: {chat_response.status_code}")
        print(f"Response: {chat_response.text}")
        return False
    
    chat_result = chat_response.json()
    print(f"✅ Message sent successfully")
    print(f"   Response: {chat_result.get('response', 'No response')[:100]}...")
    
    # Step 2: Wait a moment for processing
    print(f"\n📝 Step 2: Waiting for processing...")
    time.sleep(2)
    
    # Step 3: Get history for the SOEID
    print(f"\n📝 Step 3: Retrieving history for SOEID {test_soeid}...")
    
    history_response = requests.get(
        f"{base_url}/chat/history/{test_soeid}",
        headers={"Content-Type": "application/json"}
    )
    
    if history_response.status_code != 200:
        print(f"❌ Failed to get history: {history_response.status_code}")
        print(f"Response: {history_response.text}")
        return False
    
    history_result = history_response.json()
    print(f"✅ History retrieved successfully")
    
    # Step 4: Analyze the results
    print(f"\n📝 Step 4: Analyzing results...")
    
    sessions = history_result.get("sessions", [])
    if not sessions:
        print("❌ No sessions found in history")
        return False
    
    session = sessions[0]  # Should be our test session
    messages = session.get("messages", [])
    
    print(f"📊 Found {len(messages)} messages in session {session.get('session_id')}")
    
    # Count user and assistant messages
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    print(f"   User messages: {len(user_messages)}")
    print(f"   Assistant messages: {len(assistant_messages)}")
    
    # Check for duplicates
    user_contents = [msg.get("content") for msg in user_messages]
    assistant_contents = [msg.get("content") for msg in assistant_messages]
    
    user_duplicates = len(user_contents) != len(set(user_contents))
    assistant_duplicates = len(assistant_contents) != len(set(assistant_contents))
    
    print(f"\n📊 Duplication Analysis:")
    print(f"   User message duplicates: {'YES' if user_duplicates else 'NO'}")
    print(f"   Assistant message duplicates: {'YES' if assistant_duplicates else 'NO'}")
    
    # Print all messages for inspection
    print(f"\n📋 All messages:")
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:50]
        timestamp = msg.get("timestamp", "no timestamp")
        print(f"   {i}. [{role}] {content}... ({timestamp})")
    
    # Determine success
    if user_duplicates or assistant_duplicates:
        print(f"\n❌ DUPLICATION DETECTED!")
        print(f"   Expected: 1 user message, 1 assistant message")
        print(f"   Found: {len(user_messages)} user messages, {len(assistant_messages)} assistant messages")
        return False
    else:
        print(f"\n✅ NO DUPLICATION FOUND!")
        print(f"   Messages are properly stored without duplication")
        return True

if __name__ == "__main__":
    success = test_duplication_fix()
    if success:
        print(f"\n🎉 Test PASSED - Duplication issue is fixed!")
    else:
        print(f"\n💥 Test FAILED - Duplication issue still exists!")
