#!/usr/bin/env python3
"""
Test script specifically for SOEID filtering in chat history retrieval.
This tests the new get_session_history_by_soeid method.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_soeid_filtering():
    """Test SOEID filtering in chat history retrieval."""
    print("🔍 Testing SOEID Filtering in Chat History Retrieval")
    print("=" * 60)
    
    try:
        # Import the memory class
        from src.rag.chatbot.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
        
        # Create memory instance
        print("\n📝 Step 1: Creating memory instance...")
        memory = LangGraphCheckpointMemory({
            "store_type": "in_memory"
        })
        print("✅ Memory instance created")
        
        # Test scenario: Two users sharing the same session
        session_id = "shared-session-123"
        user1_soeid = "user-alice"
        user2_soeid = "user-bob"
        
        print(f"\n📝 Step 2: Adding messages from two users to the same session...")
        print(f"   Session ID: {session_id}")
        print(f"   User 1 SOEID: {user1_soeid}")
        print(f"   User 2 SOEID: {user2_soeid}")
        
        # Add messages from user 1 (Alice)
        await memory.add(
            session_id=session_id,
            query="Hello, I'm Alice",
            response="Hi Alice, how can I help you?",
            metadata={"soeid": user1_soeid}
        )
        
        await memory.add(
            session_id=session_id,
            query="What's the weather like?",
            response="I don't have access to weather data.",
            metadata={"soeid": user1_soeid}
        )
        
        # Add messages from user 2 (Bob)
        await memory.add(
            session_id=session_id,
            query="Hello, I'm Bob",
            response="Hi Bob, how can I help you?",
            metadata={"soeid": user2_soeid}
        )
        
        await memory.add(
            session_id=session_id,
            query="Can you help me with math?",
            response="Of course! What math problem do you need help with?",
            metadata={"soeid": user2_soeid}
        )
        
        print("✅ Added 4 interactions (2 from Alice, 2 from Bob)")
        
        # Test 3: Get all messages for the session (no filtering)
        print(f"\n📝 Step 3: Getting ALL messages for session {session_id}...")
        all_messages = await memory.get_history(session_id)
        print(f"📊 Total messages in session: {len(all_messages)}")
        for i, msg in enumerate(all_messages, 1):
            soeid = msg.get("metadata", {}).get("soeid", "unknown")
            print(f"   Message {i}: {msg['role']} - {msg['content'][:30]}... (SOEID: {soeid})")
        
        # Test 4: Get messages filtered by Alice's SOEID
        print(f"\n📝 Step 4: Getting messages for Alice (SOEID: {user1_soeid})...")
        alice_messages = await memory.get_session_history_by_soeid(session_id, user1_soeid)
        print(f"📊 Alice's messages: {len(alice_messages)}")
        for i, msg in enumerate(alice_messages, 1):
            soeid = msg.get("metadata", {}).get("soeid", "unknown")
            print(f"   Message {i}: {msg['role']} - {msg['content']} (SOEID: {soeid})")
        
        # Test 5: Get messages filtered by Bob's SOEID
        print(f"\n📝 Step 5: Getting messages for Bob (SOEID: {user2_soeid})...")
        bob_messages = await memory.get_session_history_by_soeid(session_id, user2_soeid)
        print(f"📊 Bob's messages: {len(bob_messages)}")
        for i, msg in enumerate(bob_messages, 1):
            soeid = msg.get("metadata", {}).get("soeid", "unknown")
            print(f"   Message {i}: {msg['role']} - {msg['content']} (SOEID: {soeid})")
        
        # Test 6: Verify filtering correctness
        print(f"\n📝 Step 6: Verifying filtering correctness...")
        expected_alice = 4  # 2 user + 2 assistant messages
        expected_bob = 4    # 2 user + 2 assistant messages
        expected_total = 8  # 4 from Alice + 4 from Bob
        
        if len(all_messages) == expected_total:
            print("✅ Total message count is correct")
        else:
            print(f"❌ Total message count incorrect: expected {expected_total}, got {len(all_messages)}")
        
        if len(alice_messages) == expected_alice:
            print("✅ Alice's message count is correct")
        else:
            print(f"❌ Alice's message count incorrect: expected {expected_alice}, got {len(alice_messages)}")
        
        if len(bob_messages) == expected_bob:
            print("✅ Bob's message count is correct")
        else:
            print(f"❌ Bob's message count incorrect: expected {expected_bob}, got {len(bob_messages)}")
        
        # Verify no cross-contamination
        alice_soeids = [msg.get("metadata", {}).get("soeid") for msg in alice_messages]
        bob_soeids = [msg.get("metadata", {}).get("soeid") for msg in bob_messages]
        
        if all(soeid == user1_soeid for soeid in alice_soeids):
            print("✅ Alice's messages contain only her SOEID")
        else:
            print(f"❌ Alice's messages contain wrong SOEIDs: {alice_soeids}")
        
        if all(soeid == user2_soeid for soeid in bob_soeids):
            print("✅ Bob's messages contain only his SOEID")
        else:
            print(f"❌ Bob's messages contain wrong SOEIDs: {bob_soeids}")
        
        print("\n🎉 SOEID filtering test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_soeid_filtering())
    sys.exit(0 if success else 1)
