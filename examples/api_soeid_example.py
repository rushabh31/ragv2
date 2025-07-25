#!/usr/bin/env python3
"""
Example script demonstrating how to use the new SOEID-based API endpoint.

This script shows:
1. How to send chat messages with SOEID
2. How to retrieve user history by SOEID
3. How to use the API response format
"""

import requests
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8001"
SOEID_HEADER = "SOEID_ABC123"

def send_chat_message(query: str, session_id: str = None, soeid: str = SOEID_HEADER) -> Dict[str, Any]:
    """Send a chat message via the API.
    
    Args:
        query: The user's query
        session_id: Optional session ID
        soeid: User's SOEID
        
    Returns:
        API response
    """
    url = f"{API_BASE_URL}/chat/message"
    
    data = {
        "query": query,
        "use_retrieval": True,
        "use_history": True
    }
    
    if session_id:
        data["session_id"] = session_id
    
    headers = {
        "SOEID": soeid,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending chat message: {e}")
        return {"error": str(e)}

def get_user_history(soeid: str = SOEID_HEADER, limit: int = None) -> Dict[str, Any]:
    """Get user history by SOEID.
    
    Args:
        soeid: User's SOEID
        limit: Maximum number of messages to retrieve
        
    Returns:
        User history response
    """
    url = f"{API_BASE_URL}/chat/user/history"
    
    headers = {
        "SOEID": soeid
    }
    
    params = {}
    if limit:
        params["limit"] = limit
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting user history: {e}")
        return {"error": str(e)}

def demo_api_usage():
    """Demonstrate API usage with SOEID."""
    logger.info("=== SOEID API Usage Demo ===")
    
    # Test 1: Send some chat messages
    logger.info("--- Test 1: Sending Chat Messages ---")
    
    messages = [
        ("What is machine learning?", "session_1"),
        ("Can you explain neural networks?", "session_1"),
        ("What is Python?", "session_2"),
        ("How do I install packages?", "session_2"),
        ("What is Docker?", "session_3"),
        ("How do containers work?", "session_3")
    ]
    
    for query, session_id in messages:
        logger.info(f"Sending message: {query[:50]}... (Session: {session_id})")
        
        result = send_chat_message(query, session_id)
        
        if "error" in result:
            logger.warning(f"Failed to send message: {result['error']}")
        else:
            logger.info(f"✅ Message sent successfully")
            logger.info(f"   Session ID: {result.get('session_id')}")
            logger.info(f"   Response: {result.get('response', '')[:50]}...")
    
    # Test 2: Get user history
    logger.info("\n--- Test 2: Getting User History ---")
    
    history = get_user_history(SOEID_HEADER, limit=10)
    
    if "error" in history:
        logger.warning(f"Failed to get user history: {history['error']}")
    else:
        logger.info("✅ User history retrieved successfully")
        logger.info(f"   SOEID: {history.get('soeid')}")
        logger.info(f"   Total messages: {history.get('total_messages')}")
        logger.info(f"   Total sessions: {history.get('total_sessions')}")
        
        # Display sample messages
        messages = history.get('messages', [])
        if messages:
            logger.info("   Sample messages:")
            for i, msg in enumerate(messages[:3]):
                logger.info(f"     {i+1}. Session: {msg.get('session_id')}, "
                          f"Role: {msg.get('role')}, "
                          f"Content: {msg.get('content', '')[:50]}...")
    
    # Test 3: Get user history with different SOEID
    logger.info("\n--- Test 3: Different SOEID ---")
    
    different_soeid = "SOEID_DEF456"
    history2 = get_user_history(different_soeid, limit=5)
    
    if "error" in history2:
        logger.warning(f"Failed to get user history: {history2['error']}")
    else:
        logger.info(f"✅ User history for {different_soeid} retrieved")
        logger.info(f"   Total messages: {history2.get('total_messages')}")
        logger.info(f"   Total sessions: {history2.get('total_sessions')}")
    
    # Test 4: Get limited history
    logger.info("\n--- Test 4: Limited History ---")
    
    limited_history = get_user_history(SOEID_HEADER, limit=3)
    
    if "error" in limited_history:
        logger.warning(f"Failed to get limited history: {limited_history['error']}")
    else:
        logger.info("✅ Limited history retrieved")
        logger.info(f"   Total messages: {limited_history.get('total_messages')}")
        logger.info(f"   Messages returned: {len(limited_history.get('messages', []))}")

def demo_response_format():
    """Demonstrate the API response format."""
    logger.info("\n=== API Response Format Demo ===")
    
    # Example response structure
    example_response = {
        "soeid": "SOEID_ABC123",
        "messages": [
            {
                "session_id": "session_1",
                "role": "user",
                "content": "What is machine learning?",
                "timestamp": "2025-07-23T20:30:00.000000",
                "metadata": {
                    "soeid": "SOEID_ABC123",
                    "user_id": "user_123"
                }
            },
            {
                "session_id": "session_1",
                "role": "assistant",
                "content": "Machine learning is a subset of AI...",
                "timestamp": "2025-07-23T20:30:01.000000",
                "metadata": {
                    "soeid": "SOEID_ABC123",
                    "user_id": "user_123"
                }
            }
        ],
        "total_messages": 2,
        "total_sessions": 1,
        "metadata": {
            "session_ids": ["session_1"],
            "retrieved_at": "2025-07-23T20:30:05.000000"
        }
    }
    
    logger.info("API Response Format:")
    logger.info(json.dumps(example_response, indent=2))
    
    logger.info("\nKey Features:")
    logger.info("✅ Each message includes session_id")
    logger.info("✅ Messages are from all sessions for the user")
    logger.info("✅ Includes total message and session counts")
    logger.info("✅ Includes metadata with session IDs and retrieval timestamp")
    logger.info("✅ Supports pagination with limit parameter")

def main():
    """Run the API demo."""
    logger.info("Starting SOEID API Demo")
    
    # Demo 1: API usage
    demo_api_usage()
    
    # Demo 2: Response format
    demo_response_format()
    
    logger.info("\n🎉 SOEID API Demo completed!")

if __name__ == "__main__":
    main() 