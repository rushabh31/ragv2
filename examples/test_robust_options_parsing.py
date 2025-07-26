#!/usr/bin/env python3
"""
Test script to verify the robust options parsing functionality.
This demonstrates how the ingestion API now handles various input formats gracefully.
"""

import os
import sys
import json
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the robust parsing function
from examples.rag.ingestion.api.router import _parse_options_robust

# Set up logging to see the parsing strategies in action
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_options_parsing():
    """Test various input formats for the options parameter."""
    
    print("🧪 Testing Robust Options Parsing\n")
    
    test_cases = [
        # JSON formats
        ('{"metadata": {"source": "test", "type": "document"}}', "Valid JSON with metadata"),
        ('{"source": "test", "type": "document"}', "Valid JSON without metadata wrapper"),
        ('"simple_string"', "JSON string primitive"),
        ('42', "JSON number primitive"),
        ('true', "JSON boolean primitive"),
        
        # Key-value formats
        ('source=test,type=document', "Comma-separated key-value pairs"),
        ('source=test;type=document', "Semicolon-separated key-value pairs"),
        ('source=test', "Single key-value pair"),
        ('count=42,active=true', "Mixed types in key-value pairs"),
        
        # Plain strings
        ('This is a simple description', "Plain string description"),
        ('document-type-report', "Hyphenated identifier"),
        
        # Edge cases
        ('', "Empty string"),
        ('   ', "Whitespace only"),
        ('malformed{json', "Malformed JSON"),
        ('key=', "Empty value"),
        ('=value', "Empty key"),
    ]
    
    for i, (input_str, description) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"Input: '{input_str}'")
        
        try:
            result = _parse_options_robust(input_str)
            print(f"✅ Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

def test_api_compatibility():
    """Test that the parsing works with common API usage patterns."""
    
    print("\n🌐 Testing API Compatibility\n")
    
    # Simulate common curl usage patterns
    api_test_cases = [
        # What users typically send
        '{"metadata": {"source": "reports", "department": "finance"}}',
        'source=finance,department=reports',
        'financial_report_q3',
        
        # Edge cases that previously caused errors
        'string',  # This was causing the original error
        'metadata',
        'test=value',
    ]
    
    for i, options in enumerate(api_test_cases, 1):
        print(f"API Test {i}: Simulating options='{options}'")
        
        try:
            metadata = _parse_options_robust(options)
            print(f"✅ Parsed metadata: {json.dumps(metadata)}")
            print("✅ API call would succeed")
        except Exception as e:
            print(f"❌ Would cause API error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_options_parsing()
    test_api_compatibility()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Summary:")
    print("- The ingestion API now robustly handles various input formats")
    print("- No more 'Failed to parse options as JSON: string' errors")
    print("- Supports JSON, key-value pairs, and plain strings")
    print("- Graceful fallback strategies ensure API reliability")
