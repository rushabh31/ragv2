#!/usr/bin/env python3
"""
Script to update import statements in the API code to point to core RAG components
"""

import os
import re
import sys

def update_imports_in_file(file_path):
    """Update import statements in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update import patterns for API files
        patterns = [
            # Update src.rag.chatbot to src.rag (for core components)
            (r'from src\.rag\.chatbot\.(?!api)', r'from src.rag.'),
            (r'import src\.rag\.chatbot\.(?!api)', r'import src.rag.'),
            
            # Update src.rag.ingestion to src.rag (for core components)  
            (r'from src\.rag\.ingestion\.(?!api)', r'from src.rag.'),
            (r'import src\.rag\.ingestion\.(?!api)', r'import src.rag.'),
            
            # Update references to API modules within the same API
            (r'from src\.rag\.chatbot\.api\.', r'from examples.rag.chatbot.api.'),
            (r'import src\.rag\.chatbot\.api\.', r'import examples.rag.chatbot.api.'),
            (r'from src\.rag\.ingestion\.api\.', r'from examples.rag.ingestion.api.'),
            (r'import src\.rag\.ingestion\.api\.', r'import examples.rag.ingestion.api.'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all Python files in the API directories."""
    base_dir = "/Users/rushabhsmacbook/Documents/controlsgenai"
    
    # Directories to search (only API directories)
    search_dirs = [
        os.path.join(base_dir, "examples", "rag", "chatbot", "api"),
        os.path.join(base_dir, "examples", "rag", "ingestion", "api"),
    ]
    
    updated_files = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if update_imports_in_file(file_path):
                        updated_files.append(file_path)
    
    print(f"\nUpdated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")
    
    print("\nAPI import update complete!")

if __name__ == "__main__":
    main()
