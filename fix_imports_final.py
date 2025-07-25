#!/usr/bin/env python3
"""
Script to update import statements to remove the extra src layer
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
        
        # Update import patterns to remove the extra src layer
        patterns = [
            # Update src.rag.src to src.rag
            (r'from src\.rag\.src\.', r'from src.rag.'),
            (r'import src\.rag\.src\.', r'import src.rag.'),
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
    """Main function to update all Python files."""
    base_dir = "/Users/rushabhsmacbook/Documents/controlsgenai"
    
    # Directories to search
    search_dirs = [
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "controlsgenai"),
        os.path.join(base_dir, "tests"),
        os.path.join(base_dir, "examples"),
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
    
    # Also update specific files in root
    root_files = ['run_ingestion.py', 'run_chatbot.py', 'test_installation.py']
    for file in root_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            if update_imports_in_file(file_path):
                updated_files.append(file_path)
    
    print(f"\nUpdated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")
    
    print("\nImport update complete!")

if __name__ == "__main__":
    main()
