#!/usr/bin/env python3
"""
Script to update import statements for the new controlsgenai/src/rag structure
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
        
        # Update import patterns
        patterns = [
            # Update controlsgenai.funcs.rag.src to src.rag.src
            (r'from controlsgenai\.funcs\.rag\.src\.', r'from src.rag.src.'),
            (r'import controlsgenai\.funcs\.rag\.src\.', r'import src.rag.src.'),
            
            # Update any remaining controlsgenai.funcs references
            (r'from controlsgenai\.funcs\.', r'from src.'),
            (r'import controlsgenai\.funcs\.', r'import src.'),
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
