#!/usr/bin/env python3
"""
Script to update all imports from controlgenai to controlsgenai (COMPLETED)
"""
import os
import re
import glob

def update_imports_in_file(file_path):
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace all instances of controlgenai with controlsgenai in imports
        original_content = content
        content = re.sub(r'from controlgenai\.', r'from controlsgenai.', content)
        content = re.sub(r'import controlgenai\.', r'import controlsgenai.', content)
        content = re.sub(r'controlgenai\.funcs\.rag', r'controlsgenai.funcs.rag', content)
        
        # Also update any module references
        content = re.sub(r'controlsgenai-ingestion', r'controlsgenai-ingestion', content)
        content = re.sub(r'controlsgenai-chatbot', r'controlsgenai-chatbot', content)
        
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
    """Main function to update all files"""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip venv directory
        if 'venv' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Also find markdown and other documentation files
    doc_files = []
    for pattern in ['*.md', '*.rst', '*.txt']:
        doc_files.extend(glob.glob(os.path.join(project_root, pattern)))
        doc_files.extend(glob.glob(os.path.join(project_root, '**', pattern), recursive=True))
    
    all_files = python_files + doc_files
    updated_count = 0
    
    print(f"Found {len(all_files)} files to check...")
    
    for file_path in all_files:
        if update_imports_in_file(file_path):
            updated_count += 1
    
    print(f"\nCompleted! Updated {updated_count} files.")

if __name__ == "__main__":
    main()
