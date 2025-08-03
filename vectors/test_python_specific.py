#!/usr/bin/env python3
"""
Test Python-specific chunking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import smart_chunk_content

def test_python_chunking():
    python_code = '''"""
Module-level documentation
"""

class SimpleClass:
    """Simple class with documentation."""
    
    def __init__(self):
        self.value = 0
    
    def method_one(self):
        """First method."""
        return self.value
    
    def method_two(self):
        """Second method."""
        return self.value * 2

def standalone_function():
    """Standalone function."""
    return "hello"
'''
    
    chunks = smart_chunk_content(python_code, "python", "test.py")
    
    print(f"Generated {len(chunks)} chunks from Python code:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Has documentation: {chunk.has_documentation}")
        print(f"  Size: {chunk.size_chars} chars")
        if chunk.declaration:
            print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
        print(f"  Content preview: {chunk.content[:100]}...")

if __name__ == "__main__":
    test_python_chunking()