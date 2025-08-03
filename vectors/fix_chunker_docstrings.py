#!/usr/bin/env python3
"""
Fix SmartChunker to properly handle Python docstrings
"""

import os
import re
from typing import List, Tuple
from smart_chunker_optimized import SmartChunkerOptimized

def debug_python_chunking():
    """Debug Python chunking to understand the issue"""
    
    test_code = '''def calculate_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector1: First vector as list of floats
        vector2: Second vector as list of floats
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    import math
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(a * a for a in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)'''
    
    print("Original code:")
    lines = test_code.split('\n')
    for i, line in enumerate(lines):
        print(f"{i:2d}: {line}")
    
    print("\n" + "="*60)
    
    # Test scope detection for the function
    chunker = SmartChunkerOptimized()
    
    # Find the declaration
    declarations = chunker._find_declarations_optimized(lines, 'python')
    
    print(f"Found declarations: {len(declarations)}")
    for decl in declarations:
        print(f"  {decl.declaration_type} '{decl.name}' at line {decl.line_number}")
        print(f"  Scope: {decl.scope_start} - {decl.scope_end}")
        print(f"  Signature: {decl.full_signature}")
    
    # Test full chunking
    print("\nChunking results:")
    chunks = chunker._chunk_content_optimized(test_code, 'python', 'test.py')
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Line range: {chunk.line_range}")
        print(f"  Has documentation: {chunk.has_documentation}")
        print(f"  Content lines:")
        chunk_lines = chunk.content.split('\n')
        for j, line in enumerate(chunk_lines):
            print(f"    {j:2d}: {line}")

if __name__ == "__main__":
    debug_python_chunking()