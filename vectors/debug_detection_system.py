#!/usr/bin/env python3
"""
Debug the documentation detection system to understand why confidence is low
"""

import os
from smart_chunker_optimized import SmartChunkerOptimized
from ultra_reliable_core import UniversalDocumentationDetector

def debug_detection():
    """Debug specific detection cases"""
    
    # Test case with clear documentation
    python_code_with_docs = '''
def calculate_similarity(vector1, vector2):
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
    
    return dot_product / (magnitude1 * magnitude2)
'''
    
    print("Testing Python function with clear documentation:")
    print("=" * 60)
    
    # Initialize detection systems
    chunker = SmartChunkerOptimized()
    detector = UniversalDocumentationDetector()
    
    # Test chunking
    chunks = chunker._chunk_content_optimized(python_code_with_docs, 'python', 'test.py')
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Type: {chunk.chunk_type}")
        print(f"Has documentation: {chunk.has_documentation}")
        print(f"Confidence: {chunk.confidence}")
        print(f"Documentation lines: {chunk.documentation_lines}")
        
        if chunk.declaration:
            print(f"Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
            
            # Test direct detection
            detection = detector.detect_documentation_multi_pass(
                chunk.content, 'python', chunk.declaration.line_number
            )
            
            print(f"Direct detection:")
            print(f"  Has docs: {detection['has_documentation']}")
            print(f"  Confidence: {detection['confidence']}")
            print(f"  Doc lines: {detection['documentation_lines']}")
            print(f"  Detection methods: {detection['detection_methods']}")
            print(f"  Patterns found: {detection['patterns_found'][:3]}...")  # First 3
        
        print(f"Content preview:")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
        print("-" * 40)

if __name__ == "__main__":
    debug_detection()