#!/usr/bin/env python3
"""
Debug SmartChunker integration with UniversalDocumentationDetector
"""

from smart_chunker_optimized import SmartChunkerOptimized
from ultra_reliable_core import UniversalDocumentationDetector

def debug_integration():
    """Debug the integration issue"""
    
    # Test the failing Python case
    python_code = '''@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    start_time: float
    end_time: float
    total_chars_processed: int'''
    
    print("Testing Python class with docstring:")
    print("=" * 50)
    
    lines = python_code.split('\n')
    for i, line in enumerate(lines):
        print(f"{i:2d}: {line}")
    
    # Test direct detector
    print(f"\nDirect UniversalDocumentationDetector:")
    detector = UniversalDocumentationDetector()
    direct_result = detector.detect_documentation_multi_pass(python_code, 'python', 1)
    print(f"  Has docs: {direct_result['has_documentation']}")
    print(f"  Confidence: {direct_result['confidence']}")
    print(f"  Doc lines: {direct_result['documentation_lines']}")
    
    # Test SmartChunker integration
    print(f"\nSmartChunker integration:")
    chunker = SmartChunkerOptimized()
    chunks = chunker._chunk_content_optimized(python_code, 'python', 'test.py')
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}:")
        print(f"    Type: {chunk.chunk_type}")
        print(f"    Has documentation: {chunk.has_documentation}")
        print(f"    Confidence: {chunk.confidence}")
        print(f"    Documentation lines: {chunk.documentation_lines}")
        if chunk.declaration:
            print(f"    Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}' at line {chunk.declaration.line_number}")
    
    # Debug the _create_declaration_chunk_optimized method directly
    print(f"\nDebug _create_declaration_chunk_optimized:")
    declarations = chunker._find_declarations_optimized(lines, 'python')
    if declarations:
        decl = declarations[0]
        print(f"  Found declaration: {decl.declaration_type} '{decl.name}' at line {decl.line_number}")
        
        # Test doc detection within the method
        doc_detection = chunker.doc_detector.detect_documentation_multi_pass(
            '\n'.join(lines), 'python', decl.line_number
        )
        print(f"  Doc detection in method:")
        print(f"    Has docs: {doc_detection['has_documentation']}")
        print(f"    Confidence: {doc_detection['confidence']}")
        print(f"    Doc lines: {doc_detection['documentation_lines']}")
        
        # Test the chunk creation process
        chunk = chunker._create_declaration_chunk_optimized(lines, decl, 'python')
        if chunk:
            print(f"  Created chunk:")
            print(f"    Has documentation: {chunk.has_documentation}")
            print(f"    Confidence: {chunk.confidence}")
            print(f"    Documentation lines: {chunk.documentation_lines}")

if __name__ == "__main__":
    debug_integration()