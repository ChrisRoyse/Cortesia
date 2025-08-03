#!/usr/bin/env python3
"""
Detailed debugging of documentation detection
"""

from ultra_reliable_core import UniversalDocumentationDetector

def debug_detection_detailed():
    """Debug the detection process step by step"""
    
    # Simple test case
    content = '''def calculate_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector1: First vector as list of floats
        vector2: Second vector as list of floats
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    import math
    return 0.0'''
    
    detector = UniversalDocumentationDetector()
    
    print("Testing Python function with docstring:")
    print("=" * 50)
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        print(f"{i:2d}: {line}")
    
    print("\nDetection analysis:")
    print("-" * 30)
    
    # Test the full detection
    result = detector.detect_documentation_multi_pass(content, 'python', 0)
    
    print(f"Final result:")
    print(f"  Has documentation: {result['has_documentation']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Documentation lines: {result['documentation_lines']}")
    print(f"  Patterns found: {result['patterns_found']}")
    print(f"  Detection methods: {result['detection_methods']}")
    
    # Test pass 1 manually
    print(f"\nPass 1 - Pattern Detection:")
    pass1_result = detector._pass1_pattern_detection(lines, 'python', 0)
    print(f"  Has documentation: {pass1_result['has_documentation']}")
    print(f"  Confidence: {pass1_result['confidence']}")
    print(f"  Documentation lines: {pass1_result['documentation_lines']}")
    print(f"  Patterns found: {pass1_result['patterns_found']}")
    
    # Test individual line matching
    print(f"\nIndividual line matching:")
    lang_config = detector.language_patterns.get('python', {})
    doc_patterns = lang_config.get('doc_patterns', [])
    all_patterns = doc_patterns + detector.universal_patterns['line_doc']
    
    print(f"Available patterns: {all_patterns}")
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        matches = []
        for pattern in all_patterns:
            import re
            if re.match(pattern, line, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            print(f"  Line {i}: '{line_stripped}' matches: {matches}")

if __name__ == "__main__":
    debug_detection_detailed()