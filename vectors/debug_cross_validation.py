#!/usr/bin/env python3
"""
Debug the cross-validation logic that's reducing confidence
"""

from ultra_reliable_core import UniversalDocumentationDetector

def debug_cross_validation():
    """Debug cross-validation step by step"""
    
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
    lines = content.split('\n')
    
    print("Debugging cross-validation:")
    print("=" * 50)
    
    # Get results before pass 4
    results = {'has_documentation': True, 'confidence': 0.8, 'documentation_lines': [1]}
    
    print(f"Before cross-validation:")
    print(f"  Has documentation: {results['has_documentation']}")
    print(f"  Confidence: {results['confidence']}")
    print(f"  Documentation lines: {results['documentation_lines']}")
    
    # Test false positive detection
    false_positive_penalty = detector._check_false_positives(lines, results['documentation_lines'], 'python')
    print(f"\nFalse positive penalty: {false_positive_penalty}")
    
    # Test quality assessment
    quality_score = detector._assess_documentation_quality(lines, results['documentation_lines'])
    print(f"Quality score: {quality_score}")
    
    # Apply the same logic as pass 4
    confidence_after_fp = results['confidence'] * (1.0 - false_positive_penalty)
    print(f"Confidence after false positive penalty: {confidence_after_fp}")
    
    confidence_after_quality = confidence_after_fp * quality_score
    print(f"Confidence after quality assessment: {confidence_after_quality}")
    
    # Debug false positive patterns
    print(f"\nFalse positive analysis:")
    for i, line_idx in enumerate(results['documentation_lines']):
        if line_idx < len(lines):
            line = lines[line_idx].strip().lower()
            print(f"  Line {line_idx}: '{line}'")
            
            # Check against false positive patterns
            lang_config = detector.language_patterns.get('python', {})
            false_positive_patterns = lang_config.get('false_positive_patterns', [])
            universal_false_positives = [
                r'^\s*[/\*#%;\-\s]*todo\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*fixme\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*hack\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*debug\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*temporary\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*temp\s*:?.*$',
                r'^\s*[/\*#%;\-\s]*$',
                r'^\s*[/\*#%;\-\s]*-+\s*$',
                r'^\s*[/\*#%;\-\s]*=+\s*$',
                r'^\s*[/\*#%;\-\s]*\*+\s*$',
            ]
            all_patterns = false_positive_patterns + universal_false_positives
            
            for pattern in all_patterns:
                import re
                if re.match(pattern, line, re.IGNORECASE):
                    print(f"    MATCHES FALSE POSITIVE: {pattern}")
                    break
    
    # Debug quality assessment
    print(f"\nQuality assessment analysis:")
    total_chars = 0
    meaningful_lines = 0
    
    for line_idx in results['documentation_lines']:
        if line_idx < len(lines):
            line = lines[line_idx].strip()
            # Remove comment prefixes for analysis
            import re
            clean_line = re.sub(r'^[/\*#%;\-\s]*', '', line).strip()
            
            print(f"  Line {line_idx}: '{line}' -> clean: '{clean_line}' (len: {len(clean_line)})")
            
            if len(clean_line) >= 10:  # Meaningful content
                meaningful_lines += 1
                total_chars += len(clean_line)
    
    print(f"  Meaningful lines: {meaningful_lines}")
    print(f"  Total chars: {total_chars}")
    if meaningful_lines > 0:
        avg_length = total_chars / meaningful_lines
        print(f"  Average length: {avg_length}")

if __name__ == "__main__":
    debug_cross_validation()