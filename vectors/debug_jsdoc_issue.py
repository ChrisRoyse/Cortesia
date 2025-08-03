#!/usr/bin/env python3
"""
Debug script to understand the JavaScript JSDoc detection issue
"""

from smart_chunker import SmartChunker, smart_chunk_content

# Test JavaScript JSDoc case
js_code = '''/**
 * Calculates the distance between two points
 * @param {number} x1 - X coordinate of first point
 * @param {number} y1 - Y coordinate of first point
 * @param {number} x2 - X coordinate of second point
 * @param {number} y2 - Y coordinate of second point
 * @returns {number} The distance
 */
function calculateDistance(x1, y1, x2, y2) {
    return Math.sqrt((x2-x1)**2 + (y2-y1)**2);
}'''

print("DEBUGGING JAVASCRIPT JSDOC ISSUE")
print("=" * 40)
print("Input code:")
print(js_code)
print("\n" + "=" * 40)

chunker = SmartChunker(max_chunk_size=4000, min_chunk_size=200)

# Split into lines to analyze
lines = js_code.split('\n')
print(f"Total lines: {len(lines)}")

# Test declaration finding
declarations = chunker.find_declarations(lines, "javascript")
print(f"\nDeclarations found: {len(declarations)}")
for i, decl in enumerate(declarations):
    print(f"  {i}: {decl.declaration_type} '{decl.name}' at line {decl.line_number}")
    print(f"      Scope: {decl.scope_start} - {decl.scope_end}")

# Test documentation detection
from ultra_reliable_core import UniversalDocumentationDetector
doc_detector = UniversalDocumentationDetector()

if declarations:
    decl = declarations[0]
    print(f"\nTesting documentation detection for function at line {decl.line_number}")
    
    doc_result = doc_detector.detect_documentation_multi_pass(js_code, "javascript", decl.line_number)
    print(f"Documentation detected: {doc_result['has_documentation']}")
    print(f"Confidence: {doc_result['confidence']}")
    print(f"Documentation lines: {doc_result['documentation_lines']}")
    
    # Show the lines that should be documentation
    print("\nAnalyzing lines:")
    for i, line in enumerate(lines):
        prefix = ">>>" if i in doc_result.get('documentation_lines', []) else "   "
        print(f"{prefix} {i:2d}: {line}")

# Test chunking
print(f"\n{'=' * 40}")
print("CHUNKING ANALYSIS")
chunks = smart_chunk_content(js_code, "javascript", "test.js")
print(f"Chunks generated: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(f"  Type: {chunk.chunk_type}")
    print(f"  Has documentation: {chunk.has_documentation}")
    print(f"  Confidence: {chunk.confidence}")
    print(f"  Size: {chunk.size_chars} chars")
    print(f"  Line range: {chunk.line_range}")
    if chunk.declaration:
        print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
    print(f"  Content preview: {chunk.content[:100]}...")
    
    # Check if JSDoc content is in the chunk
    has_jsdoc_start = '/**' in chunk.content
    has_jsdoc_params = '@param' in chunk.content
    has_function = 'function' in chunk.content
    
    print(f"  Contains '/**': {has_jsdoc_start}")
    print(f"  Contains '@param': {has_jsdoc_params}")
    print(f"  Contains 'function': {has_function}")
    
    if has_jsdoc_start or has_jsdoc_params:
        print("  >>> JSDoc content found but not detected as documentation!")