#!/usr/bin/env python3
"""
Debug script for JavaScript confidence issue in multi-language consistency test
"""

from smart_chunker import smart_chunk_content

js_code = '''/**
 * Calculator class with basic operations
 */
class Calculator {
    constructor() {
        this.value = 0.0;
    }
    
    /**
     * Add a number
     */
    add(n) {
        this.value += n;
    }
}'''

print("DEBUGGING JAVASCRIPT CONFIDENCE ISSUE")
print("=" * 50)
print("JavaScript code:")
print(js_code)
print("\n" + "=" * 50)

chunks = smart_chunk_content(js_code, "javascript", "calc.js")

print(f"Generated {len(chunks)} chunks:")

documented_chunks = [c for c in chunks if c.has_documentation]
total_chunks = len(chunks)
avg_confidence = sum(c.confidence for c in chunks if c.has_documentation) / max(len(documented_chunks), 1)

print(f"Documented chunks: {len(documented_chunks)}")
print(f"Total chunks: {total_chunks}")
print(f"Average confidence: {avg_confidence}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(f"  Type: {chunk.chunk_type}")
    print(f"  Has documentation: {chunk.has_documentation}")
    print(f"  Confidence: {chunk.confidence}")
    print(f"  Size: {chunk.size_chars} chars")
    if chunk.declaration:
        print(f"  Declaration: {chunk.declaration.declaration_type} '{chunk.declaration.name}'")
    print(f"  Content: {chunk.content[:100]}...")
    
    # Debug JSDoc detection
    if 'class' in chunk.content.lower() or 'function' in chunk.content.lower():
        has_jsdoc = '/**' in chunk.content
        print(f"  Contains JSDoc (/**): {has_jsdoc}")
        if has_jsdoc and not chunk.has_documentation:
            print("  >>> JSDoc present but not detected!")
        elif chunk.has_documentation and chunk.confidence < 0.5:
            print(f"  >>> Low confidence ({chunk.confidence}) despite documentation detection")
            
print(f"\nIssue: Average confidence {avg_confidence} < 0.5 expected")