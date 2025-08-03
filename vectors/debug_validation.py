#!/usr/bin/env python3
"""
Debug chunk validation to understand why AppConfig chunk is failing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker

def debug_validation():
    chunker = SmartChunker()
    
    rust_code = '''/// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
}'''
    
    lines = rust_code.split('\n')
    declarations = chunker.find_declarations(lines, 'rust')
    
    appconfig_decl = None
    for decl in declarations:
        if decl.name == 'AppConfig':
            appconfig_decl = decl
            break
    
    if not appconfig_decl:
        print("AppConfig declaration not found!")
        return
    
    print("=== AppConfig Chunk Validation Debug ===")
    
    # Create the chunk
    chunk = chunker.create_declaration_chunk(lines, appconfig_decl, 'rust')
    
    if not chunk:
        print("ERROR: No chunk created!")
        return
    
    print(f"Chunk created:")
    print(f"  Content: {repr(chunk.content)}")
    print(f"  Size: {chunk.size_chars}")
    print(f"  Has documentation: {chunk.has_documentation}")
    print(f"  Declaration: {chunk.declaration.name if chunk.declaration else 'None'}")
    print(f"  Line range: {chunk.line_range}")
    
    # Test each validation criterion
    print(f"\n=== Validation Tests ===")
    
    # Size validation
    size_ok = chunk.size_chars >= chunker.min_chunk_size
    print(f"Size >= {chunker.min_chunk_size}: {size_ok} (actual: {chunk.size_chars})")
    
    size_ok2 = chunk.size_chars <= chunker.max_chunk_size * 1.2
    print(f"Size <= {chunker.max_chunk_size * 1.2}: {size_ok2} (actual: {chunk.size_chars})")
    
    # Content validation
    content_ok = bool(chunk.content.strip())
    print(f"Content not empty: {content_ok}")
    
    # Documentation relationship validation
    if chunk.has_documentation:
        lines_in_chunk = chunk.content.split('\n')
        doc_found = False
        
        for line in lines_in_chunk:
            if chunker._is_documentation_line(line, 'rust'):
                doc_found = True
                break
        
        print(f"Documentation found in chunk: {doc_found}")
        if not doc_found:
            print(f"  ERROR: Chunk claims to have docs but no doc lines found")
    
    # Declaration validation
    if chunk.declaration:
        decl_in_content = chunk.declaration.full_signature in chunk.content
        print(f"Declaration signature in content: {decl_in_content}")
        if not decl_in_content:
            print(f"  Expected: {repr(chunk.declaration.full_signature)}")
            print(f"  In content: {repr(chunk.content)}")
    
    # Overall validation
    is_valid = chunker.validate_chunk_quality(chunk)
    print(f"\nOverall validation result: {is_valid}")

if __name__ == "__main__":
    debug_validation()