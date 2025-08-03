#!/usr/bin/env python3
"""
Debug script for SmartChunker to understand name extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker
import re

def debug_rust_patterns():
    """Debug Rust pattern matching"""
    
    chunker = SmartChunker()
    
    rust_lines = [
        '#[derive(Debug, Clone)]',
        'pub struct AppConfig {',
        '    pub database_url: String,',
        '    pub server_host: String,',
        '}',
        '',
        'impl Default for AppConfig {',
        '    fn default() -> Self {',
        '        Self {',
        '            database_url: "test".to_string(),',
        '        }',
        '    }',
        '}',
        '',
        'pub fn create_product(name: String) -> Result<Product, String> {',
        '    Ok(Product { name })',
        '}'
    ]
    
    print("=== Debugging Rust Pattern Matching ===")
    
    patterns = chunker.declaration_patterns['rust']['patterns']
    
    for i, line in enumerate(rust_lines):
        print(f"Line {i}: {line}")
        
        for j, pattern in enumerate(patterns):
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                print(f"  Matches pattern {j}: {pattern}")
                print(f"  Groups: {match.groups()}")
                
                declaration_type = chunker._extract_declaration_type(line, 'rust')
                name = chunker._extract_declaration_name(match, declaration_type)
                
                print(f"  Declaration type: {declaration_type}")
                print(f"  Extracted name: {name}")
                print()
                break
        else:
            print("  No match")
        print()

if __name__ == "__main__":
    debug_rust_patterns()