#!/usr/bin/env python3
"""
Debug script to understand chunk creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker

def debug_chunk_creation():
    chunker = SmartChunker()
    
    rust_code = '''/// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost:5432/db".to_string(),
            server_host: "0.0.0.0".to_string(),
            server_port: 8001,
        }
    }
}

/// Create a new product with validation
pub fn create_product(name: String, price: f64) -> Result<Product, String> {
    if name.is_empty() {
        return Err("Name cannot be empty".to_string());
    }
    
    Ok(Product {
        id: 0,
        name,
        price,
    })
}
'''
    
    lines = rust_code.split('\n')
    declarations = chunker.find_declarations(lines, 'rust')
    
    print("=== Chunk Creation Debug ===")
    
    for i, decl in enumerate(declarations):
        print(f"\n--- Processing Declaration {i + 1}: {decl.name} ---")
        
        # Create chunk for this declaration
        chunk = chunker.create_declaration_chunk(lines, decl, 'rust')
        
        if chunk:
            print(f"Chunk created:")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Has documentation: {chunk.has_documentation}")
            print(f"  Line range: {chunk.line_range}")
            print(f"  Size: {chunk.size_chars} chars")
            print(f"  Content preview (first 200 chars):")
            print(f"    {repr(chunk.content[:200])}")
            print(f"  Content preview (first 5 lines):")
            for j, line in enumerate(chunk.content.split('\n')[:5]):
                print(f"    {j}: {line}")
        else:
            print("No chunk created")

if __name__ == "__main__":
    debug_chunk_creation()