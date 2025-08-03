#!/usr/bin/env python3
"""
Debug the full chunking process to understand why AppConfig is missing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker

def debug_full_chunking():
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
    
    print("=== Full Chunking Process Debug ===")
    
    lines = rust_code.split('\n')
    declarations = chunker.find_declarations(lines, 'rust')
    
    print(f"Found {len(declarations)} declarations:")
    for i, decl in enumerate(declarations):
        print(f"  {i+1}. {decl.declaration_type} '{decl.name}' at line {decl.line_number}")
    
    print(f"\n=== Processing each declaration ===")
    
    chunks = []
    processed_lines = set()
    
    for declaration in declarations:
        if declaration.line_number in processed_lines:
            print(f"Skipping {declaration.name} - line {declaration.line_number} already processed")
            continue
            
        print(f"\nProcessing {declaration.name}:")
        chunk = chunker.create_declaration_chunk(lines, declaration, 'rust')
        
        if chunk and chunker.validate_chunk_quality(chunk):
            print(f"  -> Valid chunk created, size {chunk.size_chars}")
            chunks.append(chunk)
            
            # Mark lines as processed
            for line_num in range(chunk.line_range[0], chunk.line_range[1] + 1):
                processed_lines.add(line_num)
                
        else:
            print(f"  -> No valid chunk created")
    
    print(f"\nFinal chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk.declaration.name if chunk.declaration else 'N/A'} ({chunk.line_range})")

if __name__ == "__main__":
    debug_full_chunking()