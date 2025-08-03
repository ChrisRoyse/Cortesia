#!/usr/bin/env python3
"""
Simple test to debug the SmartChunker
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import smart_chunk_content

def simple_test():
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
    
    chunks = smart_chunk_content(rust_code, "rust", "main.rs")
    
    print(f"Generated {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Has documentation: {chunk.has_documentation}")
        if chunk.declaration:
            print(f"  Declaration type: {chunk.declaration.declaration_type}")
            print(f"  Declaration name: '{chunk.declaration.name}'")
        print(f"  Content preview: {chunk.content[:100]}...")

if __name__ == "__main__":
    simple_test()