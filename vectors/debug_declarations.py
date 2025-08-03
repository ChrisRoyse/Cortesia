#!/usr/bin/env python3
"""
Debug script to understand declaration finding and scoping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker

def debug_declarations():
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
    
    print("=== Finding Declarations ===")
    declarations = chunker.find_declarations(lines, 'rust')
    
    for i, decl in enumerate(declarations):
        print(f"\nDeclaration {i + 1}:")
        print(f"  Type: {decl.declaration_type}")
        print(f"  Name: {decl.name}")
        print(f"  Line: {decl.line_number}")
        print(f"  Signature: {decl.full_signature}")
        print(f"  Scope: {decl.scope_start} -> {decl.scope_end}")
        print(f"  Actual line content: {lines[decl.line_number] if decl.line_number < len(lines) else 'N/A'}")
        
        # Show scope content
        print(f"  Scope content:")
        for j in range(decl.scope_start, min(decl.scope_end + 1, len(lines))):
            print(f"    {j}: {lines[j]}")

if __name__ == "__main__":
    debug_declarations()