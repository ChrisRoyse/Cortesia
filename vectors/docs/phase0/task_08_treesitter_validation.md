# Task 08: Test Tree-sitter Parsing Functionality

## Context
You are continuing architecture validation (Phase 0, Task 8). Tasks 05-07 validated Tantivy, LanceDB, and Rayon. Now you need to validate that tree-sitter (syntax-aware parsing) works correctly on Windows for semantic code chunking.

## Objective
Implement and test tree-sitter parsing functionality on Windows, focusing on Rust and Python code parsing, AST-based chunking, and language-aware code splitting for the vector search system.

## Requirements
1. Test basic tree-sitter parsing for Rust code
2. Test tree-sitter parsing for Python code
3. Test AST-based code chunking (functions, structs, classes)
4. Test handling of syntax errors and malformed code
5. Validate semantic boundaries for code splitting
6. Test performance of parsing operations

## Implementation for validation.rs (extend existing)
```rust
use tree_sitter::{Parser, Language, Tree, Node, TreeCursor};
use anyhow::Result;
use tracing::{info, debug, warn, error};

extern "C" {
    fn tree_sitter_rust() -> Language;
    fn tree_sitter_python() -> Language;
}

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub node_type: String,
    pub language: String,
}

pub struct TreeSitterValidator;

impl TreeSitterValidator {
    /// Test tree-sitter parsing functionality on Windows
    pub fn validate_treesitter_windows() -> Result<()> {
        info!("Starting tree-sitter parsing validation on Windows");
        
        // Test Rust parsing
        Self::test_rust_parsing()?;
        
        // Test Python parsing
        Self::test_python_parsing()?;
        
        // Test semantic chunking
        Self::test_semantic_chunking()?;
        
        // Test error handling
        Self::test_error_handling()?;
        
        // Test performance
        Self::test_parsing_performance()?;
        
        info!("Tree-sitter parsing validation completed successfully");
        Ok(())
    }
    
    fn test_rust_parsing() -> Result<()> {
        debug!("Testing Rust code parsing");
        
        let mut parser = Parser::new();
        let language = unsafe { tree_sitter_rust() };
        parser.set_language(language)?;
        
        let rust_code = r#"
            use std::collections::HashMap;
            
            #[derive(Debug, Clone)]
            pub struct Config {
                pub name: String,
                pub values: HashMap<String, i32>,
            }
            
            impl Config {
                pub fn new(name: &str) -> Self {
                    Self {
                        name: name.to_string(),
                        values: HashMap::new(),
                    }
                }
                
                pub fn add_value(&mut self, key: String, value: i32) -> Result<(), String> {
                    if key.is_empty() {
                        return Err("Key cannot be empty".to_string());
                    }
                    self.values.insert(key, value);
                    Ok(())
                }
            }
            
            pub fn process_config<T>(config: &Config) -> Result<T, Error> 
            where 
                T: Default + Clone
            {
                // Process configuration
                Ok(T::default())
            }
        "#;
        
        let tree = parser.parse(rust_code, None).unwrap();
        let root_node = tree.root_node();
        
        // Verify parsing success
        assert!(!root_node.has_error(), "Rust code should parse without errors");
        
        // Extract semantic elements
        let chunks = Self::extract_rust_chunks(rust_code, &root_node)?;
        
        // Verify we found expected elements
        let struct_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.node_type == "struct_item")
            .collect();
        let impl_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.node_type == "impl_item")
            .collect();
        let function_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.node_type == "function_item")
            .collect();
        
        assert_eq!(struct_chunks.len(), 1, "Should find 1 struct");
        assert_eq!(impl_chunks.len(), 1, "Should find 1 impl block");
        assert!(function_chunks.len() >= 2, "Should find at least 2 functions");
        
        debug!("Rust parsing test passed - found {} chunks", chunks.len());
        Ok(())
    }
    
    fn test_python_parsing() -> Result<()> {
        debug!("Testing Python code parsing");
        
        let mut parser = Parser::new();
        let language = unsafe { tree_sitter_python() };
        parser.set_language(language)?;
        
        let python_code = r#"
import os
from typing import Dict, List, Optional

class DataProcessor:
    """Process data with various algorithms."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.results: List[str] = []
    
    def process_item(self, item: str) -> Optional[str]:
        """Process a single item."""
        if not item:
            return None
        
        # Apply processing logic
        processed = item.upper().strip()
        self.results.append(processed)
        return processed
    
    @classmethod
    def from_file(cls, filename: str) -> 'DataProcessor':
        """Create processor from config file."""
        with open(filename, 'r') as f:
            config = eval(f.read())  # Note: unsafe, just for testing
        return cls(config)

def process_batch(items: List[str], processor: DataProcessor) -> Dict[str, str]:
    """Process a batch of items."""
    results = {}
    for item in items:
        result = processor.process_item(item)
        if result:
            results[item] = result
    return results
        "#;
        
        let tree = parser.parse(python_code, None).unwrap();
        let root_node = tree.root_node();
        
        // Verify parsing success
        assert!(!root_node.has_error(), "Python code should parse without errors");
        
        // Extract semantic elements
        let chunks = Self::extract_python_chunks(python_code, &root_node)?;
        
        // Verify we found expected elements
        let class_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.node_type == "class_definition")
            .collect();
        let function_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.node_type == "function_definition")
            .collect();
        
        assert_eq!(class_chunks.len(), 1, "Should find 1 class");
        assert!(function_chunks.len() >= 3, "Should find at least 3 functions");
        
        debug!("Python parsing test passed - found {} chunks", chunks.len());
        Ok(())
    }
    
    fn test_semantic_chunking() -> Result<()> {
        debug!("Testing semantic code chunking");
        
        let rust_code = r#"
            pub fn small_function() -> i32 { 42 }
            
            pub fn large_function() -> String {
                let mut result = String::new();
                for i in 0..100 {
                    result.push_str(&format!("Line {}\n", i));
                }
                result
            }
        "#;
        
        let chunks = Self::chunk_code_semantically(rust_code, "rust")?;
        
        // Should create separate chunks for each function
        assert!(chunks.len() >= 2, "Should create chunks for each function");
        
        // Verify chunk boundaries respect function boundaries
        for chunk in &chunks {
            assert!(chunk.content.trim().len() > 0, "Chunks should not be empty");
            debug!("Chunk type: {}, size: {} bytes", chunk.node_type, chunk.content.len());
        }
        
        debug!("Semantic chunking test passed");
        Ok(())
    }
    
    fn test_error_handling() -> Result<()> {
        debug!("Testing error handling with malformed code");
        
        let malformed_rust = r#"
            pub fn incomplete_function( {
                // Missing closing parenthesis and brace
            
            struct MissingBrace {
                field: String
            // Missing closing brace
        "#;
        
        let chunks = Self::chunk_code_semantically(malformed_rust, "rust")?;
        
        // Should still produce some chunks, even with errors
        assert!(chunks.len() > 0, "Should handle malformed code gracefully");
        
        debug!("Error handling test passed - produced {} chunks from malformed code", chunks.len());
        Ok(())
    }
    
    fn test_parsing_performance() -> Result<()> {
        debug!("Testing parsing performance");
        
        // Create larger code sample
        let large_rust_code = format!(
            "{}{}",
            "use std::collections::HashMap;\n".repeat(50),
            (0..100).map(|i| format!(
                "pub fn function_{}() -> i32 {{ {} }}\n", i, i
            )).collect::<String>()
        );
        
        let start = std::time::Instant::now();
        let chunks = Self::chunk_code_semantically(&large_rust_code, "rust")?;
        let duration = start.elapsed();
        
        info!(
            "Parsed {} bytes into {} chunks in {:?} ({:.2} MB/s)",
            large_rust_code.len(),
            chunks.len(),
            duration,
            (large_rust_code.len() as f64) / (1024.0 * 1024.0) / duration.as_secs_f64()
        );
        
        // Performance should be reasonable (at least 1 MB/s)
        let mb_per_sec = (large_rust_code.len() as f64) / (1024.0 * 1024.0) / duration.as_secs_f64();
        if mb_per_sec < 1.0 {
            warn!("Parsing performance is below 1 MB/s: {:.2} MB/s", mb_per_sec);
        }
        
        debug!("Performance test passed");
        Ok(())
    }
    
    fn extract_rust_chunks(source: &str, node: &Node) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let mut cursor = node.walk();
        
        Self::traverse_rust_node(source, &mut cursor, &mut chunks);
        
        Ok(chunks)
    }
    
    fn traverse_rust_node(source: &str, cursor: &mut TreeCursor, chunks: &mut Vec<CodeChunk>) {
        let node = cursor.node();
        let node_type = node.kind();
        
        // Collect interesting node types
        if matches!(node_type, 
            "struct_item" | "impl_item" | "function_item" | "mod_item" | "enum_item"
        ) {
            let content = &source[node.start_byte()..node.end_byte()];
            chunks.push(CodeChunk {
                content: content.to_string(),
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                node_type: node_type.to_string(),
                language: "rust".to_string(),
            });
        }
        
        // Recurse into children
        if cursor.goto_first_child() {
            loop {
                Self::traverse_rust_node(source, cursor, chunks);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    fn extract_python_chunks(source: &str, node: &Node) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let mut cursor = node.walk();
        
        Self::traverse_python_node(source, &mut cursor, &mut chunks);
        
        Ok(chunks)
    }
    
    fn traverse_python_node(source: &str, cursor: &mut TreeCursor, chunks: &mut Vec<CodeChunk>) {
        let node = cursor.node();
        let node_type = node.kind();
        
        // Collect interesting node types
        if matches!(node_type, 
            "class_definition" | "function_definition" | "decorated_definition"
        ) {
            let content = &source[node.start_byte()..node.end_byte()];
            chunks.push(CodeChunk {
                content: content.to_string(),
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                node_type: node_type.to_string(),
                language: "python".to_string(),
            });
        }
        
        // Recurse into children
        if cursor.goto_first_child() {
            loop {
                Self::traverse_python_node(source, cursor, chunks);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
    
    pub fn chunk_code_semantically(source: &str, language: &str) -> Result<Vec<CodeChunk>> {
        let mut parser = Parser::new();
        
        let lang = match language {
            "rust" => unsafe { tree_sitter_rust() },
            "python" => unsafe { tree_sitter_python() },
            _ => return Err(anyhow::anyhow!("Unsupported language: {}", language)),
        };
        
        parser.set_language(lang)?;
        let tree = parser.parse(source, None).unwrap();
        let root_node = tree.root_node();
        
        match language {
            "rust" => Self::extract_rust_chunks(source, &root_node),
            "python" => Self::extract_python_chunks(source, &root_node),
            _ => Err(anyhow::anyhow!("Unsupported language: {}", language)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_treesitter_validation() {
        TreeSitterValidator::validate_treesitter_windows().unwrap();
    }
    
    #[test]
    fn test_rust_chunking() {
        let code = "pub fn test() -> i32 { 42 }";
        let chunks = TreeSitterValidator::chunk_code_semantically(code, "rust").unwrap();
        assert!(chunks.len() > 0);
    }
    
    #[test]
    fn test_python_chunking() {
        let code = "def test(): return 42";
        let chunks = TreeSitterValidator::chunk_code_semantically(code, "python").unwrap();
        assert!(chunks.len() > 0);
    }
}
```

## Implementation Steps
1. Add TreeSitterValidator struct to validation.rs
2. Implement Rust code parsing with tree-sitter-rust
3. Implement Python code parsing with tree-sitter-python
4. Implement semantic chunking for both languages
5. Add error handling for malformed code
6. Add performance testing with larger code samples
7. Create AST traversal functions for extracting semantic elements
8. Run tests to verify parsing accuracy and performance

## Success Criteria
- [ ] TreeSitterValidator struct implemented and compiling
- [ ] Rust code parsing works correctly
- [ ] Python code parsing works correctly
- [ ] Semantic chunking respects function/class boundaries
- [ ] Error handling works with malformed code
- [ ] Performance is reasonable (>1 MB/s parsing speed)
- [ ] AST traversal extracts correct semantic elements
- [ ] All tests pass (`cargo test`)

## Test Command
```bash
cargo test test_treesitter_validation
cargo test test_rust_chunking
cargo test test_python_chunking
```

## Expected Results
- Functions, structs, classes, and modules are correctly identified
- Code is chunked at semantic boundaries (not arbitrary character limits)
- Malformed code is handled gracefully without crashes
- Parsing performance is acceptable for large codebases

## Time Estimate
10 minutes

## Next Task
Task 09: Test comprehensive special character handling across all components (Tantivy, LanceDB, tree-sitter).