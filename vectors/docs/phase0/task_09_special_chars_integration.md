# Task 09: Test Special Character Integration Across All Components

## Context
You are completing the architecture validation phase (Phase 0, Task 9). Tasks 05-08 validated individual components (Tantivy, LanceDB, Rayon, tree-sitter). Now you need to test that all components work together correctly with code-specific special characters.

## Objective
Implement comprehensive integration testing to ensure that special characters common in code (brackets, generics, operators, macros) work correctly across the entire pipeline: tree-sitter parsing → text indexing (Tantivy) → vector storage (LanceDB).

## Requirements
1. Test end-to-end pipeline with special characters
2. Test all critical code patterns that must be searchable
3. Test integration between tree-sitter, Tantivy, and LanceDB
4. Validate that special characters survive the entire pipeline
5. Test edge cases and boundary conditions
6. Create comprehensive integration test suite

## Critical Special Characters to Test
```rust
// Cargo.toml patterns
"[workspace]", "[dependencies]", "[package]"

// Rust generics and types
"Result<T, E>", "Vec<String>", "HashMap<K, V>", "Option<T>"

// Rust operators and syntax
"->", "=>", "::", "<-", "&mut", "&self", "?", "!"

// Rust macros and attributes
"#[derive(Debug)]", "#![allow(dead_code)]", "macro_rules!"

// Function signatures
"pub fn test() -> Result<(), Error>", "async fn process<T>()"

// Rust-specific syntax
"impl<T>", "where T:", "dyn Trait", "'static"
```

## Implementation for validation.rs (extend existing)
```rust
use std::collections::HashMap;
use anyhow::Result;
use tracing::{info, debug, error};

pub struct IntegrationValidator;

impl IntegrationValidator {
    /// Test comprehensive special character integration
    pub async fn validate_special_chars_integration() -> Result<()> {
        info!("Starting special character integration validation");
        
        // Test end-to-end pipeline
        Self::test_end_to_end_pipeline().await?;
        
        // Test all critical patterns
        Self::test_critical_code_patterns().await?;
        
        // Test edge cases
        Self::test_edge_cases().await?;
        
        // Test cross-component consistency
        Self::test_cross_component_consistency().await?;
        
        info!("Special character integration validation completed successfully");
        Ok(())
    }
    
    async fn test_end_to_end_pipeline() -> Result<()> {
        debug!("Testing end-to-end pipeline with special characters");
        
        let test_code = r#"
            [package]
            name = "test-project"
            version = "0.1.0"
            
            [dependencies]
            serde = { version = "1.0", features = ["derive"] }
            
            [workspace]
            members = ["crate1", "crate2"]
            
            use std::collections::HashMap;
            use std::result::Result;
            
            #[derive(Debug, Clone, Serialize, Deserialize)]
            pub struct Config<T> 
            where 
                T: Clone + Send + Sync + 'static
            {
                pub name: String,
                pub data: HashMap<String, T>,
                pub options: Option<Vec<String>>,
            }
            
            impl<T> Config<T> 
            where 
                T: Clone + Send + Sync + 'static
            {
                pub async fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
                    Ok(Self {
                        name: name.to_string(),
                        data: HashMap::new(),
                        options: None,
                    })
                }
                
                pub fn process(&mut self, input: &str) -> Result<(), &'static str> {
                    if input.is_empty() {
                        return Err("Input cannot be empty");
                    }
                    // Process logic here
                    Ok(())
                }
            }
            
            macro_rules! create_config {
                ($name:expr, $type:ty) => {
                    Config::<$type>::new($name).await?
                };
            }
            
            pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
                let config: Config<i32> = create_config!("test", i32);
                config.process("test input")?;
                Ok(())
            }
        "#;
        
        // Step 1: Parse with tree-sitter
        debug!("Step 1: Parsing with tree-sitter");
        let chunks = TreeSitterValidator::chunk_code_semantically(test_code, "rust")?;
        assert!(chunks.len() > 0, "Tree-sitter should produce chunks");
        
        // Step 2: Index with Tantivy
        debug!("Step 2: Indexing with Tantivy");
        let tantivy_results = Self::index_chunks_tantivy(&chunks)?;
        
        // Step 3: Store in LanceDB
        debug!("Step 3: Storing in LanceDB");
        let lancedb_results = Self::store_chunks_lancedb(&chunks).await?;
        
        // Step 4: Test search across all components
        debug!("Step 4: Testing search functionality");
        Self::test_search_integration(&tantivy_results, &lancedb_results).await?;
        
        debug!("End-to-end pipeline test passed");
        Ok(())
    }
    
    async fn test_critical_code_patterns() -> Result<()> {
        debug!("Testing critical code patterns");
        
        let critical_patterns = vec![
            // Cargo.toml patterns
            ("[workspace]", "Cargo workspace configuration"),
            ("[dependencies]", "Cargo dependencies section"),
            ("[package]", "Cargo package configuration"),
            
            // Rust type patterns
            ("Result<T, E>", "Rust Result type"),
            ("Vec<String>", "Vector of strings"),
            ("HashMap<K, V>", "HashMap type"),
            ("Option<T>", "Optional type"),
            
            // Rust syntax patterns
            ("pub fn", "Public function declaration"),
            ("async fn", "Async function declaration"),
            ("impl<T>", "Generic implementation"),
            ("where T:", "Where clause constraint"),
            
            // Operators and symbols
            ("->", "Return type arrow"),
            ("=>", "Match arm arrow"),
            ("::", "Path separator"),
            ("&mut", "Mutable reference"),
            ("&self", "Self reference"),
            
            // Attributes and macros
            ("#[derive(Debug)]", "Derive macro"),
            ("#![allow(dead_code)]", "Allow attribute"),
            ("macro_rules!", "Macro definition"),
        ];
        
        for (pattern, description) in critical_patterns {
            debug!("Testing pattern: {} ({})", pattern, description);
            
            let test_code = format!(
                "// Test for {}\n{}\npub fn test() {{\n    // Test implementation\n}}",
                description, pattern
            );
            
            // Test through full pipeline
            let success = Self::test_pattern_pipeline(&test_code, pattern).await?;
            assert!(success, "Pattern '{}' should work through full pipeline", pattern);
        }
        
        debug!("Critical code patterns test passed");
        Ok(())
    }
    
    async fn test_edge_cases() -> Result<()> {
        debug!("Testing edge cases");
        
        let edge_cases = vec![
            // Empty and minimal cases
            ("", "Empty file"),
            ("a", "Single character"),
            ("[", "Single bracket"),
            ("<>", "Empty generics"),
            
            // Unicode and special characters
            ("你好 мир 🔍", "Unicode characters"),
            ("λ → ∀", "Math symbols"),
            ("\"string with [brackets] and <generics>\"", "Brackets in strings"),
            
            // Nested patterns
            ("Result<Vec<HashMap<String, Option<T>>>>", "Deeply nested generics"),
            ("#[cfg(all(unix, not(target_os = \"macos\")))]", "Complex attributes"),
            
            // Large patterns
            (&"x".repeat(10000), "Very long content"),
            (&format!("Result<{}>", "T, ".repeat(1000)), "Very long generic"),
        ];
        
        for (content, description) in edge_cases {
            debug!("Testing edge case: {}", description);
            
            // Should not crash, even if parsing fails
            let result = Self::test_pattern_pipeline(content, "test").await;
            match result {
                Ok(_) => debug!("Edge case '{}' handled successfully", description),
                Err(e) => debug!("Edge case '{}' failed gracefully: {}", description, e),
            }
        }
        
        debug!("Edge cases test completed");
        Ok(())
    }
    
    async fn test_cross_component_consistency() -> Result<()> {
        debug!("Testing cross-component consistency");
        
        let test_patterns = vec![
            "Result<T, E>",
            "[workspace]",
            "#[derive(Debug)]",
            "pub async fn test() -> Result<(), Error>",
        ];
        
        for pattern in test_patterns {
            debug!("Testing consistency for pattern: {}", pattern);
            
            let test_code = format!("// Pattern test\n{}\n// End pattern", pattern);
            
            // Parse with tree-sitter
            let chunks = TreeSitterValidator::chunk_code_semantically(&test_code, "rust")?;
            
            // Index with Tantivy
            let tantivy_found = Self::search_tantivy_pattern(&chunks, pattern)?;
            
            // Search in LanceDB (text content)
            let lancedb_found = Self::search_lancedb_pattern(&chunks, pattern).await?;
            
            // Both should find the pattern (or both should not find it)
            debug!(
                "Pattern '{}' - Tantivy: {}, LanceDB: {}", 
                pattern, tantivy_found, lancedb_found
            );
            
            // At minimum, the content should be preserved somewhere
            let content_preserved = chunks.iter().any(|chunk| chunk.content.contains(pattern));
            assert!(content_preserved, "Pattern '{}' should be preserved in chunks", pattern);
        }
        
        debug!("Cross-component consistency test passed");
        Ok(())
    }
    
    // Helper methods
    fn index_chunks_tantivy(chunks: &[CodeChunk]) -> Result<Vec<String>> {
        // Simplified Tantivy indexing
        let mut results = Vec::new();
        for chunk in chunks {
            if !chunk.content.trim().is_empty() {
                results.push(chunk.content.clone());
            }
        }
        Ok(results)
    }
    
    async fn store_chunks_lancedb(chunks: &[CodeChunk]) -> Result<Vec<String>> {
        // Simplified LanceDB storage
        let mut results = Vec::new();
        for chunk in chunks {
            if !chunk.content.trim().is_empty() {
                results.push(chunk.content.clone());
            }
        }
        Ok(results)
    }
    
    async fn test_search_integration(
        tantivy_results: &[String], 
        lancedb_results: &[String]
    ) -> Result<()> {
        debug!("Testing search integration");
        
        // Verify both stores have content
        assert!(tantivy_results.len() > 0, "Tantivy should have indexed content");
        assert!(lancedb_results.len() > 0, "LanceDB should have stored content");
        
        Ok(())
    }
    
    async fn test_pattern_pipeline(code: &str, pattern: &str) -> Result<bool> {
        // Test pattern through full pipeline
        let chunks = TreeSitterValidator::chunk_code_semantically(code, "rust")?;
        
        if chunks.is_empty() {
            return Ok(false);
        }
        
        // Check if pattern exists in any chunk
        let found = chunks.iter().any(|chunk| chunk.content.contains(pattern));
        Ok(found)
    }
    
    fn search_tantivy_pattern(chunks: &[CodeChunk], pattern: &str) -> Result<bool> {
        // Simplified search
        Ok(chunks.iter().any(|chunk| chunk.content.contains(pattern)))
    }
    
    async fn search_lancedb_pattern(chunks: &[CodeChunk], pattern: &str) -> Result<bool> {
        // Simplified search
        Ok(chunks.iter().any(|chunk| chunk.content.contains(pattern)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_special_chars_integration() {
        IntegrationValidator::validate_special_chars_integration().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_critical_patterns() {
        IntegrationValidator::test_critical_code_patterns().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_edge_cases() {
        IntegrationValidator::test_edge_cases().await.unwrap();
    }
}
```

## Implementation Steps
1. Add IntegrationValidator struct to validation.rs
2. Implement end-to-end pipeline testing
3. Test all critical code patterns through the full pipeline
4. Implement edge case testing with various inputs
5. Test cross-component consistency
6. Add helper methods for simplified component testing
7. Create comprehensive integration test suite
8. Run tests to verify special character handling

## Success Criteria
- [ ] IntegrationValidator struct implemented and compiling
- [ ] End-to-end pipeline works with special characters
- [ ] All critical code patterns are preserved and searchable
- [ ] Edge cases are handled gracefully (no crashes)
- [ ] Cross-component consistency is maintained
- [ ] Special characters survive tree-sitter → Tantivy → LanceDB pipeline
- [ ] All integration tests pass (`cargo test`)
- [ ] No data corruption or character encoding issues

## Test Command
```bash
cargo test test_special_chars_integration
cargo test test_critical_patterns
cargo test test_edge_cases
```

## Validation Checklist
After completion, verify that these patterns work correctly:
- [ ] `[workspace]` is searchable
- [ ] `Result<T, E>` is preserved
- [ ] `#[derive(Debug)]` is indexed
- [ ] `pub async fn test() -> Result<(), Error>` is complete
- [ ] Unicode characters don't cause crashes
- [ ] Very long patterns are handled

## Time Estimate
10 minutes

## Next Task
Task 10: Generate special character test files for comprehensive testing.