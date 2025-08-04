# Task 00_6: Integration Test for Foundation

**Estimated Time:** 7-9 minutes  
**Prerequisites:** Task 00_5 (basic search functionality completed)  
**Dependencies:** None (final Phase 0 task)

## Objective
Create comprehensive integration tests that verify the complete vector search pipeline works correctly, from document indexing through search operations, ensuring the foundation is solid for Phase 2 development.

## Context
You are creating the final validation for Phase 0 foundation tasks. These integration tests must verify that all components work together correctly, handle edge cases properly, and provide the functionality that Phase 2 tasks will depend on. This includes testing the neuromorphic workspace integration.

## Task Details

### What You Need to Do
1. **Create integration test module in src/lib.rs:**
   - End-to-end workflow testing
   - Error handling verification
   - Performance baseline measurement
   - Neuromorphic workspace compatibility

2. **Test complete indexing and search pipeline:**
   - Directory processing and file discovery
   - Document chunking and indexing
   - Search operations and result validation
   - Special character handling throughout pipeline

3. **Create realistic test data:**
   - Sample Rust code with special characters
   - Mixed file types (Rust, Python, TOML, Markdown)
   - Edge cases like empty files and large files

### Implementation Details

#### Add integration tests to src/lib.rs
Add this integration test module to the existing src/lib.rs:

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokio_test;
    
    /// Create a realistic test workspace with various file types
    fn create_test_workspace(temp_dir: &TempDir) -> Result<()> {
        let workspace_path = temp_dir.path();
        
        // Create directory structure
        fs::create_dir_all(workspace_path.join("src"))?;
        fs::create_dir_all(workspace_path.join("tests"))?;
        fs::create_dir_all(workspace_path.join("docs"))?;
        
        // Create Cargo.toml with special characters
        let cargo_content = r#"
[workspace]
members = [
    "crates/neuromorphic-core",
    "crates/vector-search",
]

[workspace.dependencies]
anyhow = "1.0"
tokio = { version = "1.0", features = ["full"] }

[profile.release]
opt-level = 3
"#;
        fs::write(workspace_path.join("Cargo.toml"), cargo_content)?;
        
        // Create Rust source with complex syntax
        let rust_content = r#"
//! Vector search integration with neuromorphic systems
use std::collections::HashMap;
use anyhow::{Result, Context};

/// Generic result type for neuromorphic operations
pub type NeuromorphicResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
pub struct CorticalColumn<T> {
    pub id: uuid::Uuid,
    pub state: Vec<Option<T>>,
    pub connections: HashMap<String, f64>,
}

impl<T> CorticalColumn<T> 
where 
    T: Clone + Send + Sync + 'static,
{
    /// Create new cortical column with initial state
    pub fn new(initial_state: Vec<Option<T>>) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            state: initial_state,
            connections: HashMap::new(),
        }
    }
    
    /// Process activation patterns: [0.1, 0.8, 0.3] -> Vec<f64>
    pub fn process_activation(&self, pattern: &[f64]) -> NeuromorphicResult<Vec<f64>> {
        if pattern.is_empty() {
            return Err("Empty activation pattern".into());
        }
        
        // Simulate neural processing with special characters in comments
        // Pattern: [activation] -> {processed} -> <output>
        let processed: Vec<f64> = pattern
            .iter()
            .enumerate()
            .map(|(idx, &val)| {
                val * self.connections.get(&format!("neuron_{}", idx)).unwrap_or(&1.0)
            })
            .collect();
        
        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cortical_column_creation() {
        let column: CorticalColumn<f64> = CorticalColumn::new(vec![Some(0.5), None, Some(1.0)]);
        assert_eq!(column.state.len(), 3);
        assert!(column.connections.is_empty());
    }
    
    #[test]
    fn test_activation_processing() -> Result<()> {
        let column: CorticalColumn<f64> = CorticalColumn::new(vec![Some(0.5)]);
        let result = column.process_activation(&[0.1, 0.8, 0.3])?;
        assert_eq!(result.len(), 3);
        Ok(())
    }
}
"#;
        fs::write(workspace_path.join("src").join("cortical.rs"), rust_content)?;
        
        // Create Python script
        let python_content = r#"
#!/usr/bin/env python3
"""
Neural network bridge for vector search integration.
Handles patterns like: [input] -> {processing} -> <output>
"""

import json
import numpy as np
from typing import List, Dict, Optional, Union

class NeuralBridge:
    """Bridge between vector search and neural processing."""
    
    def __init__(self, config: Dict[str, Union[str, int, float]]):
        self.config = config
        self.patterns = {}
        
    def process_embeddings(self, embeddings: List[List[float]]) -> Dict[str, float]:
        """Process embeddings with special character handling."""
        results = {}
        
        for i, embedding in enumerate(embeddings):
            # Handle various bracket types: [], {}, <>, ()
            key = f"embedding_{i}"
            results[key] = np.mean(embedding) if embedding else 0.0
            
        return results
        
    def encode_query(self, query: str) -> List[float]:
        """Encode query string to vector representation."""
        # Simulate encoding with special characters
        special_chars = ['[', ']', '{', '}', '<', '>', '#', '@', '$']
        encoding = []
        
        for char in query:
            if char in special_chars:
                encoding.append(1.0)
            else:
                encoding.append(0.5)
                
        return encoding[:128]  # Limit to 128 dimensions

if __name__ == "__main__":
    bridge = NeuralBridge({"model": "test", "dim": 128})
    test_query = "fn main() -> Result<T, E> { #[derive(Debug)] }"
    encoding = bridge.encode_query(test_query)
    print(f"Encoded query length: {len(encoding)}")
"#;
        fs::write(workspace_path.join("src").join("neural_bridge.py"), python_content)?;
        
        // Create markdown documentation
        let markdown_content = r#"
# Vector Search Integration Guide

This document describes the integration between vector search and neuromorphic systems.

## Architecture Overview

The system consists of several components:

- **DocumentIndexer**: Handles file indexing with `tantivy`
- **SearchEngine**: Provides search capabilities with `Result<T, E>` types
- **SmartChunker**: Processes files with AST-based chunking

## Configuration

Add to your `Cargo.toml`:

```toml
[dependencies]
vector-search = { path = "crates/vector-search" }
```

## Usage Examples

### Basic Indexing

```rust
use vector_search::{DocumentIndexer, SmartChunker};

let chunker = SmartChunker::new()?;
let mut indexer = DocumentIndexer::new("./index")?;

// Process files with special characters
let chunks = chunker.chunk_file("src/lib.rs")?;
for chunk in chunks {
    indexer.add_document(&chunk.content, &chunk.file_path, 0, &chunk.language)?;
}
```

### Search Operations

```rust
use vector_search::SearchEngine;

let engine = SearchEngine::new("./index")?;
let results = engine.search_text("fn main")?;

for result in results {
    println!("Found in {}: {}", result.file_path, result.score);
}
```

## Special Character Support

The system handles various special characters commonly found in code:

- Brackets: `[]`, `{}`, `<>`, `()`
- Attributes: `#[derive(Debug)]`, `#[cfg(test)]`
- Generics: `Result<T, E>`, `Vec<Option<String>>`
- Workspace: `[workspace]`, `[dependencies]`

## Error Handling

All operations return `Result<T, Error>` types for proper error handling:

```rust
match engine.search_text("query") {
    Ok(results) => println!("Found {} results", results.len()),
    Err(e) => eprintln!("Search failed: {}", e),
}
```
"#;
        fs::write(workspace_path.join("docs").join("integration.md"), markdown_content)?;
        
        // Create test file
        let test_content = r#"
use vector_search::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_workspace_integration() {
        // Test workspace member integration
        let result = std::panic::catch_unwind(|| {
            // This should compile without errors
            let _: Result<()> = Ok(());
        });
        assert!(result.is_ok());
    }
}
"#;
        fs::write(workspace_path.join("tests").join("integration.rs"), test_content)?;
        
        // Create empty file (edge case)
        fs::write(workspace_path.join("empty.txt"), "")?;
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_complete_indexing_and_search_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        create_test_workspace(&temp_dir).unwrap();
        
        // Test the complete pipeline
        let workspace_path = temp_dir.path();
        let index_path = temp_dir.path().join("test_index");
        
        // Index the test workspace
        let total_chunks = index_directory(workspace_path, &index_path).await.unwrap();
        assert!(total_chunks > 0, "Should have indexed some chunks");
        
        // Create search engine
        let search_engine = SearchEngine::new(&index_path).unwrap();
        
        // Test basic search
        let results = search_engine.search_text("neuromorphic").unwrap();
        assert!(!results.is_empty(), "Should find neuromorphic references");
        
        // Test special character search
        let bracket_results = search_engine.search_text("[workspace]").unwrap();
        assert!(!bracket_results.is_empty(), "Should find workspace brackets");
        
        let generic_results = search_engine.search_text("Result<T, E>").unwrap();
        assert!(!generic_results.is_empty(), "Should find generic types");
        
        // Test language filtering
        let rust_results = search_engine.search_by_language("CorticalColumn", "rust").unwrap();
        assert!(!rust_results.is_empty(), "Should find Rust-specific content");
        
        let python_results = search_engine.search_by_language("NeuralBridge", "python").unwrap();
        assert!(!python_results.is_empty(), "Should find Python-specific content");
        
        // Test file filtering
        let cargo_results = search_engine.search_in_files("workspace", &["Cargo"]).unwrap();
        assert!(!cargo_results.is_empty(), "Should find Cargo.toml content");
        
        // Test statistics
        let stats = search_engine.get_stats().unwrap();
        assert!(stats.total_documents > 0, "Should have indexed documents");
        assert!(stats.available_languages.contains(&"rust".to_string()));
        assert!(stats.available_languages.contains(&"python".to_string()));
        assert!(stats.available_languages.contains(&"toml".to_string()));
        assert!(stats.available_languages.contains(&"markdown".to_string()));
        
        println!("✓ Integration test passed:");
        println!("  - Indexed {} chunks", total_chunks);
        println!("  - {} total documents", stats.total_documents);
        println!("  - {} languages: {:?}", stats.unique_languages, stats.available_languages);
    }
    
    #[tokio::test]
    async fn test_error_handling_robustness() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test with non-existent directory
        let result = index_directory("/non/existent/path", temp_dir.path()).await;
        assert!(result.is_err(), "Should handle non-existent directories");
        
        // Test search on empty index
        let empty_index_path = temp_dir.path().join("empty_index");
        fs::create_dir_all(&empty_index_path).unwrap();
        
        let indexer = DocumentIndexer::new(&empty_index_path).unwrap();
        let search_engine = SearchEngine::new(&empty_index_path).unwrap();
        
        let results = search_engine.search_text("anything").unwrap();
        assert!(results.is_empty(), "Empty index should return no results");
        
        let stats = search_engine.get_stats().unwrap();
        assert_eq!(stats.total_documents, 0, "Empty index should have 0 documents");
    }
    
    #[test]
    fn test_special_character_preservation() {
        use crate::chunking::SmartChunker;
        
        let chunker = SmartChunker::new().unwrap();
        
        // Test various special character combinations
        let test_cases = vec![
            ("[workspace]", "Should preserve workspace brackets"),
            ("Result<T, E>", "Should preserve generic brackets"),
            ("#[derive(Debug)]", "Should preserve attribute syntax"),
            ("Vec<Option<String>>", "Should preserve nested generics"),
            ("HashMap<String, Vec<u64>>", "Should preserve complex generics"),
            ("fn main() -> Result<(), Box<dyn Error>>", "Should preserve function signatures"),
        ];
        
        for (input, description) in test_cases {
            let chunks = chunker.chunk_content(input, "test.rs", "rust").unwrap();
            assert!(!chunks.is_empty(), "{}: Should create chunks", description);
            
            let chunk_content = &chunks[0].content;
            assert!(chunk_content.contains(input), "{}: Content should be preserved exactly", description);
        }
    }
    
    #[test]
    fn test_performance_baseline() {
        use std::time::Instant;
        use crate::chunking::SmartChunker;
        
        let chunker = SmartChunker::new().unwrap();
        
        // Generate reasonably sized content
        let large_content = "fn test_function() { println!(\"Hello\"); }\n".repeat(1000);
        
        let start = Instant::now();
        let chunks = chunker.chunk_content(&large_content, "large.rs", "rust").unwrap();
        let chunking_time = start.elapsed();
        
        assert!(!chunks.is_empty(), "Should create chunks from large content");
        assert!(chunking_time.as_millis() < 1000, "Chunking should complete within 1 second");
        
        println!("✓ Performance baseline:");
        println!("  - Chunked {} lines in {:?}", large_content.lines().count(), chunking_time);
        println!("  - Generated {} chunks", chunks.len());
        println!("  - Average chunk size: {} chars", 
                 chunks.iter().map(|c| c.content.len()).sum::<usize>() / chunks.len());
    }
    
    #[tokio::test]
    async fn test_neuromorphic_workspace_compatibility() {
        // Verify that the vector-search crate integrates properly with neuromorphic workspace
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("neuromorphic_test");
        
        // Test workspace dependencies are available
        let _uuid = uuid::Uuid::new_v4(); // Should compile if workspace deps work
        let _result: anyhow::Result<()> = Ok(()); // Should compile if anyhow is available
        
        // Test basic indexing works
        let mut indexer = DocumentIndexer::new(&index_path).unwrap();
        indexer.add_document(
            "// Neuromorphic integration test\nuse neuromorphic_core::CorticalColumn;",
            "integration.rs",
            0,
            "rust"
        ).unwrap();
        indexer.commit().unwrap();
        
        // Test search works
        let search_engine = SearchEngine::new(&index_path).unwrap();
        let results = search_engine.search_text("neuromorphic").unwrap();
        assert!(!results.is_empty(), "Should find neuromorphic references");
        
        println!("✓ Neuromorphic workspace compatibility verified");
    }
}
```

### Add to src/lib.rs public interface
Also add this public function to make the integration easier:

```rust
/// Get version information for the vector search system
pub fn version_info() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Validate that all required components are functional
pub async fn health_check<P: AsRef<Path>>(temp_dir: P) -> Result<HealthStatus> {
    let temp_path = temp_dir.as_ref();
    let test_index = temp_path.join("health_check_index");
    
    // Test chunker
    let chunker = SmartChunker::new()?;
    let test_chunks = chunker.chunk_content("fn test() {}", "test.rs", "rust")?;
    
    // Test indexer
    let mut indexer = DocumentIndexer::new(&test_index)?;
    for chunk in test_chunks {
        indexer.add_document(&chunk.content, &chunk.file_path, 0, &chunk.language)?;
    }
    indexer.commit()?;
    
    // Test search
    let search_engine = SearchEngine::new(&test_index)?;
    let results = search_engine.search_text("test")?;
    
    // Clean up
    std::fs::remove_dir_all(&test_index).ok();
    
    Ok(HealthStatus {
        chunker_functional: true,
        indexer_functional: true,
        search_functional: !results.is_empty(),
        version: version_info().to_string(),
    })
}

#[derive(Debug)]
pub struct HealthStatus {
    pub chunker_functional: bool,
    pub indexer_functional: bool,
    pub search_functional: bool,
    pub version: String,
}
```

### Expected Output Files
- **Modified:** `crates/vector-search/src/lib.rs` (add integration tests and health check)
- **Validation:** `cargo test -p vector-search --test integration_tests` should pass

## Success Criteria
- [ ] Complete pipeline test passes (indexing → search → results)
- [ ] Special character handling verified throughout pipeline
- [ ] Error handling robustness confirmed
- [ ] Performance baseline established (< 1 second for reasonable content)
- [ ] Neuromorphic workspace integration verified
- [ ] Health check function validates all components
- [ ] All edge cases handled gracefully (empty files, missing directories)
- [ ] Multiple file types indexed and searchable

## Common Pitfalls to Avoid
- Don't skip testing special characters in the complete pipeline
- Ensure tests work on Windows paths and file separators
- Test realistic file sizes and content complexity
- Verify error messages are helpful for debugging
- Don't assume file system operations always succeed

## Context for Phase 2
This foundation enables Phase 2 tasks to:
- Import and use `vector_search::*` components confidently
- Build on proven indexing and search capabilities
- Expect proper error handling and special character support
- Rely on performance baselines for optimization work
- Integrate with the existing neuromorphic workspace seamlessly

## Integration Notes
Phase 0 foundation is now complete and provides:
- ✅ Workspace integration with neuromorphic codebase
- ✅ Complete Tantivy-based indexing and search pipeline
- ✅ Special character support for code files
- ✅ Multi-language file processing (Rust, Python, TOML, Markdown)
- ✅ Error handling consistent with neuromorphic patterns
- ✅ Performance baselines and health monitoring
- ✅ Comprehensive test coverage including edge cases
- ✅ Foundation ready for Phase 2's advanced features