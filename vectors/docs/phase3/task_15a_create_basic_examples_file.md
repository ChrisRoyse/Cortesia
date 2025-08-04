# Task 15a: Create Basic Examples File

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 14f completion**

## Context
The vector search system needs comprehensive documentation examples for developers. This task creates the foundational examples file that demonstrates basic usage patterns, imports, and setup procedures for the vector indexing system.

## Your Task
Create a complete basic examples file structure with imports, setup, and fundamental usage patterns that developers can immediately use as a starting point.

## Required Implementation

```rust
//! # Vector Search Basic Examples
//! 
//! This module demonstrates basic usage patterns for the vector search system.
//! These examples provide a foundation for understanding core functionality.

use vector_search::{
    IndexBuilder, 
    SearchEngine, 
    VectorStore,
    DocumentBuilder,
    SearchQuery,
    SearchResult,
    IndexConfig
};
use std::path::Path;
use tokio;

/// Basic example demonstrating document indexing and simple search
#[tokio::main]
async fn basic_search_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create index configuration
    let config = IndexConfig::new()
        .with_field("title", true)  // searchable
        .with_field("content", true)
        .with_field("metadata", false); // stored only

    // 2. Initialize the search engine
    let mut engine = SearchEngine::new(config).await?;
    
    // 3. Build and add documents
    let doc1 = DocumentBuilder::new()
        .add_text("title", "Introduction to Rust")
        .add_text("content", "Rust is a systems programming language")
        .add_text("metadata", "tutorial,programming")
        .build();
    
    let doc2 = DocumentBuilder::new()
        .add_text("title", "Vector Databases")
        .add_text("content", "Vector databases enable semantic search")
        .add_text("metadata", "database,search")
        .build();
    
    // 4. Index the documents
    engine.add_document(doc1).await?;
    engine.add_document(doc2).await?;
    engine.commit().await?;
    
    // 5. Perform basic search
    let query = SearchQuery::new("programming language");
    let results = engine.search(&query).await?;
    
    // 6. Process results
    for result in results {
        println!("Score: {:.2}, Title: {}", 
                result.score(), 
                result.get_field("title").unwrap_or("N/A"));
    }
    
    Ok(())
}

/// Example showing batch document processing
async fn batch_indexing_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = IndexConfig::default();
    let mut engine = SearchEngine::new(config).await?;
    
    // Batch document creation
    let documents = vec![
        ("doc1", "Machine learning fundamentals"),
        ("doc2", "Deep learning neural networks"),
        ("doc3", "Natural language processing")
    ];
    
    // Batch indexing for performance
    let mut batch = Vec::new();
    for (id, content) in documents {
        let doc = DocumentBuilder::new()
            .add_text("id", id)
            .add_text("content", content)
            .build();
        batch.push(doc);
    }
    
    engine.add_documents_batch(batch).await?;
    engine.commit().await?;
    
    println!("Indexed {} documents in batch", 3);
    Ok(())
}

/// Configuration examples showing different setups
fn configuration_examples() {
    // Minimal configuration
    let minimal = IndexConfig::new();
    
    // Performance-optimized configuration
    let performance = IndexConfig::new()
        .with_memory_budget(512_000_000) // 512MB
        .with_commit_on_add(false)       // Manual commits
        .with_parallel_indexing(true);
    
    // Search-optimized configuration
    let search_optimized = IndexConfig::new()
        .with_fuzzy_search(true)
        .with_proximity_search(true)
        .with_phrase_search(true)
        .with_wildcard_search(true);
        
    println!("Configuration examples created");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_workflow() {
        let result = basic_search_example().await;
        assert!(result.is_ok(), "Basic example should execute successfully");
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let result = batch_indexing_example().await;
        assert!(result.is_ok(), "Batch example should execute successfully");
    }
}
```

## Success Criteria
- [ ] Basic examples file created with complete import structure
- [ ] Demonstrates fundamental indexing and search operations
- [ ] Includes batch processing examples
- [ ] Shows different configuration options
- [ ] Examples compile and run successfully
- [ ] Provides clear foundation for developers

## Validation
Run the examples:
```bash
cargo test test_basic_workflow
cargo test test_batch_processing
```

## Next Task
Task 15b will add performance guidelines and usage best practices to complement these basic examples.