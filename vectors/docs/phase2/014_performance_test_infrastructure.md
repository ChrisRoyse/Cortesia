# Task 014: Performance Test Infrastructure

## Prerequisites
- Task 001-013 completed: Full boolean search system functional
- All core structs available: BooleanSearchEngine, SearchResult, DocumentResult
- Cross-chunk validation working

## Required Imports
```rust
// Add to new test file or test section
use anyhow::Result;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::DocumentLevelValidator;
```

## Context
You have working boolean search functionality and need to establish performance testing infrastructure. This task focuses on creating the basic test setup and infrastructure for measuring performance, laying the foundation for comprehensive performance validation.

The Phase 2 document specifies performance targets:
- Boolean AND/OR queries < 50ms
- Complex nested queries < 100ms  
- Cross-chunk queries < 150ms

## Your Task (10 minutes max)
Create basic performance testing infrastructure with simple benchmark tests to establish baseline measurements.

## Success Criteria
1. Write performance test helper functions for index creation
2. Create basic timing infrastructure for measurements
3. Implement simple AND/OR performance tests
4. Establish baseline performance measurements
5. All basic performance tests execute successfully

## Implementation Steps

### 1. RED: Write failing performance test infrastructure
```rust
// Add to src/boolean.rs tests
use std::time::Instant;
use tempfile::TempDir;

#[test]
fn test_basic_and_query_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    // Create basic performance test index
    create_basic_performance_index(&index_path, 100)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test simple AND query performance
    let start = Instant::now();
    let results = engine.search_boolean("pub AND struct")?;
    let duration = start.elapsed();
    
    // Should complete in under 50ms
    assert!(duration.as_millis() < 50, 
           "AND query took {}ms, should be under 50ms", duration.as_millis());
    assert!(!results.is_empty(), "Should find results");
    
    println!("Basic AND performance: {}ms, {} results", 
             duration.as_millis(), results.len());
    
    Ok(())
}

#[test]
fn test_basic_or_query_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_basic_performance_index(&index_path, 100)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test simple OR query performance
    let start = Instant::now();
    let results = engine.search_boolean("struct OR fn")?;
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 50, 
           "OR query took {}ms, should be under 50ms", duration.as_millis());
    
    println!("Basic OR performance: {}ms, {} results", 
             duration.as_millis(), results.len());
    
    Ok(())
}
```

### 2. GREEN: Add performance test helper functions
```rust
// Add performance test utilities
fn create_basic_performance_index(index_path: &Path, num_docs: usize) -> Result<()> {
    use tantivy::schema::{Schema, TEXT, STORED};
    use tantivy::{Index, doc};
    
    let mut schema_builder = Schema::builder();
    let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
    let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
    let schema = schema_builder.build();
    
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    // Create diverse but simple test documents
    for i in 0..num_docs {
        let content = match i % 4 {
            0 => format!("pub struct Data{} {{ value: i32 }}", i),
            1 => format!("fn process{}() {{ println!(\"test\"); }}", i),
            2 => format!("impl Display for Type{} {{ }}", i),
            3 => format!("pub fn helper{}() -> bool {{ true }}", i),
            _ => unreachable!(),
        };
        
        index_writer.add_document(doc!(
            file_path_field => format!("file_{}.rs", i),
            content_field => content.clone(),
            raw_content_field => content,
            chunk_index_field => 0u64
        ))?;
    }
    
    index_writer.commit()?;
    Ok(())
}

// Performance measurement utility
fn measure_query_performance<F>(operation: F, operation_name: &str) -> Result<(std::time::Duration, usize)>
where
    F: FnOnce() -> Result<Vec<SearchResult>>,
{
    let start = Instant::now();
    let results = operation()?;
    let duration = start.elapsed();
    
    println!("{}: {}ms, {} results", operation_name, duration.as_millis(), results.len());
    
    Ok((duration, results.len()))
}
```

### 3. REFACTOR: Add basic performance reporting
```rust
#[test]
fn test_performance_baseline_report() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_basic_performance_index(&index_path, 100)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    println!("\n=== Basic Performance Baseline ===");
    
    // Test basic query types
    let basic_queries = vec![
        ("Simple AND", "pub AND struct"),
        ("Simple OR", "fn OR impl"),  
        ("Simple NOT", "pub NOT Error"),
    ];
    
    let mut all_passed = true;
    
    for (name, query) in basic_queries {
        let (duration, result_count) = measure_query_performance(
            || engine.search_boolean(query),
            name
        )?;
        
        // Basic performance check (50ms for simple queries)
        if duration.as_millis() >= 50 {
            println!("⚠️ {} exceeded 50ms target", name);
            all_passed = false;
        }
    }
    
    assert!(all_passed, "All basic queries should meet performance targets");
    println!("=== Basic Performance Tests Passed ===\n");
    
    Ok(())
}
```

## Validation Checklist
- [ ] Performance test infrastructure is created
- [ ] Basic timing measurements work correctly
- [ ] Simple AND/OR queries execute and are measured
- [ ] Performance helper functions are implemented
- [ ] Baseline performance measurements are established

## Context for Next Task
Next task (015) will use this infrastructure to test cross-chunk performance with realistic data scales and more complex scenarios.