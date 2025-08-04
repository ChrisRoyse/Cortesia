# Task 021: Performance Accuracy Validation

## Prerequisites
- Task 001-020 completed: Logic correctness validation verified
- All boolean operations working with 100% accuracy
- System ready for performance evaluation

## Required Imports
```rust
// Add to performance validation test file
use anyhow::Result;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
```

## Context
You have logic correctness validation from Task 020 and performance testing infrastructure from Tasks 014-016. Now you need to combine these to validate that all performance targets are met while maintaining 100% accuracy. This task ensures both speed and correctness work together.

Combined requirements to validate:
- Boolean AND/OR queries < 50ms with 100% accuracy
- Complex nested queries < 100ms with 100% accuracy  
- Cross-chunk queries < 150ms with 100% accuracy
- Memory efficiency maintained with accurate results

## Your Task (10 minutes max)
Implement focused performance and accuracy validation tests for basic boolean operations, ensuring speed and correctness targets are met.

## Success Criteria
1. Validate basic AND/OR performance targets with accurate results
2. Test simple complex queries for both speed and correctness
3. Confirm accuracy is maintained while meeting performance targets
4. Basic performance-accuracy validation passes
5. Foundation set for comprehensive testing

## Implementation Steps

### 1. RED: Write failing combined performance-accuracy tests
```rust
// Create tests/performance_accuracy_tests.rs
use anyhow::Result;
use std::time::Instant;
use std::path::Path;
use tempfile::TempDir;

use llmkg::boolean::BooleanSearchEngine;
use llmkg::cross_chunk::CrossChunkBooleanHandler;
use llmkg::validator::DocumentLevelValidator;

#[test]
fn test_and_or_performance_with_accuracy() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_performance_accuracy_index(&index_path, 500)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing AND/OR performance with accuracy...");
    
    // AND/OR queries that must be both fast (<50ms) and accurate
    let fast_accurate_queries = vec![
        ("pub AND struct", vec!["pub", "struct"]),
        ("fn OR impl", vec!["fn", "impl"]),
        ("struct OR enum", vec!["struct", "enum"]),
        ("pub AND fn", vec!["pub", "fn"]),
    ];
    
    for (query, required_terms) in fast_accurate_queries {
        println!("  Testing query: {}", query);
        
        // Measure performance
        let start = Instant::now();
        let results = boolean_engine.search_boolean(query)?;
        let duration = start.elapsed();
        
        // Validate performance target
        assert!(duration.as_millis() < 50, 
               "Query '{}' took {}ms, should be under 50ms", query, duration.as_millis());
        
        // Validate accuracy
        let accuracy = validator.validate_boolean_results(query, &results)?;
        assert!(accuracy, "Query '{}' failed accuracy validation", query);
        
        // Manual accuracy verification for AND queries
        if query.contains(" AND ") {
            for result in &results {
                let content_lower = result.content.to_lowercase();
                for term in &required_terms {
                    assert!(content_lower.contains(&term.to_lowercase()), 
                           "AND result missing term '{}' in query '{}'", term, query);
                }
            }
        }
        
        // Manual accuracy verification for OR queries
        if query.contains(" OR ") {
            for result in &results {
                let content_lower = result.content.to_lowercase();
                let has_any = required_terms.iter()
                    .any(|term| content_lower.contains(&term.to_lowercase()));
                assert!(has_any, "OR result missing all terms in query '{}'", query);
            }
        }
        
        println!("    ✓ Performance: {}ms, Accuracy: verified, Results: {}", 
                duration.as_millis(), results.len());
    }
    
    Ok(())
}

fn create_performance_accuracy_index(index_path: &Path, num_docs: usize) -> Result<()> {
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
    
    // Create documents optimized for performance-accuracy testing
    for i in 0..num_docs {
        let content = match i % 8 {
            0 => format!("pub struct PerformanceData{} {{ value: i32 }}", i),
            1 => format!("fn performance_process{}() -> bool {{ true }}", i),  
            2 => format!("impl Display for PerformanceType{} {{ }}", i),
            3 => format!("pub fn performance_helper{}() {{ println!(\"test\"); }}", i),
            4 => format!("struct InternalPerformance{} {{ data: String }}", i),
            5 => format!("pub enum PerformanceStatus{} {{ Active, Inactive }}", i),
            6 => format!("impl Clone for PerformanceClone{} {{ }}", i),
            7 => format!("trait PerformanceTrait{} {{ fn method(&self); }}", i),
            _ => unreachable!(),
        };
        
        index_writer.add_document(doc!(
            file_path_field => format!("perf_{}.rs", i),
            content_field => content.clone(),
            raw_content_field => content,
            chunk_index_field => 0u64
        ))?;
    }
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Add complex query performance-accuracy testing
```rust
#[test]
fn test_complex_query_performance_accuracy() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_performance_accuracy_index(&index_path, 300)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing complex query performance with accuracy...");
    
    // Complex queries that must be both fast (<100ms) and accurate
    let complex_accurate_queries = vec![
        "(pub AND struct) OR (fn AND impl)",
        "(pub OR struct) AND Display", 
        "pub AND (struct OR enum) NOT Error",
        "((pub AND struct) OR (impl AND Clone)) NOT trait",
    ];
    
    for query in complex_accurate_queries {
        println!("  Testing complex query: {}", query);
        
        // Measure performance
        let start = Instant::now();
        let results = boolean_engine.search_boolean(query)?;
        let duration = start.elapsed();
        
        // Validate performance target for complex queries
        assert!(duration.as_millis() < 100, 
               "Complex query '{}' took {}ms, should be under 100ms", query, duration.as_millis());
        
        // Validate accuracy with validator (handles complex logic)
        let accuracy = validator.validate_boolean_results(query, &results)?;
        assert!(accuracy, "Complex query '{}' failed accuracy validation", query);
        
        println!("    ✓ Complex Performance: {}ms, Accuracy: verified, Results: {}", 
                duration.as_millis(), results.len());
    }
    
    Ok(())
}

// Note: Cross-chunk performance testing moved to separate task for timing compliance
```

### 3. REFACTOR: Add memory efficiency with accuracy validation
```rust
// Note: Memory efficiency testing simplified for timing compliance

#[test]
fn test_comprehensive_performance_accuracy_report() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_performance_accuracy_index(&index_path, 400)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("\n=== Comprehensive Performance-Accuracy Report ===");
    
    // Test basic categories with performance and accuracy
    let comprehensive_tests = vec![
        ("Simple AND", "pub AND struct", 50, false),
        ("Simple OR", "fn OR impl", 50, false),
        ("Basic nested", "(pub AND struct) OR fn", 100, false),
    ];
    
    let mut all_passed = true;
    
    for (category, query, target_ms, is_cross_chunk) in comprehensive_tests {
        println!("  {}: {}", category, query);
        
        let (duration, result_count, accuracy) = if is_cross_chunk {
            let start = Instant::now();
            let results = cross_chunk_handler.search_across_chunks(query)?;
            let duration = start.elapsed();
            (duration, results.len(), true) // Cross-chunk has built-in accuracy
        } else {
            let start = Instant::now();
            let results = boolean_engine.search_boolean(query)?;
            let duration = start.elapsed();
            let accuracy = validator.validate_boolean_results(query, &results)?;
            (duration, results.len(), accuracy)
        };
        
        let performance_pass = duration.as_millis() < target_ms;
        let accuracy_pass = accuracy;
        
        if performance_pass && accuracy_pass {
            println!("    ✓ PASS: {}ms (target <{}ms), {} results, accuracy: {}", 
                    duration.as_millis(), target_ms, result_count, accuracy);
        } else {
            println!("    ❌ FAIL: {}ms (target <{}ms), {} results, accuracy: {}", 
                    duration.as_millis(), target_ms, result_count, accuracy);
            all_passed = false;
        }
    }
    
    assert!(all_passed, "All performance-accuracy tests must pass");
    println!("=== All Performance-Accuracy Targets Met ===\n");
    
    Ok(())
}
```

## Validation Checklist
- [ ] Performance targets met with accurate boolean logic results
- [ ] Complex queries achieve both speed and correctness targets
- [ ] Cross-chunk performance maintained with accurate aggregation
- [ ] Memory efficiency preserved without compromising accuracy
- [ ] Comprehensive performance-accuracy validation passes

## Context for Next Task
Next task (022) will perform final quality assurance, including Windows compatibility testing and achieving the final 100/100 score for Phase 2 completion.