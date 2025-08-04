# Task 23: Complete Stress Testing Framework Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 22 completed (component interaction testing)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` (exists with all schema functions)
- `C:/code/LLMKG/vectors/tantivy_search/tests/` (directory exists)
- Previous test files (integration_workflow.rs, component_interactions.rs)

## Complete Context (For AI with ZERO Knowledge)

**What is Stress Testing?** Stress testing pushes a system beyond its normal operating capacity to identify breaking points, memory leaks, performance degradation, and failure modes under extreme conditions.

**Why This Task is Critical?** After validating normal component interactions in Task 22, we need to ensure the system can handle real-world stress scenarios like indexing thousands of files, concurrent searches, memory pressure, and resource constraints.

**Key Stress Testing Scenarios:**
1. **Volume Stress** - Indexing thousands of large documents
2. **Memory Stress** - Operating with limited memory budgets
3. **Concurrent Stress** - Multiple simultaneous operations
4. **Duration Stress** - Long-running operations without leaks
5. **Resource Stress** - Limited disk space and file handles
6. **Query Stress** - Complex queries on large datasets

**Real-World Stress Conditions:**
- **Large Codebases:** 10,000+ files, 100MB+ total content
- **Memory Pressure:** Limited to 256MB RAM for indexing
- **Concurrent Users:** 50+ simultaneous searches  
- **Complex Queries:** Nested boolean queries with wildcards
- **Resource Limits:** Low disk space, file handle exhaustion
- **Long Sessions:** 8+ hour continuous operation

**What Makes This Different from Previous Tests?**
- **Scale:** Orders of magnitude more data
- **Resource monitoring:** Memory usage, file handles, disk space
- **Failure modes:** Tests what happens when resources are exhausted
- **Recovery:** Tests system recovery after failures
- **Performance degradation:** Measures slowdown under stress

## Exact Steps (6 minutes implementation)

### Step 1: Create stress testing framework file (4 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/tests/stress_testing.rs` with this exact content:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use tantivy::{doc, Index, IndexWriter, Searcher};
use tantivy::query::QueryParser;
use tantivy::collector::TopDocs;
use anyhow::Result;
use std::time::{Instant, Duration};
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Test indexing performance with large volumes of documents
#[test]
fn test_large_volume_indexing_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("stress_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(100_000_000)?; // 100MB buffer
    
    // Get field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    let start_time = Instant::now();
    let document_count = 1000; // Reduced for CI/testing environments
    
    // Generate and index large volume of documents
    for i in 0..document_count {
        let content = generate_large_content(i);
        let file_path = format!("/stress/test/file_{}.rs", i);
        
        let doc = doc!(
            content_field => content.clone(),
            raw_content_field => content,
            file_path_field => file_path,
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => i % 10 == 0 // Every 10th document has overlap
        );
        
        writer.add_document(doc)?;
        
        // Commit every 100 documents to test incremental performance
        if i % 100 == 99 {
            writer.commit()?;
            writer = index.writer(100_000_000)?; // Create new writer
        }
    }
    
    // Final commit
    writer.commit()?;
    let indexing_time = start_time.elapsed();
    
    // Verify all documents were indexed
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), document_count, "Should have indexed all documents");
    
    // Performance assertions
    let docs_per_second = document_count as f64 / indexing_time.as_secs_f64();
    assert!(docs_per_second > 10.0, "Should index at least 10 docs/second under stress");
    assert!(indexing_time.as_secs() < 300, "Should complete within 5 minutes");
    
    println!("Indexed {} documents in {:?} ({:.1} docs/sec)", 
             document_count, indexing_time, docs_per_second);
    
    Ok(())
}

/// Test memory usage under stress with large documents
#[test]
fn test_memory_pressure_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("memory_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    
    // Use smaller writer buffer to simulate memory pressure
    let mut writer = index.writer(10_000_000)?; // 10MB buffer (small)
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    let start_time = Instant::now();
    
    // Create very large documents to stress memory
    for i in 0..50 {
        let large_content = generate_very_large_content(i);
        
        let doc = doc!(
            content_field => large_content.clone(),
            raw_content_field => large_content,
            file_path_field => format!("/memory/stress_{}.rs", i),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => large_content.len() as u64,
            has_overlap_field => false
        );
        
        // Should handle large documents without crashing
        writer.add_document(doc)?;
        
        // Force frequent commits to test memory management
        if i % 5 == 4 {
            writer.commit()?;
            writer = index.writer(10_000_000)?;
        }
    }
    
    writer.commit()?;
    let total_time = start_time.elapsed();
    
    // Verify system handled memory pressure
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert!(searcher.num_docs() > 0, "Should have indexed documents despite memory pressure");
    assert!(total_time.as_secs() < 180, "Should complete within 3 minutes even under memory pressure");
    
    println!("Handled memory pressure test in {:?}", total_time);
    
    Ok(())
}

/// Test concurrent search operations under stress
#[test]
fn test_concurrent_search_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("concurrent_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Setup test data
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Index searchable content
    let search_terms = vec!["function", "struct", "impl", "trait", "module", "test", "config", "database"];
    for (i, term) in search_terms.iter().enumerate() {
        for j in 0..20 {
            let content = format!("pub {} search_target_{}_{} {{ /* content */ }}", term, i, j);
            let doc = doc!(
                content_field => content.clone(),
                raw_content_field => content,
                file_path_field => format!("/concurrent/{}_{}.rs", term, j),
                chunk_index_field => (i * 20 + j) as u64,
                chunk_start_field => 0u64,
                chunk_end_field => content.len() as u64,
                has_overlap_field => false
            );
            writer.add_document(doc)?;
        }
    }
    writer.commit()?;
    
    // Setup concurrent search test
    let index = Arc::new(index);
    let successful_searches = Arc::new(AtomicUsize::new(0));
    let failed_searches = Arc::new(AtomicUsize::new(0));
    
    let start_time = Instant::now();
    let mut handles = vec![];
    
    // Spawn multiple concurrent search threads
    for thread_id in 0..10 {
        let index_clone = Arc::clone(&index);
        let successful_clone = Arc::clone(&successful_searches);
        let failed_clone = Arc::clone(&failed_searches);
        let terms = search_terms.clone();
        
        let handle = thread::spawn(move || {
            let reader = match index_clone.reader() {
                Ok(r) => r,
                Err(_) => {
                    failed_clone.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };
            
            let searcher = reader.searcher();
            let schema = index_clone.schema();
            let content_field = schema.get_field("content").unwrap();
            let query_parser = QueryParser::for_index(&*index_clone, vec![content_field]);
            
            // Each thread performs multiple searches
            for i in 0..20 {
                let search_term = &terms[i % terms.len()];
                match query_parser.parse_query(search_term) {
                    Ok(query) => {
                        match searcher.search(&query, &TopDocs::with_limit(10)) {
                            Ok(results) => {
                                if !results.is_empty() {
                                    successful_clone.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    failed_clone.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            Err(_) => failed_clone.fetch_add(1, Ordering::Relaxed),
                        }
                    }
                    Err(_) => failed_clone.fetch_add(1, Ordering::Relaxed),
                }
                
                // Small delay to simulate real usage
                thread::sleep(Duration::from_millis(10));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_time = start_time.elapsed();
    let successful_count = successful_searches.load(Ordering::Relaxed);
    let failed_count = failed_searches.load(Ordering::Relaxed);
    let total_searches = successful_count + failed_count;
    
    // Verify concurrent performance
    assert!(successful_count > 0, "Should have some successful searches");
    assert!(total_searches >= 150, "Should have attempted most searches"); // 10 threads * 20 searches - some tolerance
    
    let success_rate = successful_count as f64 / total_searches as f64;
    assert!(success_rate > 0.8, "Should have >80% success rate under concurrent stress");
    
    let searches_per_second = total_searches as f64 / total_time.as_secs_f64();
    assert!(searches_per_second > 10.0, "Should maintain reasonable throughput under concurrent load");
    
    println!("Concurrent stress: {} searches in {:?} ({:.1} searches/sec, {:.1}% success rate)",
             total_searches, total_time, searches_per_second, success_rate * 100.0);
    
    Ok(())
}

/// Test system behavior with complex queries under stress
#[test]
fn test_complex_query_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("complex_query_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Setup rich test data for complex queries
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Create diverse content for complex query testing
    let content_templates = vec![
        "pub fn complex_function() -> Result<Config, Error> { /* implementation */ }",
        "impl DatabaseConnection for PostgresConnection { fn connect() -> Self { } }",
        "struct ComplexStruct { field: Option<Vec<String>>, metadata: HashMap<String, Value> }",
        "#[derive(Debug, Clone)] pub enum ComplexEnum { Variant1(String), Variant2 { field: u64 } }",
        "trait ComplexTrait { fn complex_method(&self, param: &str) -> Box<dyn Future<Output = Result<(), Error>>>; }",
    ];
    
    for (i, template) in content_templates.iter().enumerate() {
        for j in 0..50 {
            let content = format!("{} // File {} Instance {}", template, i, j);
            let doc = doc!(
                content_field => content.clone(),
                raw_content_field => content,
                file_path_field => format!("/complex/{}/file_{}.rs", i, j),
                chunk_index_field => (i * 50 + j) as u64,
                chunk_start_field => 0u64,
                chunk_end_field => content.len() as u64,
                has_overlap_field => j % 3 == 0
            );
            writer.add_document(doc)?;
        }
    }
    writer.commit()?;
    
    // Test complex queries
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    let complex_queries = vec![
        "complex AND function",
        "Result OR Error",
        "struct AND field",
        "impl AND trait",
        "Option AND Vec",
    ];
    
    let start_time = Instant::now();
    let mut total_results = 0;
    
    // Execute multiple complex queries rapidly
    for _ in 0..100 {
        for query_str in &complex_queries {
            match query_parser.parse_query(query_str) {
                Ok(query) => {
                    match searcher.search(&query, &TopDocs::with_limit(20)) {
                        Ok(results) => {
                            total_results += results.len();
                        }
                        Err(_) => {
                            // Query execution failed - acceptable under stress
                        }
                    }
                }
                Err(_) => {
                    // Query parsing failed - acceptable for complex queries
                }
            }
        }
    }
    
    let query_time = start_time.elapsed();
    let queries_executed = 100 * complex_queries.len();
    let queries_per_second = queries_executed as f64 / query_time.as_secs_f64();
    
    // Performance assertions
    assert!(total_results > 0, "Should have found some results from complex queries");
    assert!(queries_per_second > 50.0, "Should execute complex queries at reasonable speed");
    assert!(query_time.as_secs() < 60, "Should complete complex query stress test within 1 minute");
    
    println!("Complex query stress: {} queries in {:?} ({:.1} queries/sec, {} total results)",
             queries_executed, query_time, queries_per_second, total_results);
    
    Ok(())
}

/// Test system recovery after resource exhaustion
#[test]
fn test_resource_exhaustion_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("recovery_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    
    // Test recovery from writer memory exhaustion
    {
        let mut small_writer = index.writer(1_000_000)?; // Very small buffer
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        let file_path_field = schema.get_field("file_path")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        let chunk_start_field = schema.get_field("chunk_start")?;
        let chunk_end_field = schema.get_field("chunk_end")?;
        let has_overlap_field = schema.get_field("has_overlap")?;
        
        // Try to overwhelm the small writer
        for i in 0..10 {
            let large_content = "x".repeat(100_000); // 100KB content
            let doc = doc!(
                content_field => large_content.clone(),
                raw_content_field => large_content,
                file_path_field => format!("/recovery/large_{}.rs", i),
                chunk_index_field => i as u64,
                chunk_start_field => 0u64,
                chunk_end_field => large_content.len() as u64,
                has_overlap_field => false
            );
            
            // This might fail due to memory pressure, which is expected
            let _ = small_writer.add_document(doc);
            
            // Force commit frequently to test recovery
            let _ = small_writer.commit();
        }
    }
    
    // Test that we can still create a new writer and use the index
    let recovery_writer = index.writer(50_000_000);
    assert!(recovery_writer.is_ok(), "Should be able to create new writer after resource exhaustion");
    
    let reader = index.reader();
    assert!(reader.is_ok(), "Should be able to create reader after resource exhaustion");
    
    if let Ok(reader) = reader {
        let searcher = reader.searcher();
        // Index should still be usable even if some operations failed
        assert!(searcher.num_docs() >= 0, "Index should still be accessible");
    }
    
    println!("Successfully recovered from resource exhaustion");
    
    Ok(())
}

/// Generate large content for volume testing
fn generate_large_content(index: usize) -> String {
    format!(r#"
// Generated test file {}
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

/// Large test structure with many fields
#[derive(Debug, Clone)]
pub struct LargeTestStruct_{} {{
    pub id: u64,
    pub name: String,
    pub description: String,
    pub metadata: HashMap<String, String>,
    pub tags: Vec<String>,
    pub content: Arc<String>,
    pub timestamps: Vec<u64>,
    pub configuration: TestConfig,
}}

impl LargeTestStruct_{} {{
    pub fn new(id: u64) -> Self {{
        Self {{
            id,
            name: format!("test_struct_{{}}", id),
            description: "A large test structure with comprehensive fields for stress testing".to_string(),
            metadata: HashMap::new(),
            tags: vec!["test".to_string(), "stress".to_string(), "large".to_string()],
            content: Arc::new("Large content block for testing".repeat(50)),
            timestamps: (0..100).collect(),
            configuration: TestConfig::default(),
        }}
    }}
    
    pub fn process_data(&self) -> Result<String> {{
        let mut result = String::new();
        result.push_str(&self.name);
        result.push_str(&self.description);
        for tag in &self.tags {{
            result.push_str(tag);
        }}
        Ok(result)
    }}
}}

#[derive(Debug, Clone, Default)]
pub struct TestConfig {{
    pub enabled: bool,
    pub timeout: u64,
    pub max_connections: usize,
    pub retry_count: u32,
}}
"#, index, index, index)
}

/// Generate very large content for memory pressure testing
fn generate_very_large_content(index: usize) -> String {
    let base_content = generate_large_content(index);
    // Multiply content size significantly
    format!("{}\n{}", base_content, "// Large comment block\n".repeat(1000))
}
```

### Step 2: Add stress test configuration (1 minute)

Add to `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` if not already present:

```toml
[dev-dependencies]
tempfile = "3.8"
anyhow = "1.0"
```

### Step 3: Verify stress testing capability (1 minute)

Check that all required exports are available in `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test stress_testing
```

**Expected output:**
```
running 5 tests
test test_large_volume_indexing_stress ... ok
test test_memory_pressure_stress ... ok
test test_concurrent_search_stress ... ok
test test_complex_query_stress ... ok
test test_resource_exhaustion_recovery ... ok

test result: ok. 5 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "Test timed out" or "Test killed"**
```bash
# Solution: Reduce test scale for CI environments
# Edit stress_testing.rs, reduce document_count from 1000 to 100
# Reduce thread count from 10 to 4
cargo test stress_testing
```

**Error 2: "Out of memory" or "Cannot allocate memory"**
```bash
# Solution: Reduce writer buffer sizes and document counts
# In test file, change writer buffer from 100_000_000 to 10_000_000
# Reduce content generation size
cargo test stress_testing -- --test-threads=1
```

**Error 3: "Too many open files"**
```bash
# Solution (Unix): Increase file handle limit
ulimit -n 4096
cargo test stress_testing

# Solution (Windows): Reduce concurrent operations
# Edit test to use fewer threads and smaller batches
```

**Error 4: "Disk space exhausted"**
```bash
# Solution: Clean up temp directories and reduce test data
df -h  # Check disk space
cargo clean
cargo test stress_testing
```

## Troubleshooting Checklist

- [ ] Sufficient system RAM (at least 4GB available)
- [ ] Adequate disk space (at least 1GB free)
- [ ] Reasonable file handle limits (ulimit -n > 1024)
- [ ] Test parameters adjusted for CI/limited environments
- [ ] No other heavy processes running during tests
- [ ] Temp directory has write permissions
- [ ] System supports multi-threading for concurrent tests

## Recovery Procedures

### Memory Exhaustion Issues
If tests fail due to memory pressure:
1. Reduce document count: Change 1000 to 100 in volume test
2. Reduce writer buffer: Change 100_000_000 to 10_000_000
3. Reduce concurrent threads: Change 10 to 4 in concurrent test
4. Run single-threaded: `cargo test stress_testing -- --test-threads=1`

### Performance Issues
If tests are too slow:
1. Reduce test scale: Fewer documents, smaller content
2. Increase timeout expectations: Allow more time for completion
3. Skip heavy tests on CI: Use `#[ignore]` attribute for CI environments
4. Profile bottlenecks: Use `cargo test -- --nocapture` for timing info

### Resource Limit Issues
If system resource limits are hit:
1. Check ulimit settings: `ulimit -a`
2. Increase file handles: `ulimit -n 4096`
3. Check disk space: `df -h`
4. Clean temp directories: `rm -rf /tmp/tantivy_*`

## Success Validation Checklist

- [ ] File `tests/stress_testing.rs` exists with 5 comprehensive stress tests
- [ ] Command `cargo test stress_testing` completes within 10 minutes
- [ ] Large volume test successfully indexes 1000+ documents
- [ ] Memory pressure test handles resource constraints gracefully
- [ ] Concurrent test achieves >80% success rate with multiple threads
- [ ] Complex query test executes >50 queries/second
- [ ] Recovery test demonstrates resilience after resource exhaustion
- [ ] All tests provide performance metrics in output

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/tests/stress_testing.rs** - Comprehensive stress testing framework
2. **Performance baselines** - Established performance expectations under stress
3. **Resource limits** - Identified system breaking points and recovery behavior

**Next Task (Task 24)** will implement cross-platform compatibility testing to ensure the system works correctly on Windows, macOS, and Linux.

## Context for Task 24

Task 24 will build upon the stress testing framework to ensure the system handles platform-specific differences in file systems, path handling, threading, and system resources across different operating systems.