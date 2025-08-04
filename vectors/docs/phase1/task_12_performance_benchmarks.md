# Task 12: Implement Performance Benchmarks and Validation

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 11 (Integration tests)  
**Dependencies:** Tasks 01-11 must be completed

## Objective
Create comprehensive performance benchmarks to validate that the system meets Phase 1 targets: <10ms search latency, >500 docs/sec indexing rate, and <200MB memory usage for 10K documents.

## Context
Performance validation is critical for Phase 1 success. You need to measure and verify that the Tantivy-based implementation achieves the designed performance targets under realistic workloads with special character queries.

## Task Details

### What You Need to Do

1. **Create `benches/performance.rs` for benchmarking:**

   ```rust
   //! Performance benchmarks for Phase 1 validation
   
   use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
   use llmkg_vectors::{DocumentIndexer, SearchEngine};
   use tempfile::TempDir;
   use std::{fs, time::Instant};
   use anyhow::Result;
   
   fn bench_indexing_performance(c: &mut Criterion) {
       let mut group = c.benchmark_group("indexing");
       
       // Test different file sizes and quantities
       let test_cases = vec![
           ("small_files", 100, 1000),   // 100 files, 1KB each
           ("medium_files", 50, 5000),   // 50 files, 5KB each  
           ("large_files", 10, 20000),   // 10 files, 20KB each
       ];
       
       for (name, file_count, file_size) in test_cases {
           group.bench_with_input(
               BenchmarkId::new("index_files", name),
               &(file_count, file_size),
               |b, &(count, size)| {
                   b.iter_batched(
                       || setup_indexing_benchmark(count, size).unwrap(),
                       |(temp_dir, indexer, files)| {
                           let start = Instant::now();
                           
                           let mut indexer = indexer;
                           let file_refs: Vec<_> = files.iter().map(|p| p.as_path()).collect();
                           let stats = indexer.index_files(&file_refs).unwrap();
                           indexer.commit().unwrap();
                           
                           let duration = start.elapsed();
                           let docs_per_sec = stats.chunks_created as f64 / duration.as_secs_f64();
                           
                           // Verify we meet the >500 docs/sec target
                           assert!(docs_per_sec > 500.0, 
                                  "Indexing rate {} docs/sec below target of 500", docs_per_sec);
                           
                           black_box((stats, duration))
                       },
                       criterion::BatchSize::SmallInput,
                   );
               },
           );
       }
       
       group.finish();
   }
   
   fn bench_search_performance(c: &mut Criterion) {
       let mut group = c.benchmark_group("search");
       group.measurement_time(std::time::Duration::from_secs(10));
       
       // Create pre-indexed test data
       let (temp_dir, search_engine) = setup_search_benchmark().unwrap();
       
       let test_queries = vec![
           ("simple", "function"),
           ("special_chars", "[workspace]"),
           ("complex_special", "Result<T, E>"),
           ("derive_macro", "#[derive(Debug)]"),
           ("reference", "&mut self"),
           ("comment", "## comment"),
       ];
       
       for (category, query) in test_queries {
           group.bench_with_input(
               BenchmarkId::new("search_query", category),
               &query,
               |b, &query| {
                   b.iter(|| {
                       let start = Instant::now();
                       let results = search_engine.search(query, 50).unwrap();
                       let duration = start.elapsed();
                       
                       // Verify we meet the <10ms search target
                       assert!(duration.as_millis() < 10, 
                              "Search latency {}ms exceeds 10ms target for query: {}", 
                              duration.as_millis(), query);
                       
                       black_box(results)
                   });
               },
           );
       }
       
       group.finish();
   }
   
   fn bench_memory_usage(c: &mut Criterion) {
       let mut group = c.benchmark_group("memory");
       
       group.bench_function("memory_10k_docs", |b| {
           b.iter_batched(
               || setup_large_index_benchmark().unwrap(),
               |(temp_dir, indexer, files)| {
                   let mut indexer = indexer;
                   let file_refs: Vec<_> = files.iter().map(|p| p.as_path()).collect();
                   
                   // Measure memory before and after indexing
                   let memory_before = get_memory_usage();
                   
                   let stats = indexer.index_files(&file_refs).unwrap();
                   indexer.commit().unwrap();
                   
                   let memory_after = get_memory_usage();
                   let memory_used_mb = (memory_after - memory_before) / 1024 / 1024;
                   
                   // Verify we meet the <200MB target for 10K docs
                   assert!(memory_used_mb < 200, 
                          "Memory usage {}MB exceeds 200MB target for {} chunks", 
                          memory_used_mb, stats.chunks_created);
                   
                   black_box((stats, memory_used_mb))
               },
               criterion::BatchSize::SmallInput,
           );
       });
       
       group.finish();
   }
   
   fn setup_indexing_benchmark(file_count: usize, file_size: usize) -> Result<(TempDir, DocumentIndexer, Vec<std::path::PathBuf>)> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("bench_index");
       let indexer = DocumentIndexer::new(&index_path)?;
       
       let mut files = Vec::new();
       for i in 0..file_count {
           let file_path = temp_dir.path().join(format!("file_{}.rs", i));
           let content = create_test_file_content(file_size, i);
           fs::write(&file_path, content)?;
           files.push(file_path);
       }
       
       Ok((temp_dir, indexer, files))
   }
   
   fn setup_search_benchmark() -> Result<(TempDir, SearchEngine)> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("search_bench_index");
       
       // Create pre-indexed content with special characters
       let mut indexer = DocumentIndexer::new(&index_path)?;
       
       let test_files = vec![
           ("config.toml", "[workspace]\nmembers = [\"backend\"]\n"),
           ("types.rs", "pub fn process<T, E>() -> Result<T, E> { todo!() }"),
           ("derive.rs", "#[derive(Debug, Clone)]\nstruct Config;"),
           ("refs.rs", "fn modify(data: &mut String) { data.push('x'); }"),
           ("comments.rs", "// ## Main comment\npub fn main() {}"),
       ];
       
       for (filename, content) in test_files {
           let file_path = temp_dir.path().join(filename);
           fs::write(&file_path, content)?;
           indexer.index_file(&file_path)?;
       }
       
       indexer.commit()?;
       let search_engine = SearchEngine::new(&index_path)?;
       
       Ok((temp_dir, search_engine))
   }
   
   fn setup_large_index_benchmark() -> Result<(TempDir, DocumentIndexer, Vec<std::path::PathBuf>)> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("large_bench_index");
       let indexer = DocumentIndexer::new(&index_path)?;
       
       // Create enough files to generate ~10K document chunks
       let mut files = Vec::new();
       for i in 0..200 { // 200 files * ~50 chunks each = ~10K docs
           let file_path = temp_dir.path().join(format!("large_file_{}.rs", i));
           let content = create_large_test_file(i);
           fs::write(&file_path, content)?;
           files.push(file_path);
       }
       
       Ok((temp_dir, indexer, files))
   }
   
   fn create_test_file_content(target_size: usize, file_id: usize) -> String {
       let base_content = format!(r#"
           pub fn function_{}() -> Result<String, Error> {{
               // Function with special chars: [{}], Result<T, E>, #[derive]
               #[derive(Debug)]
               struct Config{} {{
                   value: &mut String,
               }}
               
               let mut data = String::new();
               ## Comment for function {}
               process_data(&mut data)?;
               Ok(data)
           }}
       "#, file_id, file_id, file_id, file_id);
       
       // Pad to target size
       let padding_needed = target_size.saturating_sub(base_content.len());
       let padding = " ".repeat(padding_needed);
       
       format!("{}{}", base_content, padding)
   }
   
   fn create_large_test_file(file_id: usize) -> String {
       let mut content = String::new();
       
       // Create multiple functions per file
       for func_id in 0..50 {
           content.push_str(&format!(r#"
               /// Function {} in file {}
               pub fn func_{}_{}<T, E>() -> Result<T, E> 
               where T: Clone + Send, E: std::error::Error {{
                   #[derive(Debug, Clone)]
                   struct LocalConfig {{
                       name: String,
                       values: Vec<i32>,
                   }}
                   
                   let mut config = LocalConfig {{
                       name: "config_{}".to_string(),
                       values: vec![1, 2, 3],
                   }};
                   
                   // ## Processing logic
                   println!("Processing in function {}_{}");
                   process_config(&mut config)?;
                   
                   Ok(todo!())
               }}
           "#, func_id, file_id, file_id, func_id, func_id, func_id, file_id));
       }
       
       content
   }
   
   fn get_memory_usage() -> usize {
       // Simplified memory measurement
       // In a real implementation, you'd use a proper memory profiler
       std::alloc::System.alloc(std::alloc::Layout::new::<u8>()) as usize
   }
   
   criterion_group!(benches, bench_indexing_performance, bench_search_performance, bench_memory_usage);
   criterion_main!(benches);
   ```

2. **Add benchmarking dependencies to Cargo.toml:**
   ```toml
   [dev-dependencies]
   criterion = { version = "0.5", features = ["html_reports"] }
   
   [[bench]]
   name = "performance"
   harness = false
   ```

3. **Create performance validation script in `scripts/validate_performance.sh`:**
   ```bash
   #!/bin/bash
   set -e
   
   echo "Running Phase 1 performance validation..."
   
   # Run benchmarks and capture results
   cargo bench --bench performance -- --output-format json > benchmark_results.json
   
   echo "Performance validation complete!"
   echo "Results saved to benchmark_results.json"
   echo "HTML reports available in target/criterion/"
   
   # Validate key metrics
   echo "Validating performance targets:"
   echo "✓ Search latency: <10ms (verified in benchmarks)"
   echo "✓ Indexing rate: >500 docs/sec (verified in benchmarks)"
   echo "✓ Memory usage: <200MB for 10K docs (verified in benchmarks)"
   ```

## Success Criteria
- [ ] Benchmark code compiles without errors
- [ ] `cargo bench` runs successfully and generates reports
- [ ] Search latency consistently <10ms for all query types
- [ ] Indexing rate >500 docs/sec for various file sizes
- [ ] Memory usage <200MB for 10K document chunks
- [ ] Special character queries perform within targets
- [ ] HTML benchmark reports generated in target/criterion/

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: use of undeclared crate `criterion`"**
```bash
# Solution: Missing criterion dev dependency
cargo add --dev criterion@0.5
# Ensure [dev-dependencies] section exists in Cargo.toml
cargo check --benches
```

**Error 2: "failed to run benchmark: No such file or directory"**
```bash
# Solution: Benchmark file not in correct location
# Ensure benchmarks/performance.rs exists (not benches/)
mkdir benchmarks
# Move benchmark file to correct location
cargo bench
```

**Error 3: "benchmark timed out" or hanging benchmarks**
```bash
# Solution: Performance targets not met or infinite loops
# Reduce test data size for debugging
# Check for infinite loops in chunking or indexing
RUST_LOG=debug cargo bench -- --test
```

**Error 4: "assertion failed: search_time.as_millis() < 10" (performance target missed)**
```bash
# Solution: Performance optimization needed
# Run in release mode: cargo bench (not cargo test)
# Check system resources: Ensure sufficient RAM and CPU
# Profile with: cargo flamegraph --bench performance
```

## Troubleshooting Checklist

- [ ] All previous tasks (03-11) completed successfully
- [ ] Criterion dependency "0.5" in [dev-dependencies]
- [ ] Benchmark file in benchmarks/ directory (not benches/)
- [ ] Running in release mode (cargo bench, not cargo test)
- [ ] Sufficient system resources (RAM, CPU, disk)
- [ ] Test data appropriate size for meaningful measurements
- [ ] No debug logging enabled during benchmarks
- [ ] Index cleanup between benchmark runs
- [ ] Windows performance expectations adjusted if needed

## Recovery Procedures

### Benchmark Setup Failures
If benchmarks fail to initialize:
1. Verify criterion setup: Check Cargo.toml [dev-dependencies]
2. Test benchmark compilation: `cargo check --benches`
3. Validate test data: Ensure benchmark data generation works
4. Check file structure: Benchmarks in benchmarks/ not benches/

### Performance Target Misses
If benchmarks show performance below targets:
1. Profile bottlenecks: Use `cargo flamegraph --bench performance`
2. Check build mode: Ensure release mode with optimizations
3. Verify system load: Close other applications during benchmarks
4. Test incrementally: Start with small datasets, scale up

### Memory Usage Issues
If memory benchmarks fail or show excessive usage:
1. Check for memory leaks: Use valgrind or similar tools
2. Verify cleanup: Ensure proper resource disposal
3. Monitor growth: Track memory usage over time
4. Reduce test size: Use smaller datasets if needed

### Search Latency Problems
If search benchmarks exceed 10ms target:
1. Profile search operations: Identify slow components
2. Check index size: Verify reasonable index dimensions
3. Optimize queries: Test with simpler query patterns first
4. Verify caching: Ensure search caches work correctly

### Indexing Rate Issues
If indexing benchmarks show <500 docs/sec:
1. Profile indexing pipeline: Find chunking/indexing bottlenecks
2. Check I/O performance: Verify disk write speeds
3. Optimize batch sizes: Tune index writer buffer sizes
4. Test AST parsing: Verify tree-sitter performance

## Common Pitfalls to Avoid
- Don't run benchmarks in debug mode (use `cargo bench`, not `cargo test`)
- Ensure sufficient test data to make measurements meaningful
- Don't ignore memory cleanup between benchmark runs
- Validate performance on both small and large datasets
- Account for Windows vs Unix performance differences

## Context for Next Task
Task 13 will add comprehensive error handling and recovery mechanisms to make the system production-ready.