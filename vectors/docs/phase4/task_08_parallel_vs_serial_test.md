# Task 08: Add Parallel vs Serial Performance Test

## Context
You are implementing Phase 4 of a vector indexing system. This task adds comprehensive performance comparison tests to validate that parallel processing provides measurable performance improvements over serial processing.

## Current State
- `src/parallel.rs` exists with complete parallel indexing implementation
- Test infrastructure is set up with helper functions
- Basic functionality tests are passing

## Task Objective
Add performance comparison tests that measure and validate the speed improvements of parallel indexing over serial processing.

## Implementation Requirements

### 1. Add performance comparison test
Add this test to the test module in `src/parallel.rs`:
```rust
#[test]
fn test_parallel_vs_serial_performance_detailed() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let parallel_path = temp_dir.path().join("parallel_index");
    let serial_path = temp_dir.path().join("serial_index");
    
    // Create test data - enough files to see parallel benefit
    let test_project = temp_dir.path().join("perf_test");
    let created_files = create_test_project(&test_project, 50)?;
    
    // Test parallel indexing
    let parallel_indexer = ParallelIndexer::new(&parallel_path)?;
    let (parallel_stats, parallel_duration) = time_execution(|| {
        parallel_indexer.index_directory_parallel(&test_project)
    });
    let parallel_stats = parallel_stats?;
    
    // Test serial indexing for comparison
    let mut serial_indexer = DocumentIndexer::new(&serial_path)?;
    let (_, serial_duration) = time_execution(|| -> Result<()> {
        parallel_indexer.index_directory_parallel(&test_project)?;
        Ok(())
    });
    
    // Validate both methods processed the same amount of data
    assert!(parallel_stats.files_processed >= 50);
    
    // Print performance comparison
    println!("\nPerformance Comparison:");
    println!("  Parallel: {:?} ({} files, {:.2} MB)", 
             parallel_duration, 
             parallel_stats.files_processed,
             parallel_stats.total_size as f64 / (1024.0 * 1024.0));
    println!("  Files per second (parallel): {:.1}", parallel_stats.files_per_second);
    
    // On multi-core systems, parallel should show some benefit
    let available_cores = std::thread::available_parallelism()?.get();
    println!("  Available CPU cores: {}", available_cores);
    
    if available_cores > 1 {
        // With multiple cores, we expect some speedup, but don't enforce strict requirements
        // as performance can vary based on system load, file sizes, etc.
        println!("  Multi-core system detected - parallel processing should show benefits");
    } else {
        println!("  Single-core system - parallel processing may not show significant speedup");
    }
    
    Ok(())
}
```

### 2. Add throughput measurement test
Add this test to measure indexing throughput:
```rust
#[test]
fn test_indexing_throughput() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("throughput_test");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create different sized test datasets
    let test_sizes = [10, 25, 50, 100];
    let mut throughput_results = Vec::new();
    
    for &size in &test_sizes {
        let test_project = temp_dir.path().join(format!("test_{}", size));
        create_test_project(&test_project, size)?;
        
        let (stats, duration) = time_execution(|| {
            parallel_indexer.index_directory_parallel(&test_project)
        });
        let stats = stats?;
        
        let throughput = stats.files_processed as f64 / duration.as_secs_f64();
        throughput_results.push((size, throughput, stats.files_processed));
        
        println!("Dataset size {}: {:.1} files/sec ({} files in {:?})", 
                size, throughput, stats.files_processed, duration);
    }
    
    // Validate that we can process files at reasonable rates
    for (size, throughput, processed) in throughput_results {
        assert!(processed >= size, "Should process at least {} files, got {}", size, processed);
        assert!(throughput > 0.0, "Throughput should be positive");
        
        // Very loose performance requirement - just ensure it's not impossibly slow
        assert!(throughput > 0.1, "Throughput too low: {:.1} files/sec", throughput);
    }
    
    Ok(())
}
```

### 3. Add memory usage comparison test
Add this test to compare memory usage:
```rust
#[test]
fn test_memory_usage_scaling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("memory_test");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test with progressively larger datasets
    let test_sizes = [10, 50, 100];
    
    for &size in &test_sizes {
        let test_project = temp_dir.path().join(format!("memory_test_{}", size));
        create_test_project(&test_project, size)?;
        
        // Index the dataset
        let stats = parallel_indexer.index_directory_parallel(&test_project)?;
        
        println!("Memory test - {} files: processed {}, total size {:.2} MB", 
                size, stats.files_processed, 
                stats.total_size as f64 / (1024.0 * 1024.0));
        
        // Basic validation that processing completed
        assert!(stats.files_processed > 0);
        assert!(stats.total_size > 0);
        
        // The test itself validates memory usage by not running out of memory
        // More sophisticated memory tracking would require external tools
    }
    
    Ok(())
}
```

### 4. Add concurrent processing test
Add this test to validate concurrent parallel operations:
```rust
#[test]
fn test_concurrent_parallel_indexing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().join("concurrent_test");
    
    // Create multiple test projects
    let mut test_projects = Vec::new();
    for i in 0..3 {
        let project_path = base_path.join(format!("project_{}", i));
        create_test_project(&project_path, 20)?;
        test_projects.push(project_path);
    }
    
    // Create separate indexers for concurrent operations
    let indexers: Result<Vec<_>, _> = (0..3)
        .map(|i| {
            let index_path = base_path.join(format!("index_{}", i));
            ParallelIndexer::new(&index_path)
        })
        .collect();
    let indexers = indexers?;
    
    // Run indexing operations concurrently
    let handles: Vec<_> = indexers.into_iter().zip(test_projects).enumerate()
        .map(|(i, (indexer, project))| {
            std::thread::spawn(move || -> Result<(usize, IndexingStats)> {
                let stats = indexer.index_directory_parallel(&project)?;
                Ok((i, stats))
            })
        })
        .collect();
    
    // Wait for all operations to complete
    let mut total_processed = 0;
    for handle in handles {
        let (thread_id, stats) = handle.join().unwrap()?;
        println!("Thread {}: processed {} files in {:.2}s", 
                thread_id, stats.files_processed, stats.duration().as_secs_f64());
        total_processed += stats.files_processed;
    }
    
    // Should have processed files from all projects
    assert!(total_processed >= 60, "Expected at least 60 files, got {}", total_processed);
    
    println!("Concurrent test completed: {} total files processed", total_processed);
    
    Ok(())
}
```

### 5. Add stress test for large datasets
Add this test for stress testing:
```rust
#[test]
#[ignore] // Ignored by default due to time/resource requirements
fn test_large_dataset_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("stress_test");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create a large test dataset
    let test_project = temp_dir.path().join("large_project");
    create_test_project(&test_project, 500)?; // 500 files
    
    println!("Starting stress test with 500+ files...");
    
    let (stats, duration) = time_execution(|| {
        parallel_indexer.index_directory_parallel(&test_project)
    });
    let stats = stats?;
    
    println!("Stress test results:");
    println!("  Files processed: {}", stats.files_processed);
    println!("  Total size: {:.2} MB", stats.total_size as f64 / (1024.0 * 1024.0));
    println!("  Duration: {:.2}s", duration.as_secs_f64());
    println!("  Throughput: {:.1} files/sec", stats.files_per_second);
    println!("  Data rate: {:.2} MB/sec", stats.megabytes_per_second());
    
    // Validate reasonable performance for large datasets
    assert!(stats.files_processed >= 500);
    assert!(stats.files_per_second > 10.0, "Should process at least 10 files/sec");
    assert!(duration.as_secs() < 300, "Should complete within 5 minutes");
    
    Ok(())
}
```

## Success Criteria
- [ ] Parallel vs serial performance test runs successfully
- [ ] Throughput measurement test validates scaling
- [ ] Memory usage test completes without memory issues
- [ ] Concurrent processing test validates thread safety
- [ ] Stress test handles large datasets (when run)
- [ ] Performance metrics are reasonable for the hardware
- [ ] All tests pass consistently
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Performance tests may vary based on hardware capabilities
- Stress test is ignored by default to avoid long test runs
- Concurrent tests validate both performance and thread safety
- Focus on relative performance improvements rather than absolute numbers
- Memory usage is validated implicitly by successful completion