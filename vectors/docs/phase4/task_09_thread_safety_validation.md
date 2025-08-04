# Task 09: Thread Safety Validation

## Context
You are implementing Phase 4 of a vector indexing system. This task focuses on comprehensive thread safety validation for the parallel indexing system. After implementing performance comparison tests, you now need to ensure that the parallel indexer handles concurrent access correctly and safely across multiple threads.

## Current State
- `src/parallel.rs` exists with complete parallel indexing implementation
- Performance comparison tests validate speed improvements
- Basic thread safety tests exist but need comprehensive validation
- Rayon parallel processing is working correctly

## Task Objective
Implement comprehensive thread safety validation tests that ensure the parallel indexer can handle concurrent operations, shared state access, and multi-threaded scenarios without data races or corruption.

## Implementation Requirements

### 1. Add comprehensive concurrent access test
Add this test to the test module in `src/parallel.rs`:
```rust
#[test]
fn test_thread_safety_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("thread_safety_test");
    
    // Create shared indexer instance
    let parallel_indexer = Arc::new(ParallelIndexer::new(&index_path)?);
    
    // Create multiple test datasets
    let mut test_datasets = Vec::new();
    for i in 0..5 {
        let dataset_path = temp_dir.path().join(format!("dataset_{}", i));
        create_test_project(&dataset_path, 15)?;
        test_datasets.push(dataset_path);
    }
    
    // Launch multiple threads that access the same indexer concurrently
    let handles: Vec<_> = test_datasets.into_iter().enumerate()
        .map(|(thread_id, dataset)| {
            let indexer = Arc::clone(&parallel_indexer);
            std::thread::spawn(move || -> Result<(usize, IndexingStats)> {
                // Each thread processes its own dataset
                let stats = indexer.index_directory_parallel(&dataset)?;
                
                // Simulate some additional work to increase chance of race conditions
                std::thread::sleep(std::time::Duration::from_millis(10));
                
                Ok((thread_id, stats))
            })
        })
        .collect();
    
    // Collect results from all threads
    let mut all_results = Vec::new();
    for handle in handles {
        let result = handle.join().unwrap()?;
        all_results.push(result);
    }
    
    // Validate that all threads completed successfully
    assert_eq!(all_results.len(), 5);
    
    let total_files: usize = all_results.iter().map(|(_, stats)| stats.files_processed).sum();
    assert!(total_files >= 75, "Expected at least 75 files total, got {}", total_files);
    
    // Verify no thread had zero processing (indicates a deadlock or similar issue)
    for (thread_id, stats) in &all_results {
        assert!(stats.files_processed > 0, "Thread {} processed no files", thread_id);
        println!("Thread {}: {} files, {:.2}s", 
                thread_id, stats.files_processed, stats.duration().as_secs_f64());
    }
    
    println!("Thread safety test completed: {} threads, {} total files", 
            all_results.len(), total_files);
    
    Ok(())
}
```

### 2. Add atomic operations validation test
Add this test to validate atomic operations and shared state:
```rust
#[test]
fn test_atomic_operations_safety() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("atomic_test");
    let parallel_indexer = Arc::new(ParallelIndexer::new(&index_path)?);
    
    // Shared counters to track operations
    let operations_counter = Arc::new(AtomicUsize::new(0));
    let files_counter = Arc::new(AtomicUsize::new(0));
    
    // Create test data
    let test_project = temp_dir.path().join("atomic_project");
    create_test_project(&test_project, 30)?;
    
    // Launch multiple threads that perform concurrent operations
    let num_threads = 8;
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let indexer = Arc::clone(&parallel_indexer);
        let ops_counter = Arc::clone(&operations_counter);
        let files_counter = Arc::clone(&files_counter);
        let project_path = test_project.clone();
        
        std::thread::spawn(move || -> Result<()> {
            for iteration in 0..3 {
                // Increment operation counter atomically
                ops_counter.fetch_add(1, Ordering::SeqCst);
                
                // Perform indexing operation
                let stats = indexer.index_directory_parallel(&project_path)?;
                
                // Update files counter atomically
                files_counter.fetch_add(stats.files_processed, Ordering::SeqCst);
                
                println!("Thread {} iteration {}: {} files", 
                        thread_id, iteration, stats.files_processed);
                
                // Small delay to increase contention
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Ok(())
        })
    }).collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    // Validate atomic counters
    let total_operations = operations_counter.load(Ordering::SeqCst);
    let total_files = files_counter.load(Ordering::SeqCst);
    
    assert_eq!(total_operations, num_threads * 3, 
              "Expected {} operations, got {}", num_threads * 3, total_operations);
    assert!(total_files > 0, "No files were processed");
    
    println!("Atomic operations test: {} operations, {} total file accesses", 
            total_operations, total_files);
    
    Ok(())
}
```

### 3. Add resource contention test
Add this test to validate behavior under heavy resource contention:
```rust
#[test]
fn test_resource_contention_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().join("contention_test");
    
    // Create many small datasets to increase I/O contention
    let mut datasets = Vec::new();
    for i in 0..10 {
        let dataset_path = base_path.join(format!("dataset_{}", i));
        create_test_project(&dataset_path, 5)?; // Small datasets for quick processing
        datasets.push(dataset_path);
    }
    
    // Create multiple indexers sharing the same base directory
    let indexers: Result<Vec<_>, _> = (0..10).map(|i| {
        let index_path = base_path.join(format!("index_{}", i));
        ParallelIndexer::new(&index_path)
    }).collect();
    let indexers: Vec<_> = indexers?.into_iter().map(Arc::new).collect();
    
    // Launch many concurrent operations to create resource contention
    let handles: Vec<_> = indexers.into_iter().zip(datasets).enumerate()
        .map(|(i, (indexer, dataset))| {
            std::thread::spawn(move || -> Result<(usize, IndexingStats)> {
                // Stagger start times slightly to increase contention
                std::thread::sleep(std::time::Duration::from_millis((i % 5) * 2));
                
                let stats = indexer.index_directory_parallel(&dataset)?;
                Ok((i, stats))
            })
        })
        .collect();
    
    // Track completion times and results
    let mut results = Vec::new();
    let start_time = std::time::Instant::now();
    
    for handle in handles {
        let result = handle.join().unwrap()?;
        results.push((result, start_time.elapsed()));
    }
    
    // Validate all operations completed successfully despite contention
    assert_eq!(results.len(), 10);
    
    let total_files: usize = results.iter().map(|((_, stats), _)| stats.files_processed).sum();
    assert!(total_files >= 50, "Expected at least 50 files, got {}", total_files);
    
    // Check that operations completed in reasonable time despite contention
    let max_duration = results.iter().map(|(_, duration)| *duration).max().unwrap();
    assert!(max_duration.as_secs() < 60, "Operations took too long: {:?}", max_duration);
    
    println!("Resource contention test: {} operations completed in {:?}", 
            results.len(), max_duration);
    
    Ok(())
}
```

### 4. Add deadlock prevention test
Add this test to ensure no deadlocks occur:
```rust
#[test]
fn test_deadlock_prevention() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("deadlock_test");
    let parallel_indexer = Arc::new(ParallelIndexer::new(&index_path)?);
    
    // Create test data
    let test_project = temp_dir.path().join("deadlock_project");
    create_test_project(&test_project, 20)?;
    
    // Use a timeout to detect potential deadlocks
    let timeout_duration = std::time::Duration::from_secs(30);
    let start_time = std::time::Instant::now();
    
    // Launch threads that could potentially create circular dependencies
    let handles: Vec<_> = (0..6).map(|thread_id| {
        let indexer = Arc::clone(&parallel_indexer);
        let project_path = test_project.clone();
        
        std::thread::spawn(move || -> Result<usize> {
            let mut total_processed = 0;
            
            // Each thread performs multiple operations that could interact
            for _ in 0..5 {
                let stats = indexer.index_directory_parallel(&project_path)?;
                total_processed += stats.files_processed;
                
                // Yield to other threads to encourage race conditions
                std::thread::yield_now();
            }
            
            Ok(total_processed)
        })
    }).collect();
    
    // Wait for all threads with timeout to detect deadlocks
    let mut total_processed = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        // Check if we're approaching timeout
        if start_time.elapsed() > timeout_duration {
            panic!("Potential deadlock detected - operations took longer than {:?}", timeout_duration);
        }
        
        match handle.join() {
            Ok(result) => {
                let processed = result?;
                total_processed += processed;
                println!("Thread {} completed: {} files", i, processed);
            }
            Err(_) => {
                panic!("Thread {} panicked - possible deadlock or race condition", i);
            }
        }
    }
    
    let total_duration = start_time.elapsed();
    
    // Validate successful completion
    assert!(total_processed > 0, "No files were processed");
    assert!(total_duration < timeout_duration, 
           "Operations took too long, possible performance issue");
    
    println!("Deadlock prevention test completed: {} files in {:?}", 
            total_processed, total_duration);
    
    Ok(())
}
```

### 5. Add data integrity validation test
Add this test to ensure data integrity under concurrent access:
```rust
#[test]
fn test_data_integrity_concurrent() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("integrity_test");
    let parallel_indexer = Arc::new(ParallelIndexer::new(&index_path)?);
    
    // Create deterministic test data for integrity checking
    let test_project = temp_dir.path().join("integrity_project");
    let expected_files = create_test_project(&test_project, 25)?;
    let expected_count = expected_files.len();
    
    // Run the same indexing operation from multiple threads
    let num_threads = 4;
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let indexer = Arc::clone(&parallel_indexer);
        let project_path = test_project.clone();
        
        std::thread::spawn(move || -> Result<IndexingStats> {
            indexer.index_directory_parallel(&project_path)
        })
    }).collect();
    
    // Collect all results
    let mut all_stats = Vec::new();
    for handle in handles {
        let stats = handle.join().unwrap()?;
        all_stats.push(stats);
    }
    
    // Validate data integrity - all threads should see consistent results
    for (i, stats) in all_stats.iter().enumerate() {
        // Each run should process at least the files we created
        assert!(stats.files_processed >= expected_count, 
               "Thread {} processed {} files, expected at least {}", 
               i, stats.files_processed, expected_count);
        
        // All runs should see similar amounts of data (within reason)
        assert!(stats.total_size > 0, "Thread {} saw zero data size", i);
        
        println!("Thread {} integrity check: {} files, {} bytes", 
                i, stats.files_processed, stats.total_size);
    }
    
    // Check consistency across threads (should be very similar)
    let files_counts: Vec<_> = all_stats.iter().map(|s| s.files_processed).collect();
    let sizes: Vec<_> = all_stats.iter().map(|s| s.total_size).collect();
    
    let min_files = *files_counts.iter().min().unwrap();
    let max_files = *files_counts.iter().max().unwrap();
    let min_size = *sizes.iter().min().unwrap();
    let max_size = *sizes.iter().max().unwrap();
    
    // Results should be identical or very close (allowing for minor variations)
    assert!(max_files - min_files <= 2, 
           "File count variation too high: {} to {}", min_files, max_files);
    
    // Size variation should be minimal (within 10% for concurrent access variations)
    let size_variation = (max_size - min_size) as f64 / min_size as f64;
    assert!(size_variation < 0.1, 
           "Size variation too high: {:.2}% ({} to {})", 
           size_variation * 100.0, min_size, max_size);
    
    println!("Data integrity validation passed: {} threads, consistent results", 
            all_stats.len());
    
    Ok(())
}
```

## Success Criteria
- [ ] Comprehensive concurrent access test validates shared indexer usage
- [ ] Atomic operations test ensures thread-safe counter updates
- [ ] Resource contention test handles high-concurrency scenarios
- [ ] Deadlock prevention test completes within timeout
- [ ] Data integrity test shows consistent results across threads
- [ ] All tests pass consistently across multiple runs
- [ ] No data races or corruption detected
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Thread safety is critical for production parallel processing
- Tests should reveal race conditions and synchronization issues
- Use Arc for shared ownership and proper synchronization primitives
- Timeout detection helps identify deadlocks early
- Data integrity validation ensures correctness under concurrency
- Tests may need multiple runs to catch intermittent race conditions