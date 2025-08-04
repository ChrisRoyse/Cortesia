# Task 07: Test Rayon Parallelism on Windows

## Context
You are continuing architecture validation (Phase 0, Task 7). Tasks 05-06 validated Tantivy and LanceDB. Now you need to validate that Rayon (data parallelism library) works correctly on Windows with thread safety and optimal performance.

## Objective
Implement and test Rayon parallel processing on Windows, focusing on thread safety, performance validation, and Windows-specific thread handling for the vector search system.

## Requirements
1. Test basic parallel iterator functionality
2. Test parallel file processing (critical for code indexing)
3. Test thread safety with shared data structures
4. Test performance comparison (sequential vs parallel)
5. Validate Windows thread creation and management
6. Test parallel vector operations

## Implementation for validation.rs (extend existing)
```rust
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::collections::HashMap;
use anyhow::Result;
use tracing::{info, debug, warn};

pub struct RayonValidator;

impl RayonValidator {
    /// Test Rayon parallelism on Windows
    pub fn validate_rayon_windows() -> Result<()> {
        info!("Starting Rayon parallelism validation on Windows");
        
        // Test basic parallel operations
        Self::test_basic_parallel_ops()?;
        
        // Test parallel file processing
        Self::test_parallel_file_processing()?;
        
        // Test thread safety
        Self::test_thread_safety()?;
        
        // Test performance comparison
        Self::test_performance_comparison()?;
        
        // Test with vector operations
        Self::test_parallel_vectors()?;
        
        info!("Rayon parallelism validation completed successfully");
        Ok(())
    }
    
    fn test_basic_parallel_ops() -> Result<()> {
        debug!("Testing basic parallel operations");
        
        // Test parallel map
        let data: Vec<i32> = (0..1000).collect();
        let results: Vec<i32> = data.par_iter().map(|x| x * 2).collect();
        
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[999], 1998);
        
        // Test parallel filter
        let filtered: Vec<i32> = data.par_iter()
            .filter(|&&x| x % 2 == 0)
            .cloned()
            .collect();
        
        assert_eq!(filtered.len(), 500);
        
        // Test parallel reduce
        let sum: i32 = data.par_iter().sum();
        assert_eq!(sum, 499500); // Sum of 0..1000
        
        debug!("Basic parallel operations test passed");
        Ok(())
    }
    
    fn test_parallel_file_processing() -> Result<()> {
        debug!("Testing parallel file processing");
        
        // Create test files
        std::fs::create_dir_all("test_data/parallel")?;
        let test_files: Vec<String> = (0..100)
            .map(|i| format!("test_data/parallel/file_{}.rs", i))
            .collect();
        
        // Create files with Rust code content
        for (i, file_path) in test_files.iter().enumerate() {
            let content = format!(
                "pub fn function_{}() -> Result<String, Error> {{\n    Ok(\"test_{}\")\n}}",
                i, i
            );
            std::fs::write(file_path, content)?;
        }
        
        // Process files in parallel
        let results: Vec<(String, usize)> = test_files
            .par_iter()
            .map(|file_path| {
                let content = std::fs::read_to_string(file_path).unwrap_or_default();
                (file_path.clone(), content.len())
            })
            .collect();
        
        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|(_, size)| *size > 0));
        
        debug!("Parallel file processing test passed");
        Ok(())
    }
    
    fn test_thread_safety() -> Result<()> {
        debug!("Testing thread safety with shared data");
        
        // Test with Arc<Mutex<T>>
        let shared_counter = Arc::new(Mutex::new(0));
        let shared_map: Arc<Mutex<HashMap<String, i32>>> = Arc::new(Mutex::new(HashMap::new()));
        
        // Parallel operations on shared data
        (0..1000).into_par_iter().for_each(|i| {
            // Update counter
            {
                let mut counter = shared_counter.lock().unwrap();
                *counter += 1;
            }
            
            // Update map
            {
                let mut map = shared_map.lock().unwrap();
                map.insert(format!("key_{}", i), i);
            }
        });
        
        let final_counter = *shared_counter.lock().unwrap();
        let final_map_size = shared_map.lock().unwrap().len();
        
        assert_eq!(final_counter, 1000);
        assert_eq!(final_map_size, 1000);
        
        debug!("Thread safety test passed");
        Ok(())
    }
    
    fn test_performance_comparison() -> Result<()> {
        debug!("Testing performance comparison (sequential vs parallel)");
        
        let data: Vec<i32> = (0..100_000).collect();
        
        // Sequential processing
        let start = Instant::now();
        let sequential_result: i64 = data.iter()
            .map(|&x| (x as i64).pow(2))
            .sum();
        let sequential_time = start.elapsed();
        
        // Parallel processing
        let start = Instant::now();
        let parallel_result: i64 = data.par_iter()
            .map(|&x| (x as i64).pow(2))
            .sum();
        let parallel_time = start.elapsed();
        
        assert_eq!(sequential_result, parallel_result);
        
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        
        info!(
            "Performance comparison - Sequential: {:?}, Parallel: {:?}, Speedup: {:.2}x",
            sequential_time, parallel_time, speedup
        );
        
        if speedup < 1.5 {
            warn!("Parallel speedup is less than 1.5x, may indicate threading issues");
        }
        
        debug!("Performance comparison test completed");
        Ok(())
    }
    
    fn test_parallel_vectors() -> Result<()> {
        debug!("Testing parallel vector operations");
        
        // Create large vector dataset
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| (0..384).map(|j| (i * j) as f32 * 0.001).collect())
            .collect();
        
        // Parallel vector normalization
        let normalized: Vec<Vec<f32>> = vectors
            .par_iter()
            .map(|vec| {
                let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if magnitude > 0.0 {
                    vec.iter().map(|x| x / magnitude).collect()
                } else {
                    vec.clone()
                }
            })
            .collect();
        
        assert_eq!(normalized.len(), 1000);
        assert_eq!(normalized[0].len(), 384);
        
        // Parallel dot product computation
        let query_vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        
        let similarities: Vec<f32> = normalized
            .par_iter()
            .map(|vec| {
                vec.iter()
                    .zip(query_vector.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        
        assert_eq!(similarities.len(), 1000);
        
        debug!("Parallel vector operations test passed");
        Ok(())
    }
    
    /// Test Windows-specific thread behavior
    pub fn test_windows_threading() -> Result<()> {
        debug!("Testing Windows-specific threading behavior");
        
        // Test thread pool configuration
        let thread_count = rayon::current_num_threads();
        info!("Rayon thread pool size: {}", thread_count);
        
        // Test thread IDs
        let thread_ids: Vec<_> = (0..100).into_par_iter()
            .map(|_| std::thread::current().id())
            .collect();
        
        // Count unique thread IDs
        let mut unique_threads = std::collections::HashSet::new();
        for id in thread_ids {
            unique_threads.insert(id);
        }
        
        info!("Number of unique threads used: {}", unique_threads.len());
        assert!(unique_threads.len() > 1, "Multiple threads should be used");
        
        debug!("Windows threading test passed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rayon_validation() {
        RayonValidator::validate_rayon_windows().unwrap();
    }
    
    #[test]
    fn test_windows_threading() {
        RayonValidator::test_windows_threading().unwrap();
    }
}
```

## Implementation Steps
1. Add RayonValidator struct to validation.rs
2. Implement basic parallel operations testing (map, filter, reduce)
3. Implement parallel file processing for code files
4. Implement thread safety testing with Arc<Mutex<T>>
5. Implement performance comparison (sequential vs parallel)
6. Implement parallel vector operations testing
7. Add Windows-specific threading tests
8. Run tests to verify performance and correctness

## Success Criteria
- [ ] RayonValidator struct implemented and compiling
- [ ] Basic parallel operations work correctly
- [ ] Parallel file processing handles multiple Rust files
- [ ] Thread safety tests pass with shared data structures
- [ ] Parallel processing shows performance improvement over sequential
- [ ] Vector operations work correctly in parallel
- [ ] Multiple threads are utilized on Windows
- [ ] All tests pass (`cargo test`)

## Test Command
```bash
cargo test test_rayon_validation
cargo test test_windows_threading
```

## Performance Expectations
- Parallel processing should show at least 1.5x speedup on multi-core systems
- Thread pool should utilize multiple threads
- No data races or deadlocks should occur

## Time Estimate
10 minutes

## Next Task
Task 08: Test tree-sitter parsing functionality for semantic code chunking.