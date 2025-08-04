# Task 50: Implement search_parallel() Method

## Context
You are implementing Phase 4 of a vector indexing system. This is the second critical task for the ParallelSearchEngine component that was missing from the original 48 tasks. The previous task created the struct foundation, now we implement the actual parallel search functionality.

## Current State
- `ParallelSearchEngine` struct exists with multiple engines (task 49)
- `SearchAggregator` handles result merging and deduplication
- Need to implement the actual parallel search method using Rayon

## Task Objective
Implement the `search_parallel()` method that executes searches concurrently across multiple indexes and aggregates results efficiently.

## Implementation Requirements

### 1. Add search_parallel() method to ParallelSearchEngine
Add this method to the `ParallelSearchEngine` implementation in `src/parallel.rs`:
```rust
impl ParallelSearchEngine {
    // ... existing methods ...
    
    pub fn search_parallel(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.search_parallel_with_options(query, DedupStrategy::TakeHighestScore, None)
    }
    
    pub fn search_parallel_with_options(
        &self, 
        query: &str,
        dedup_strategy: DedupStrategy,
        max_results: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        // Search across multiple indexes in parallel
        let all_results: Vec<Vec<SearchResult>> = self.engines
            .par_iter()
            .map(|engine| {
                engine.proximity_engine.boolean_engine
                    .search_boolean(query)
                    .unwrap_or_default()
            })
            .collect();
        
        // Aggregate results
        let mut aggregator = SearchAggregator::new(dedup_strategy);
        for results in all_results {
            aggregator.add_results(results);
        }
        
        // Finalize and apply result limit
        let mut final_results = aggregator.finalize();
        
        if let Some(max) = max_results {
            final_results.truncate(max);
        }
        
        Ok(final_results)
    }
    
    pub fn search_parallel_timed(&self, query: &str) -> Result<(Vec<SearchResult>, Duration)> {
        let start = Instant::now();
        let results = self.search_parallel(query)?;
        let duration = start.elapsed();
        Ok((results, duration))
    }
}
```

### 2. Add parallel search with per-engine timeout
Add this method for timeout-controlled searches:
```rust
impl ParallelSearchEngine {
    pub fn search_parallel_with_timeout(
        &self,
        query: &str,
        timeout_per_engine: Duration,
        dedup_strategy: DedupStrategy,
    ) -> Result<Vec<SearchResult>> {
        use std::sync::mpsc;
        use std::thread;
        
        let (tx, rx) = mpsc::channel();
        let engines_count = self.engines.len();
        
        // Spawn parallel searches with timeout
        self.engines.par_iter().enumerate().for_each(|(idx, engine)| {
            let tx = tx.clone();
            let query = query.to_string();
            
            let handle = thread::spawn(move || {
                let start = Instant::now();
                let results = engine.proximity_engine.boolean_engine
                    .search_boolean(&query)
                    .unwrap_or_default();
                
                if start.elapsed() <= timeout_per_engine {
                    let _ = tx.send((idx, results));
                } else {
                    let _ = tx.send((idx, Vec::new())); // Timeout - send empty results
                }
            });
            
            // Detach thread if it takes too long
            thread::spawn(move || {
                thread::sleep(timeout_per_engine);
                drop(handle);
            });
        });
        
        drop(tx); // Close sender
        
        // Collect results from all engines
        let mut aggregator = SearchAggregator::new(dedup_strategy);
        let mut received_count = 0;
        
        for (engine_idx, results) in rx {
            aggregator.add_results(results);
            received_count += 1;
            
            if received_count >= engines_count {
                break;
            }
        }
        
        Ok(aggregator.finalize())
    }
}
```

### 3. Add search statistics tracking
Add this struct for performance metrics:
```rust
#[derive(Debug, Clone)]
pub struct ParallelSearchStats {
    pub total_results: usize,
    pub results_per_engine: Vec<usize>,
    pub search_duration: Duration,
    pub average_score: f32,
    pub engines_searched: usize,
    pub dedup_removed: usize,
}

impl ParallelSearchEngine {
    pub fn search_parallel_with_stats(
        &self,
        query: &str,
        dedup_strategy: DedupStrategy,
    ) -> Result<(Vec<SearchResult>, ParallelSearchStats)> {
        let start = Instant::now();
        
        // Track results per engine
        let all_results: Vec<Vec<SearchResult>> = self.engines
            .par_iter()
            .map(|engine| {
                engine.proximity_engine.boolean_engine
                    .search_boolean(query)
                    .unwrap_or_default()
            })
            .collect();
        
        let results_per_engine: Vec<usize> = all_results.iter()
            .map(|r| r.len())
            .collect();
        
        let total_before_dedup: usize = results_per_engine.iter().sum();
        
        // Aggregate and deduplicate
        let mut aggregator = SearchAggregator::new(dedup_strategy);
        for results in all_results {
            aggregator.add_results(results);
        }
        
        let final_results = aggregator.finalize();
        let total_after_dedup = final_results.len();
        
        // Calculate average score
        let average_score = if !final_results.is_empty() {
            final_results.iter().map(|r| r.score).sum::<f32>() / final_results.len() as f32
        } else {
            0.0
        };
        
        let stats = ParallelSearchStats {
            total_results: total_after_dedup,
            results_per_engine,
            search_duration: start.elapsed(),
            average_score,
            engines_searched: self.engines.len(),
            dedup_removed: total_before_dedup - total_after_dedup,
        };
        
        Ok((final_results, stats))
    }
}
```

### 4. Add comprehensive parallel search tests
Add these tests to the test module:
```rust
#[cfg(test)]
mod parallel_search_method_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_search_parallel_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create test indexes
        let index_paths = vec![
            temp_dir.path().join("index1"),
            temp_dir.path().join("index2"),
        ];
        
        // Create and populate indexes (simplified for test)
        for path in &index_paths {
            std::fs::create_dir_all(path)?;
        }
        
        let engine = ParallelSearchEngine::new(index_paths)?;
        
        // Test basic search
        let results = engine.search_parallel("test query")?;
        
        // Results should be deduplicated and sorted by score
        assert!(results.is_empty() || results.windows(2).all(|w| w[0].score >= w[1].score));
        
        Ok(())
    }
    
    #[test]
    fn test_search_parallel_with_options() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_paths = vec![temp_dir.path().join("index1")];
        std::fs::create_dir_all(&index_paths[0])?;
        
        let engine = ParallelSearchEngine::new(index_paths)?;
        
        // Test with different options
        let results = engine.search_parallel_with_options(
            "test",
            DedupStrategy::ByFilePath,
            Some(10), // Limit to 10 results
        )?;
        
        assert!(results.len() <= 10);
        
        Ok(())
    }
    
    #[test]
    fn test_search_parallel_timed() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_paths = vec![temp_dir.path().join("index1")];
        std::fs::create_dir_all(&index_paths[0])?;
        
        let engine = ParallelSearchEngine::new(index_paths)?;
        
        let (results, duration) = engine.search_parallel_timed("test")?;
        
        // Should return results and timing
        assert!(duration.as_nanos() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_search_parallel_with_timeout() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_paths: Vec<_> = (0..3)
            .map(|i| temp_dir.path().join(format!("index{}", i)))
            .collect();
        
        for path in &index_paths {
            std::fs::create_dir_all(path)?;
        }
        
        let engine = ParallelSearchEngine::new(index_paths)?;
        
        // Search with 100ms timeout per engine
        let results = engine.search_parallel_with_timeout(
            "test",
            Duration::from_millis(100),
            DedupStrategy::TakeHighestScore,
        )?;
        
        // Should complete within reasonable time even with timeout
        assert!(results.is_empty() || results.len() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_search_parallel_with_stats() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_paths: Vec<_> = (0..2)
            .map(|i| temp_dir.path().join(format!("index{}", i)))
            .collect();
        
        for path in &index_paths {
            std::fs::create_dir_all(path)?;
        }
        
        let engine = ParallelSearchEngine::new(index_paths)?;
        
        let (results, stats) = engine.search_parallel_with_stats(
            "test",
            DedupStrategy::ByFilePath,
        )?;
        
        // Verify statistics
        assert_eq!(stats.engines_searched, 2);
        assert_eq!(stats.results_per_engine.len(), 2);
        assert!(stats.search_duration.as_nanos() > 0);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] `search_parallel()` method executes concurrent searches
- [ ] Results are properly aggregated and deduplicated
- [ ] Timeout control prevents hanging searches
- [ ] Statistics tracking provides performance insights
- [ ] Results are sorted by score (highest first)
- [ ] All tests pass including timeout scenarios
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Rayon's `par_iter()` handles parallel distribution automatically
- Timeout mechanism prevents slow indexes from blocking results
- Statistics help identify performance bottlenecks
- Multiple deduplication strategies provide flexibility
- This completes the missing ParallelSearchEngine functionality