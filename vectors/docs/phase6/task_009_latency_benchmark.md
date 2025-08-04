# Task 009: Implement Latency Benchmark Method

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 008. The latency benchmark measures single-query response times under normal conditions.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `run_latency_benchmark()` method that measures individual query response times and calculates latency percentiles.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. Implement latency measurement for individual queries
3. Calculate percentile statistics
4. Handle errors gracefully
5. Support multiple iterations per query

## Expected Code Structure to Add
```rust
impl PerformanceBenchmark {
    pub async fn run_latency_benchmark(
        &mut self, 
        queries: Vec<String>, 
        iterations: usize
    ) -> Result<()> {
        println!("Running latency benchmark with {} queries, {} iterations each...", 
                queries.len(), iterations);
        
        let mut all_latencies = Vec::new();
        let mut failed_count = 0;
        let mut total_queries = 0;
        
        for (i, query) in queries.iter().enumerate() {
            println!("Testing query {}/{}: {}", i + 1, queries.len(), 
                    if query.len() > 50 { &query[..50] } else { query });
            
            for iteration in 0..iterations {
                total_queries += 1;
                let start = Instant::now();
                
                match self.search_system.search_hybrid(query, SearchMode::Hybrid).await {
                    Ok(_results) => {
                        let latency = start.elapsed();
                        all_latencies.push(latency);
                        
                        // Print progress for long benchmarks
                        if iteration % 10 == 0 && iterations > 20 {
                            println!("  Iteration {}/{}, current latency: {}ms", 
                                    iteration + 1, iterations, latency.as_millis());
                        }
                    }
                    Err(e) => {
                        failed_count += 1;
                        println!("  Query failed on iteration {}: {}", iteration + 1, e);
                    }
                }
            }
        }
        
        // Update metrics
        self.metrics.latencies = all_latencies;
        self.metrics.total_queries = total_queries;
        self.metrics.failed_queries = failed_count;
        
        // Calculate percentiles and averages
        self.calculate_percentiles();
        
        println!("Latency benchmark completed:");
        println!("  Total queries: {}", total_queries);
        println!("  Failed queries: {}", failed_count);
        println!("  Success rate: {:.2}%", self.metrics.success_rate());
        
        Ok(())
    }
    
    pub async fn run_single_query_benchmark(
        &mut self,
        query: &str,
        iterations: usize,
        search_mode: SearchMode,
    ) -> Result<QueryBenchmarkResult> {
        let mut latencies = Vec::new();
        let mut failed_count = 0;
        let mut result_counts = Vec::new();
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            match self.search_system.search_hybrid(query, search_mode.clone()).await {
                Ok(results) => {
                    let latency = start.elapsed();
                    latencies.push(latency);
                    result_counts.push(results.len());
                }
                Err(_) => {
                    failed_count += 1;
                }
            }
        }
        
        // Calculate statistics
        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort();
        
        let result = QueryBenchmarkResult {
            query: query.to_string(),
            total_iterations: iterations,
            successful_iterations: latencies.len(),
            failed_iterations: failed_count,
            min_latency_ms: sorted_latencies.first().map(|d| d.as_millis() as u64).unwrap_or(0),
            max_latency_ms: sorted_latencies.last().map(|d| d.as_millis() as u64).unwrap_or(0),
            avg_latency_ms: if !latencies.is_empty() {
                latencies.iter().map(|d| d.as_millis() as u64).sum::<u64>() as f64 / latencies.len() as f64
            } else {
                0.0
            },
            p50_latency_ms: if !sorted_latencies.is_empty() {
                sorted_latencies[sorted_latencies.len() / 2].as_millis() as u64
            } else {
                0
            },
            avg_result_count: if !result_counts.is_empty() {
                result_counts.iter().sum::<usize>() as f64 / result_counts.len() as f64
            } else {
                0.0
            },
        };
        
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBenchmarkResult {
    pub query: String,
    pub total_iterations: usize,
    pub successful_iterations: usize,
    pub failed_iterations: usize,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub avg_result_count: f64,
}
```

## Success Criteria
- Latency benchmark method works correctly
- Percentile calculations are accurate
- Error handling doesn't crash the benchmark
- Progress reporting is helpful for long tests
- Single query benchmark provides detailed statistics
- Results are properly stored in metrics

## Time Limit
10 minutes maximum