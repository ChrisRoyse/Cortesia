# Task 020: Implement Throughput Benchmark Method

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 008-009. The throughput benchmark measures queries per second (QPS) under sustained load conditions.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `run_throughput_benchmark()` method that measures sustained query throughput over a specified time period to determine maximum QPS capacity.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. Measure queries per second over time duration
3. Handle query failures gracefully
4. Track resource usage during sustained load
5. Support different query patterns

## Expected Code Structure to Add
```rust
impl PerformanceBenchmark {
    pub async fn run_throughput_benchmark(
        &mut self,
        queries: Vec<String>,
        duration_secs: u64,
    ) -> Result<ThroughputResult> {
        println!("Running throughput benchmark for {} seconds with {} unique queries...", 
                duration_secs, queries.len());
        
        if queries.is_empty() {
            return Err(anyhow::anyhow!("No queries provided for throughput benchmark"));
        }
        
        let start_time = Instant::now();
        let mut query_count = 0;
        let mut failed_count = 0;
        let mut latencies = Vec::new();
        let mut query_index = 0;
        
        // Track resource usage
        let mut peak_memory_mb = 0.0;
        let mut resource_samples = Vec::new();
        
        while start_time.elapsed().as_secs() < duration_secs {
            let query = &queries[query_index % queries.len()];
            query_index += 1;
            
            let query_start = Instant::now();
            
            match self.search_system.search_hybrid(query, SearchMode::Hybrid).await {
                Ok(_results) => {
                    let latency = query_start.elapsed();
                    latencies.push(latency);
                    query_count += 1;
                }
                Err(e) => {
                    failed_count += 1;
                    println!("Query failed during throughput test: {}", e);
                }
            }
            
            // Sample resource usage every 100 queries
            if query_count % 100 == 0 {
                let memory_usage = self.get_memory_usage_mb();
                if memory_usage > peak_memory_mb {
                    peak_memory_mb = memory_usage;
                }
                resource_samples.push(ResourceSample {
                    timestamp: start_time.elapsed(),
                    memory_mb: memory_usage,
                    queries_completed: query_count,
                });
            }
            
            // Break if we've exceeded the duration
            if start_time.elapsed().as_secs() >= duration_secs {
                break;
            }
        }
        
        let total_duration = start_time.elapsed();
        let qps = query_count as f64 / total_duration.as_secs_f64();
        
        // Update metrics
        self.metrics.throughput_qps = qps;
        self.metrics.total_queries = query_count;
        self.metrics.failed_queries = failed_count;
        self.metrics.memory_usage_mb = peak_memory_mb;
        
        // Calculate latency statistics for throughput test
        if !latencies.is_empty() {
            let mut sorted_latencies = latencies.clone();
            sorted_latencies.sort();
            
            self.metrics.average_latency_ms = latencies.iter()
                .map(|d| d.as_millis() as f64)
                .sum::<f64>() / latencies.len() as f64;
        }
        
        let result = ThroughputResult {
            queries_per_second: qps,
            total_queries: query_count,
            failed_queries: failed_count,
            success_rate: ((query_count - failed_count) as f64 / query_count as f64) * 100.0,
            duration_seconds: total_duration.as_secs_f64(),
            peak_memory_mb,
            resource_samples,
            average_latency_ms: self.metrics.average_latency_ms,
        };
        
        println!("Throughput benchmark completed:");
        println!("  Queries per second: {:.2}", result.queries_per_second);
        println!("  Total queries: {}", result.total_queries);
        println!("  Success rate: {:.2}%", result.success_rate);
        println!("  Peak memory: {:.2}MB", result.peak_memory_mb);
        
        Ok(result)
    }
    
    fn get_memory_usage_mb(&self) -> f64 {
        // Implementation depends on system monitoring
        // For now, return a placeholder that can be implemented with sysinfo
        use sysinfo::{System, SystemExt, ProcessExt};
        
        let mut system = System::new_all();
        system.refresh_all();
        
        if let Some(process) = system.process(sysinfo::get_current_pid().unwrap()) {
            process.memory() as f64 / 1_048_576.0 // Convert bytes to MB
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputResult {
    pub queries_per_second: f64,
    pub total_queries: usize,
    pub failed_queries: usize,
    pub success_rate: f64,
    pub duration_seconds: f64,
    pub peak_memory_mb: f64,
    pub resource_samples: Vec<ResourceSample>,
    pub average_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    pub timestamp: Duration,
    pub memory_mb: f64,
    pub queries_completed: usize,
}
```

## Success Criteria
- Throughput benchmark runs for specified duration
- QPS calculation is accurate
- Resource usage tracking works
- Failed queries don't stop the benchmark
- Results provide comprehensive throughput metrics
- Memory usage monitoring functions correctly

## Time Limit
10 minutes maximum