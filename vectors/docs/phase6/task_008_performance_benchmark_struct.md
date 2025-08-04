# Task 008: Create PerformanceBenchmark Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The PerformanceBenchmark measures search latency, throughput, and system resource usage under various load conditions.

## Project Structure
```
src/
  validation/
    performance.rs     <- Create this file
  lib.rs
```

## Task Description
Create the `PerformanceBenchmark` struct that measures search performance including latency, throughput, concurrent access, and resource usage.

## Requirements
1. Create `src/validation/performance.rs`
2. Implement `PerformanceBenchmark` struct
3. Create `PerformanceMetrics` struct for results
4. Add async constructor and basic methods
5. Set up timing and measurement infrastructure

## Expected Code Structure
```rust
use std::time::{Duration, Instant};
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

// Import the search system (adjust as needed)
use crate::{UnifiedSearchSystem, SearchMode};

pub struct PerformanceBenchmark {
    search_system: UnifiedSearchSystem,
    metrics: PerformanceMetrics,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latencies: Vec<Duration>,
    pub throughput_qps: f64,
    pub index_rate_fps: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub total_queries: usize,
    pub failed_queries: usize,
    pub average_latency_ms: f64,
}

impl PerformanceBenchmark {
    pub async fn new<P: AsRef<Path>>(
        text_index_path: P,
        vector_db_path: &str,
    ) -> Result<Self> {
        let search_system = UnifiedSearchSystem::new(text_index_path.as_ref(), vector_db_path)
            .await
            .context("Failed to initialize search system for performance testing")?;
        
        Ok(Self {
            search_system,
            metrics: PerformanceMetrics::default(),
        })
    }
    
    pub fn reset_metrics(&mut self) {
        self.metrics = PerformanceMetrics::default();
    }
    
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    pub async fn warmup(&mut self, warmup_queries: &[String]) -> Result<()> {
        println!("Warming up search system with {} queries...", warmup_queries.len());
        
        for query in warmup_queries {
            let _ = self.search_system.search_hybrid(query, SearchMode::Hybrid).await;
        }
        
        println!("Warmup completed");
        Ok(())
    }
    
    fn calculate_percentiles(&mut self) {
        if self.metrics.latencies.is_empty() {
            return;
        }
        
        let mut sorted_latencies = self.metrics.latencies.clone();
        sorted_latencies.sort();
        
        let len = sorted_latencies.len();
        self.metrics.p50_latency_ms = sorted_latencies[len / 2].as_millis() as u64;
        self.metrics.p95_latency_ms = sorted_latencies[(len * 95) / 100].as_millis() as u64;
        self.metrics.p99_latency_ms = sorted_latencies[(len * 99) / 100].as_millis() as u64;
        
        // Calculate average
        let total_ms: u64 = sorted_latencies.iter().map(|d| d.as_millis() as u64).sum();
        self.metrics.average_latency_ms = total_ms as f64 / len as f64;
    }
    
    pub fn print_summary(&self) {
        println!("\n=== Performance Benchmark Results ===");
        println!("Total queries: {}", self.metrics.total_queries);
        println!("Failed queries: {}", self.metrics.failed_queries);
        println!("Average latency: {:.2}ms", self.metrics.average_latency_ms);
        println!("P50 latency: {}ms", self.metrics.p50_latency_ms);
        println!("P95 latency: {}ms", self.metrics.p95_latency_ms);
        println!("P99 latency: {}ms", self.metrics.p99_latency_ms);
        println!("Throughput: {:.2} QPS", self.metrics.throughput_qps);
        println!("Memory usage: {:.2}MB", self.metrics.memory_usage_mb);
    }
}

impl PerformanceMetrics {
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            return 100.0;
        }
        ((self.total_queries - self.failed_queries) as f64 / self.total_queries as f64) * 100.0
    }
}
```

## Dependencies to Add
```toml
[dependencies]
rayon = "1.7"
sysinfo = "0.29"  # For system resource monitoring
```

## Success Criteria
- PerformanceBenchmark struct compiles without errors
- Metrics calculation works correctly
- Warmup functionality is implemented
- Percentile calculations are accurate
- Summary reporting is clear and useful
- Basic timing infrastructure is in place

## Time Limit
10 minutes maximum