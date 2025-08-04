# Task 98: Production Monitoring and Telemetry Setup

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 97  
**Dependencies:** Integration verification complete

## Objective
Implement production-grade monitoring, metrics collection, and telemetry for the Phase 1 system.

## Context
You're adding observability to the vector search system. The system is functionally complete but needs monitoring to track performance, usage patterns, and potential issues in production. This includes metrics for indexing speed, search latency, memory usage, and error rates.

## Task Details

### What You Need to Do

1. **Add monitoring dependencies to Cargo.toml**:
```toml
prometheus = "0.13"
tracing = "0.1"
tracing-subscriber = "0.3"
metrics = "0.23"
metrics-exporter-prometheus = "0.14"
```

2. **Create monitoring module** (`src/monitoring.rs`):
```rust
use metrics::{counter, gauge, histogram, describe_counter, describe_gauge, describe_histogram};
use std::time::Instant;
use tracing::{info, warn, error, instrument};

pub struct MetricsCollector {
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::register_metrics();
        Self { start_time: Instant::now() }
    }
    
    fn register_metrics() {
        describe_counter!("tantivy_documents_indexed", "Total documents indexed");
        describe_counter!("tantivy_search_queries", "Total search queries executed");
        describe_counter!("tantivy_errors", "Total errors encountered");
        
        describe_histogram!("tantivy_indexing_duration_ms", "Document indexing duration");
        describe_histogram!("tantivy_search_duration_ms", "Search query duration");
        describe_histogram!("tantivy_chunk_size_bytes", "Document chunk sizes");
        
        describe_gauge!("tantivy_index_size_bytes", "Current index size on disk");
        describe_gauge!("tantivy_memory_usage_bytes", "Current memory usage");
        describe_gauge!("tantivy_active_searches", "Currently active searches");
    }
    
    #[instrument]
    pub fn record_indexing(&self, file_path: &str, duration_ms: f64, success: bool) {
        if success {
            counter!("tantivy_documents_indexed", "status" => "success").increment(1);
            histogram!("tantivy_indexing_duration_ms").record(duration_ms);
            info!(file_path, duration_ms, "Document indexed successfully");
        } else {
            counter!("tantivy_documents_indexed", "status" => "failed").increment(1);
            counter!("tantivy_errors", "type" => "indexing").increment(1);
            warn!(file_path, "Document indexing failed");
        }
    }
    
    #[instrument]
    pub fn record_search(&self, query: &str, duration_ms: f64, result_count: usize) {
        counter!("tantivy_search_queries").increment(1);
        histogram!("tantivy_search_duration_ms").record(duration_ms);
        
        info!(
            query, 
            duration_ms, 
            result_count,
            "Search completed"
        );
    }
    
    pub fn record_chunk(&self, size_bytes: usize) {
        histogram!("tantivy_chunk_size_bytes").record(size_bytes as f64);
    }
    
    pub fn update_index_size(&self, size_bytes: u64) {
        gauge!("tantivy_index_size_bytes").set(size_bytes as f64);
    }
    
    pub fn update_memory_usage(&self, bytes: usize) {
        gauge!("tantivy_memory_usage_bytes").set(bytes as f64);
    }
    
    pub fn track_active_search(&self, active: bool) {
        if active {
            gauge!("tantivy_active_searches").increment(1.0);
        } else {
            gauge!("tantivy_active_searches").decrement(1.0);
        }
    }
}

// Integrate with DocumentIndexer
pub trait MonitoredIndexer {
    fn index_file_monitored(&mut self, file_path: &Path, metrics: &MetricsCollector) -> Result<()>;
}

impl MonitoredIndexer for DocumentIndexer {
    fn index_file_monitored(&mut self, file_path: &Path, metrics: &MetricsCollector) -> Result<()> {
        let start = Instant::now();
        
        let result = self.index_file(file_path);
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        metrics.record_indexing(
            &file_path.to_string_lossy(),
            duration_ms,
            result.is_ok()
        );
        
        // Track chunk sizes
        if result.is_ok() {
            let content = std::fs::read_to_string(file_path)?;
            let chunks = self.chunker.chunk_code_file(&content, &self.detect_language(file_path));
            for chunk in chunks {
                metrics.record_chunk(chunk.content.len());
            }
        }
        
        result
    }
}

// Integrate with SearchEngine
pub trait MonitoredSearch {
    fn search_monitored(&self, query: &str, metrics: &MetricsCollector) -> Result<Vec<SearchResult>>;
}

impl MonitoredSearch for SearchEngine {
    fn search_monitored(&self, query: &str, metrics: &MetricsCollector) -> Result<Vec<SearchResult>> {
        metrics.track_active_search(true);
        let start = Instant::now();
        
        let results = self.search(query);
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        if let Ok(ref res) = results {
            metrics.record_search(query, duration_ms, res.len());
        }
        
        metrics.track_active_search(false);
        results
    }
}
```

3. **Add metrics endpoint setup**:
```rust
pub fn setup_metrics_endpoint() -> Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use std::net::SocketAddr;
    
    let addr: SocketAddr = "127.0.0.1:9090".parse()?;
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()?;
    
    info!("Metrics endpoint available at http://{}/metrics", addr);
    Ok(())
}
```

4. **Create test for monitoring**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collection() -> Result<()> {
        let metrics = MetricsCollector::new();
        
        // Test indexing metrics
        metrics.record_indexing("test.rs", 15.5, true);
        metrics.record_indexing("fail.rs", 5.2, false);
        
        // Test search metrics
        metrics.record_search("Result<T, E>", 2.3, 5);
        
        // Test gauges
        metrics.update_index_size(1024 * 1024); // 1MB
        metrics.update_memory_usage(50 * 1024 * 1024); // 50MB
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] All metrics are properly collected and exposed
- [ ] Prometheus endpoint serves metrics at :9090/metrics
- [ ] Tracing logs include structured context
- [ ] No performance overhead > 5% from monitoring
- [ ] Metrics survive system restart
- [ ] Error tracking captures all failure modes

## Common Pitfalls to Avoid
- Don't create unbounded cardinality in metric labels
- Avoid blocking operations in metric collection
- Ensure metrics are thread-safe
- Don't log sensitive data in traces
- Handle metric endpoint failures gracefully

## Context for Next Task
Task 99 will implement health checks and readiness probes for deployment.