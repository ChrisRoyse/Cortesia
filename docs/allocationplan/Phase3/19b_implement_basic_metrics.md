# Task 19b: Implement Basic Metrics

**Estimated Time**: 5 minutes  
**Dependencies**: 19a  
**Stage**: Performance Monitoring  

## Objective
Implement basic metrics collection for memory allocation and search operations.

## Implementation Steps

1. Create `src/monitoring/metrics.rs`:
```rust
use prometheus::{Counter, Histogram, Gauge, Registry};
use std::sync::Arc;

pub struct BasicMetrics {
    // Memory allocation metrics
    pub memory_allocations_total: Counter,
    pub memory_allocation_duration: Histogram,
    pub memory_usage_bytes: Gauge,
    
    // Search operation metrics
    pub search_operations_total: Counter,
    pub search_duration: Histogram,
    pub search_results_count: Histogram,
    
    // System metrics
    pub cpu_usage_percent: Gauge,
    pub active_connections: Gauge,
    pub cache_hit_rate: Gauge,
}

impl BasicMetrics {
    pub fn new(registry: &Registry) -> prometheus::Result<Self> {
        let memory_allocations_total = Counter::new(
            "memory_allocations_total",
            "Total number of memory allocations"
        )?;
        registry.register(Box::new(memory_allocations_total.clone()))?;
        
        let memory_allocation_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "memory_allocation_duration_seconds",
                "Duration of memory allocation operations"
            ).buckets(vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0])
        )?;
        registry.register(Box::new(memory_allocation_duration.clone()))?;
        
        let memory_usage_bytes = Gauge::new(
            "memory_usage_bytes",
            "Current memory usage in bytes"
        )?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;
        
        let search_operations_total = Counter::new(
            "search_operations_total",
            "Total number of search operations"
        )?;
        registry.register(Box::new(search_operations_total.clone()))?;
        
        let search_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "search_duration_seconds",
                "Duration of search operations"
            ).buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
        )?;
        registry.register(Box::new(search_duration.clone()))?;
        
        let search_results_count = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "search_results_count",
                "Number of results returned by search operations"
            ).buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
        )?;
        registry.register(Box::new(search_results_count.clone()))?;
        
        let cpu_usage_percent = Gauge::new(
            "cpu_usage_percent",
            "Current CPU usage percentage"
        )?;
        registry.register(Box::new(cpu_usage_percent.clone()))?;
        
        let active_connections = Gauge::new(
            "active_connections",
            "Number of active database connections"
        )?;
        registry.register(Box::new(active_connections.clone()))?;
        
        let cache_hit_rate = Gauge::new(
            "cache_hit_rate",
            "Cache hit rate percentage"
        )?;
        registry.register(Box::new(cache_hit_rate.clone()))?;
        
        Ok(Self {
            memory_allocations_total,
            memory_allocation_duration,
            memory_usage_bytes,
            search_operations_total,
            search_duration,
            search_results_count,
            cpu_usage_percent,
            active_connections,
            cache_hit_rate,
        })
    }
    
    pub fn record_memory_allocation(&self, duration_seconds: f64) {
        self.memory_allocations_total.inc();
        self.memory_allocation_duration.observe(duration_seconds);
    }
    
    pub fn record_search_operation(&self, duration_seconds: f64, result_count: usize) {
        self.search_operations_total.inc();
        self.search_duration.observe(duration_seconds);
        self.search_results_count.observe(result_count as f64);
    }
    
    pub fn update_system_metrics(&self, cpu_percent: f64, memory_bytes: f64, connections: i64) {
        self.cpu_usage_percent.set(cpu_percent);
        self.memory_usage_bytes.set(memory_bytes);
        self.active_connections.set(connections as f64);
    }
    
    pub fn update_cache_hit_rate(&self, hit_rate_percent: f64) {
        self.cache_hit_rate.set(hit_rate_percent);
    }
}
```

## Acceptance Criteria
- [ ] Basic metrics implemented
- [ ] Prometheus metrics properly registered
- [ ] Helper methods for recording metrics

## Success Metrics
- Metrics collection works without errors
- Prometheus metrics are properly formatted
- Performance impact minimal

## Next Task
19c_implement_health_checks.md