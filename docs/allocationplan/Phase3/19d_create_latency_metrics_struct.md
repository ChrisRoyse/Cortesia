# Task 19d: Create Latency Metrics Struct
**Time**: 4 minutes
**Dependencies**: 19c
**Stage**: Performance Foundation

## Objective
Create struct to track latency percentiles and timing metrics.

## Implementation
Add to `src/inheritance/monitoring/performance_monitor.rs`:
```rust
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub mean: Duration,
    pub max: Duration,
    pub min: Duration,
    pub sample_count: u64,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            p50: Duration::ZERO,
            p95: Duration::ZERO,
            p99: Duration::ZERO,
            mean: Duration::ZERO,
            max: Duration::ZERO,
            min: Duration::MAX,
            sample_count: 0,
        }
    }
}

impl LatencyMetrics {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_sample(&mut self, duration: Duration) {
        self.sample_count += 1;
        if duration > self.max {
            self.max = duration;
        }
        if duration < self.min {
            self.min = duration;
        }
        // Simplified mean calculation - real implementation would use proper percentile calculation
        let total_nanos = self.mean.as_nanos() * (self.sample_count - 1) as u128 + duration.as_nanos();
        self.mean = Duration::from_nanos((total_nanos / self.sample_count as u128) as u64);
    }
}
```

## Success Criteria
- LatencyMetrics struct created
- Default implementation
- Sample recording functionality
- Compiles and works correctly

**Next**: 19e