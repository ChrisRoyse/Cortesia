# Task 19a: Create Basic Monitoring Struct
**Time**: 4 minutes (1 min read, 2 min implement, 1 min verify)
**Dependencies**: None
**Stage**: Performance Foundation

## Objective
Create the basic PerformanceMonitor struct with essential fields.

## Implementation
Create file `src/inheritance/monitoring/performance_monitor.rs`:
```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug)]
pub struct PerformanceMonitor {
    pub config: MonitoringConfig,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub collection_interval_ms: u64,
    pub max_metrics_history: usize,
    pub enable_real_time_alerts: bool,
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            is_active: false,
        }
    }
    
    pub fn start(&mut self) {
        self.is_active = true;
    }
    
    pub fn stop(&mut self) {
        self.is_active = false;
    }
}
```

## Success Criteria
- File created and compiles
- Basic struct with config
- Start/stop methods functional

**Next**: 19b