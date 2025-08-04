# Task 19b: Add Metrics Registry
**Time**: 3 minutes (45 sec read, 1.5 min implement, 45 sec verify)
**Dependencies**: 19a
**Stage**: Performance Foundation

## Objective
Add prometheus metrics registry to the PerformanceMonitor.

## Implementation
Update `src/inheritance/monitoring/performance_monitor.rs`:
```rust
use prometheus::Registry;

#[derive(Debug)]
pub struct PerformanceMonitor {
    pub config: MonitoringConfig,
    pub metrics_registry: Arc<Registry>,
    pub is_active: bool,
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics_registry: Arc::new(Registry::new()),
            is_active: false,
        }
    }
    
    pub fn get_registry(&self) -> Arc<Registry> {
        self.metrics_registry.clone()
    }
}
```

## Success Criteria
- Registry added to struct
- Registry accessible via getter
- Compiles without errors

**Next**: 19c