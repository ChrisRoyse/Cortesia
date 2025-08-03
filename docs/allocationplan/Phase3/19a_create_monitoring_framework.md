# Task 19a: Create Monitoring Framework

**Estimated Time**: 5 minutes  
**Dependencies**: 18i  
**Stage**: Performance Monitoring  

## Objective
Create basic performance monitoring infrastructure and setup.

## Implementation Steps

1. Add monitoring dependencies to `Cargo.toml`:
```toml
[dependencies]
prometheus = "0.13"
tokio-metrics = "0.3"
sysinfo = "0.30"
chrono = { version = "0.4", features = ["serde"] }
serde_json = "1.0"
```

2. Create `src/monitoring/mod.rs`:
```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use prometheus::{Registry, Counter, Histogram, Gauge};

pub mod metrics;
pub mod alerts;
pub mod health;

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub collection_interval_ms: u64,
    pub alert_thresholds: AlertThresholds,
    pub enable_detailed_metrics: bool,
    pub enable_health_checks: bool,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub memory_usage_percent: f64,
    pub cpu_usage_percent: f64,
    pub response_time_ms: f64,
    pub error_rate_percent: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 5000, // 5 seconds
            alert_thresholds: AlertThresholds {
                memory_usage_percent: 80.0,
                cpu_usage_percent: 70.0,
                response_time_ms: 1000.0,
                error_rate_percent: 5.0,
            },
            enable_detailed_metrics: true,
            enable_health_checks: true,
        }
    }
}

pub struct PerformanceMonitor {
    registry: Arc<Registry>,
    config: MonitoringConfig,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        let registry = Arc::new(Registry::new());
        
        Self {
            registry,
            config,
            start_time: Instant::now(),
        }
    }
    
    pub fn get_registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
    
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}
```

## Acceptance Criteria
- [ ] Monitoring framework created
- [ ] Dependencies added
- [ ] Basic configuration structure ready

## Success Metrics
- Framework initializes successfully
- Configuration is properly structured
- Ready for metric collection

## Next Task
19b_implement_basic_metrics.md