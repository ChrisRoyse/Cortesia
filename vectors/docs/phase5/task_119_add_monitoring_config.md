# Task 119: Add MonitoringConfig Setup

## Prerequisites Check
- [ ] Task 118 completed: MetricStats aggregation implemented
- [ ] All metric structures are functional
- [ ] Run: `cargo check` (should pass)

## Context
Add comprehensive monitoring configuration with thresholds and settings.

## Task Objective
Implement MonitoringConfig struct with performance thresholds and collection settings.

## Steps
1. Add monitoring configuration:
   ```rust
   /// Performance monitoring configuration
   #[derive(Debug, Clone)]
   pub struct MonitoringConfig {
       /// Enable performance monitoring
       pub enabled: bool,
       /// Metric collection interval in seconds
       pub collection_interval: u64,
       /// Metric retention period in seconds
       pub retention_period: u64,
       /// Maximum metrics to store per type
       pub max_metrics_per_type: usize,
       /// Enable real-time alerting
       pub enable_alerting: bool,
       /// Performance thresholds for alerting
       pub thresholds: HashMap<MetricType, f64>,
   }
   ```
2. Verify compilation

## Success Criteria
- [ ] MonitoringConfig structure with all configuration options
- [ ] Collection interval and retention settings
- [ ] Maximum metrics limit per type
- [ ] Alerting configuration with thresholds
- [ ] Threshold mapping for different metric types
- [ ] Compiles without errors

## Time: 3 minutes

## Next Task
Task 120 will add default configurations.

## Notes
MonitoringConfig provides comprehensive control over performance monitoring behavior and alerting.