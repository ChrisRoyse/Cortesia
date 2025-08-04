# Task 120: Add Default Monitoring Configurations

## Prerequisites Check
- [ ] Task 119 completed: MonitoringConfig structure implemented
- [ ] Monitoring configuration is functional
- [ ] Run: `cargo check` (should pass)

## Context
Add sensible default configurations for production monitoring setup.

## Task Objective
Implement Default trait for MonitoringConfig with production-ready thresholds.

## Steps
1. Add default configurations:
   ```rust
   impl Default for MonitoringConfig {
       fn default() -> Self {
           let mut thresholds = HashMap::new();
           thresholds.insert(MetricType::QueryResponseTime, 1000.0); // 1 second
           thresholds.insert(MetricType::CacheHitRatio, 0.8); // 80%
           thresholds.insert(MetricType::ErrorRate, 0.05); // 5%
           thresholds.insert(MetricType::SearchAccuracy, 0.9); // 90%
           
           Self {
               enabled: true,
               collection_interval: 10, // 10 seconds
               retention_period: 24 * 3600, // 24 hours
               max_metrics_per_type: 1000,
               enable_alerting: true,
               thresholds,
           }
       }
   }
   ```
2. Verify compilation

## Success Criteria
- [ ] Default implementation with reasonable production values
- [ ] Performance thresholds for key metrics
- [ ] Sensible collection and retention intervals
- [ ] Alerting enabled with appropriate limits
- [ ] Ready for production deployment
- [ ] Compiles without errors

## Time: 2 minutes

## Next Task
Task 121 will implement the core PerformanceMonitor struct.

## Notes
Default configuration provides immediate production-ready monitoring setup with industry-standard thresholds.