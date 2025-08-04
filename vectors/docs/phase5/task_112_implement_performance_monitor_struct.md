# Task 117: Implement PerformanceMonitor Struct

## Prerequisites Check
- [ ] Task 116 completed: performance monitoring foundation created
- [ ] MetricData and MonitoringConfig are defined
- [ ] Run: `cargo check` (should pass)

## Context
Implement the core PerformanceMonitor struct that collects, stores, and analyzes performance metrics.

## Task Objective
Create the main PerformanceMonitor struct with metric collection and analysis capabilities.

## Steps
1. Add PerformanceMonitor struct:
   ```rust
   /// Performance monitoring system
   pub struct PerformanceMonitor {
       /// Configuration
       config: MonitoringConfig,
       /// Metric storage by type and component
       metrics: Arc<RwLock<HashMap<(MetricType, String), Vec<MetricData>>>>,
       /// Cached statistics
       cached_stats: Arc<RwLock<HashMap<(MetricType, String), MetricStats>>>,
       /// Last cleanup timestamp
       last_cleanup: Arc<RwLock<Instant>>,
       /// Alert history
       alert_history: Arc<RwLock<Vec<PerformanceAlert>>>,
   }
   
   /// Performance alert
   #[derive(Debug, Clone)]
   pub struct PerformanceAlert {
       /// Alert ID
       pub id: String,
       /// Metric that triggered alert
       pub metric_type: MetricType,
       /// Component name
       pub component: String,
       /// Threshold value
       pub threshold: f64,
       /// Actual value that triggered alert
       pub actual_value: f64,
       /// Alert timestamp
       pub timestamp: Instant,
       /// Alert message
       pub message: String,
   }
   ```
2. Add constructor and basic methods:
   ```rust
   impl PerformanceMonitor {
       /// Create new performance monitor
       pub fn new(config: MonitoringConfig) -> Self {
           Self {
               config,
               metrics: Arc::new(RwLock::new(HashMap::new())),
               cached_stats: Arc::new(RwLock::new(HashMap::new())),
               last_cleanup: Arc::new(RwLock::new(Instant::now())),
               alert_history: Arc::new(RwLock::new(Vec::new())),
           }
       }
       
       /// Get current configuration
       pub fn config(&self) -> &MonitoringConfig {
           &self.config
       }
       
       /// Update configuration
       pub fn update_config(&mut self, config: MonitoringConfig) {
           self.config = config;
       }
       
       /// Check if monitoring is enabled
       pub fn is_enabled(&self) -> bool {
           self.config.enabled
       }
   }
   ```
3. Add metric collection methods:
   ```rust
   impl PerformanceMonitor {
       /// Record a performance metric
       pub async fn record_metric(
           &self,
           metric_type: MetricType,
           value: f64,
           component: String,
           labels: Option<HashMap<String, String>>,
       ) {
           if !self.config.enabled {
               return;
           }
           
           let mut metric = MetricData::new(metric_type.clone(), value, component.clone());
           if let Some(labels) = labels {
               for (key, val) in labels {
                   metric = metric.with_label(key, val);
               }
           }
           
           // Store metric
           {
               let mut metrics = self.metrics.write().await;
               let key = (metric_type.clone(), component.clone());
               let entry = metrics.entry(key).or_insert_with(Vec::new);
               entry.push(metric.clone());
               
               // Limit storage per type/component
               if entry.len() > self.config.max_metrics_per_type {
                   entry.remove(0);
               }
           }
           
           // Check for alerts
           if self.config.enable_alerting {
               self.check_alert_threshold(&metric_type, &component, value).await;
           }
           
           // Invalidate cached stats
           {
               let mut cached = self.cached_stats.write().await;
               cached.remove(&(metric_type, component));
           }
       }
       
       /// Record query response time
       pub async fn record_query_time(&self, component: &str, duration: Duration, query_type: &str) {
           let labels = HashMap::from([
               ("query_type".to_string(), query_type.to_string()),
           ]);
           
           self.record_metric(
               MetricType::QueryResponseTime,
               duration.as_millis() as f64,
               component.to_string(),
               Some(labels),
           ).await;
       }
       
       /// Record search accuracy
       pub async fn record_search_accuracy(&self, component: &str, accuracy: f64, search_mode: &str) {
           let labels = HashMap::from([
               ("search_mode".to_string(), search_mode.to_string()),
           ]);
           
           self.record_metric(
               MetricType::SearchAccuracy,
               accuracy,
               component.to_string(),
               Some(labels),
           ).await;
       }
       
       /// Record cache hit ratio
       pub async fn record_cache_hit_ratio(&self, component: &str, hit_ratio: f64) {
           self.record_metric(
               MetricType::CacheHitRatio,
               hit_ratio,
               component.to_string(),
               None,
           ).await;
       }
   }
   ```
4. Add metric analysis methods:
   ```rust
   impl PerformanceMonitor {
       /// Get statistics for a metric type and component
       pub async fn get_stats(&self, metric_type: &MetricType, component: &str) -> Option<MetricStats> {
           let key = (metric_type.clone(), component.to_string());
           
           // Check cache first
           {
               let cached = self.cached_stats.read().await;
               if let Some(stats) = cached.get(&key) {
                   return Some(stats.clone());
               }
           }
           
           // Calculate stats
           let metrics = self.metrics.read().await;
           if let Some(metric_list) = metrics.get(&key) {
               if metric_list.is_empty() {
                   return None;
               }
               
               let values: Vec<f64> = metric_list.iter().map(|m| m.value).collect();
               let stats = self.calculate_stats(metric_type.clone(), component.to_string(), &values);
               
               // Cache the result
               {
                   let mut cached = self.cached_stats.write().await;
                   cached.insert(key, stats.clone());
               }
               
               Some(stats)
           } else {
               None
           }
       }
       
       /// Calculate statistics from values
       fn calculate_stats(&self, metric_type: MetricType, component: String, values: &[f64]) -> MetricStats {
           let mut sorted_values = values.to_vec();
           sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
           
           let count = sorted_values.len();
           let min = sorted_values[0];
           let max = sorted_values[count - 1];
           let avg = sorted_values.iter().sum::<f64>() / count as f64;
           
           let p50 = self.percentile(&sorted_values, 50.0);
           let p95 = self.percentile(&sorted_values, 95.0);
           let p99 = self.percentile(&sorted_values, 99.0);
           
           MetricStats {
               metric_type,
               component,
               count,
               min,
               max,
               avg,
               p50,
               p95,
               p99,
               window_start: Instant::now() - Duration::from_secs(self.config.retention_period),
               window_end: Instant::now(),
           }
       }
       
       /// Calculate percentile from sorted values
       fn percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
           if sorted_values.is_empty() {
               return 0.0;
           }
           
           let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64) as usize;
           sorted_values[index.min(sorted_values.len() - 1)]
       }
   }
   ```
5. Add alerting methods:
   ```rust
   impl PerformanceMonitor {
       /// Check if metric value exceeds alert threshold
       async fn check_alert_threshold(&self, metric_type: &MetricType, component: &str, value: f64) {
           if let Some(threshold) = self.config.thresholds.get(metric_type) {
               let should_alert = match metric_type {
                   MetricType::QueryResponseTime | MetricType::ErrorRate | MetricType::MemoryUsage => {
                       value > *threshold
                   }
                   MetricType::CacheHitRatio | MetricType::SearchAccuracy => {
                       value < *threshold
                   }
                   _ => false,
               };
               
               if should_alert {
                   self.create_alert(metric_type.clone(), component.to_string(), *threshold, value).await;
               }
           }
       }
       
       /// Create performance alert
       async fn create_alert(&self, metric_type: MetricType, component: String, threshold: f64, actual_value: f64) {
           let alert = PerformanceAlert {
               id: Uuid::new_v4().to_string(),
               metric_type: metric_type.clone(),
               component: component.clone(),
               threshold,
               actual_value,
               timestamp: Instant::now(),
               message: format!(
                   "{:?} for {} exceeded threshold: {:.2} > {:.2}",
                   metric_type, component, actual_value, threshold
               ),
           };
           
           let mut alerts = self.alert_history.write().await;
           alerts.push(alert);
           
           // Keep only recent alerts (last 100)
           if alerts.len() > 100 {
               alerts.remove(0);
           }
       }
       
       /// Get recent alerts
       pub async fn get_recent_alerts(&self, limit: Option<usize>) -> Vec<PerformanceAlert> {
           let alerts = self.alert_history.read().await;
           let limit = limit.unwrap_or(10);
           alerts.iter()
               .rev()
               .take(limit)
               .cloned()
               .collect()
       }
   }
   ```
6. Verify compilation

## Success Criteria
- [ ] PerformanceMonitor with metric collection and storage
- [ ] Metric recording methods for common performance indicators
- [ ] Statistics calculation with percentiles
- [ ] Alert threshold checking and notification
- [ ] Cached statistics for performance optimization
- [ ] Recent alert history tracking
- [ ] Configuration-based enabling/disabling
- [ ] Memory limits for metric storage
- [ ] Compiles without errors

## Time: 7 minutes

## Next Task
Task 118 will implement system health dashboard and reporting.

## Notes
PerformanceMonitor provides comprehensive metric collection with statistical analysis and intelligent alerting based on configurable thresholds.