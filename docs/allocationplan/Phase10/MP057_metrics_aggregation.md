# MP057: Metrics Aggregation

## Task Description
Implement comprehensive metrics aggregation system for collecting, processing, and analyzing performance data from graph algorithms and neuromorphic components.

## Prerequisites
- MP001-MP050 completed
- Understanding of time-series data management
- Knowledge of statistical aggregation methods
- Familiarity with metrics visualization principles

## Detailed Steps

1. Create `src/neuromorphic/metrics/aggregation.rs`

2. Implement metrics collection and time-series data management:
   ```rust
   use chrono::{DateTime, Utc, Duration};
   use serde::{Serialize, Deserialize};
   use uuid::Uuid;
   use std::collections::{HashMap, BTreeMap};
   use std::sync::Arc;
   use tokio::sync::RwLock;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct MetricPoint {
       pub timestamp: DateTime<Utc>,
       pub value: f64,
       pub tags: HashMap<String, String>,
       pub metadata: Option<serde_json::Value>,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct TimeSeries {
       pub metric_name: String,
       pub points: BTreeMap<DateTime<Utc>, MetricPoint>,
       pub retention_period: Duration,
       pub aggregation_intervals: Vec<Duration>,
   }
   
   #[derive(Debug, Clone)]
   pub struct MetricsAggregator {
       time_series: Arc<RwLock<HashMap<String, TimeSeries>>>,
       aggregated_data: Arc<RwLock<HashMap<String, AggregatedMetrics>>>,
       collection_config: MetricsConfig,
       storage_backend: Arc<dyn MetricsStorage>,
   }
   
   #[derive(Debug, Clone)]
   pub struct MetricsConfig {
       pub collection_interval: Duration,
       pub retention_period: Duration,
       pub max_points_per_series: usize,
       pub aggregation_intervals: Vec<Duration>, // 1m, 5m, 1h, 1d
       pub compression_threshold: usize,
   }
   
   impl MetricsAggregator {
       pub fn new(config: MetricsConfig, storage: Arc<dyn MetricsStorage>) -> Self {
           Self {
               time_series: Arc::new(RwLock::new(HashMap::new())),
               aggregated_data: Arc::new(RwLock::new(HashMap::new())),
               collection_config: config,
               storage_backend: storage,
           }
       }
       
       pub async fn record_metric(
           &self,
           metric_name: &str,
           value: f64,
           tags: HashMap<String, String>,
           timestamp: Option<DateTime<Utc>>,
       ) -> Result<(), MetricsError> {
           let timestamp = timestamp.unwrap_or_else(Utc::now);
           
           let point = MetricPoint {
               timestamp,
               value,
               tags: tags.clone(),
               metadata: None,
           };
           
           // Store in time series
           {
               let mut series_map = self.time_series.write().await;
               let series_key = self.generate_series_key(metric_name, &tags);
               
               let series = series_map.entry(series_key.clone()).or_insert_with(|| {
                   TimeSeries {
                       metric_name: metric_name.to_string(),
                       points: BTreeMap::new(),
                       retention_period: self.collection_config.retention_period,
                       aggregation_intervals: self.collection_config.aggregation_intervals.clone(),
                   }
               });
               
               series.points.insert(timestamp, point.clone());
               
               // Enforce retention policy
               self.enforce_retention(&mut series.points, series.retention_period);
           }
           
           // Trigger real-time aggregation
           self.update_real_time_aggregations(metric_name, &point).await?;
           
           // Persist to storage backend
           self.storage_backend.store_metric_point(metric_name, &point).await?;
           
           Ok(())
       }
       
       async fn update_real_time_aggregations(
           &self,
           metric_name: &str,
           point: &MetricPoint,
       ) -> Result<(), MetricsError> {
           let mut aggregated = self.aggregated_data.write().await;
           
           for interval in &self.collection_config.aggregation_intervals {
               let window_start = self.get_window_start(point.timestamp, *interval);
               let agg_key = format!("{}:{}:{}", metric_name, interval.num_seconds(), window_start.timestamp());
               
               let agg_metrics = aggregated.entry(agg_key).or_insert_with(|| {
                   AggregatedMetrics {
                       metric_name: metric_name.to_string(),
                       window_start,
                       window_end: window_start + *interval,
                       interval: *interval,
                       count: 0,
                       sum: 0.0,
                       min: f64::INFINITY,
                       max: f64::NEG_INFINITY,
                       mean: 0.0,
                       variance: 0.0,
                       percentiles: HashMap::new(),
                       tags: point.tags.clone(),
                   }
               });
               
               agg_metrics.update(point.value);
           }
           
           Ok(())
       }
       
       fn generate_series_key(&self, metric_name: &str, tags: &HashMap<String, String>) -> String {
           let mut key = metric_name.to_string();
           
           // Sort tags for consistent key generation
           let mut sorted_tags: Vec<_> = tags.iter().collect();
           sorted_tags.sort_by_key(|(k, _)| *k);
           
           for (tag_key, tag_value) in sorted_tags {
               key.push_str(&format!(":{}={}", tag_key, tag_value));
           }
           
           key
       }
   }
   ```

3. Implement statistical aggregation functions:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct AggregatedMetrics {
       pub metric_name: String,
       pub window_start: DateTime<Utc>,
       pub window_end: DateTime<Utc>,
       pub interval: Duration,
       pub count: u64,
       pub sum: f64,
       pub min: f64,
       pub max: f64,
       pub mean: f64,
       pub variance: f64,
       pub percentiles: HashMap<String, f64>, // "p50", "p95", "p99"
       pub tags: HashMap<String, String>,
   }
   
   impl AggregatedMetrics {
       pub fn update(&mut self, value: f64) {
           self.count += 1;
           self.sum += value;
           self.min = self.min.min(value);
           self.max = self.max.max(value);
           
           // Update running mean and variance using Welford's algorithm
           let delta = value - self.mean;
           self.mean += delta / self.count as f64;
           let delta2 = value - self.mean;
           self.variance += delta * delta2;
       }
       
       pub fn finalize(&mut self) {
           if self.count > 1 {
               self.variance /= (self.count - 1) as f64;
           }
       }
       
       pub fn standard_deviation(&self) -> f64 {
           self.variance.sqrt()
       }
   }
   
   pub struct PercentileCalculator {
       values: Vec<f64>,
       is_sorted: bool,
   }
   
   impl PercentileCalculator {
       pub fn new() -> Self {
           Self {
               values: Vec::new(),
               is_sorted: false,
           }
       }
       
       pub fn add_value(&mut self, value: f64) {
           self.values.push(value);
           self.is_sorted = false;
       }
       
       pub fn calculate_percentile(&mut self, percentile: f64) -> Option<f64> {
           if self.values.is_empty() || percentile < 0.0 || percentile > 100.0 {
               return None;
           }
           
           if !self.is_sorted {
               self.values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
               self.is_sorted = true;
           }
           
           let index = (percentile / 100.0) * (self.values.len() - 1) as f64;
           let lower_index = index.floor() as usize;
           let upper_index = index.ceil() as usize;
           
           if lower_index == upper_index {
               Some(self.values[lower_index])
           } else {
               let lower_value = self.values[lower_index];
               let upper_value = self.values[upper_index];
               let weight = index - lower_index as f64;
               Some(lower_value + weight * (upper_value - lower_value))
           }
       }
       
       pub fn calculate_all_percentiles(&mut self) -> HashMap<String, f64> {
           let percentiles = vec![50.0, 75.0, 90.0, 95.0, 99.0, 99.9];
           let mut result = HashMap::new();
           
           for p in percentiles {
               if let Some(value) = self.calculate_percentile(p) {
                   result.insert(format!("p{}", p), value);
               }
           }
           
           result
       }
   }
   ```

4. Implement metrics query and analysis engine:
   ```rust
   #[derive(Debug, Clone)]
   pub struct MetricsQuery {
       pub metric_name: String,
       pub start_time: DateTime<Utc>,
       pub end_time: DateTime<Utc>,
       pub tags: HashMap<String, String>,
       pub aggregation: AggregationType,
       pub group_by: Vec<String>,
       pub having: Option<HavingCondition>,
   }
   
   #[derive(Debug, Clone)]
   pub enum AggregationType {
       Raw,
       Sum,
       Average,
       Min,
       Max,
       Count,
       Rate,
       Percentile(f64),
   }
   
   #[derive(Debug, Clone)]
   pub struct HavingCondition {
       pub field: String,
       pub operator: ComparisonOperator,
       pub value: f64,
   }
   
   #[derive(Debug, Clone)]
   pub enum ComparisonOperator {
       GreaterThan,
       LessThan,
       Equal,
       GreaterEqual,
       LessEqual,
   }
   
   pub struct MetricsQueryEngine {
       aggregator: Arc<MetricsAggregator>,
       cache: Arc<RwLock<lru::LruCache<String, QueryResult>>>,
   }
   
   impl MetricsQueryEngine {
       pub async fn execute_query(&self, query: &MetricsQuery) -> Result<QueryResult, MetricsError> {
           let cache_key = self.generate_cache_key(query);
           
           // Check cache first
           {
               let mut cache = self.cache.write().await;
               if let Some(cached_result) = cache.get(&cache_key) {
                   if !self.is_cache_expired(cached_result) {
                       return Ok(cached_result.clone());
                   }
               }
           }
           
           // Execute query
           let result = self.execute_query_internal(query).await?;
           
           // Cache result
           {
               let mut cache = self.cache.write().await;
               cache.put(cache_key, result.clone());
           }
           
           Ok(result)
       }
       
       async fn execute_query_internal(&self, query: &MetricsQuery) -> Result<QueryResult, MetricsError> {
           let time_series = self.aggregator.time_series.read().await;
           let mut matching_series = Vec::new();
           
           // Find matching time series
           for (series_key, series) in time_series.iter() {
               if series.metric_name == query.metric_name {
                   if self.matches_tags(&series.points, &query.tags) {
                       matching_series.push(series);
                   }
               }
           }
           
           if matching_series.is_empty() {
               return Ok(QueryResult {
                   metric_name: query.metric_name.clone(),
                   series: Vec::new(),
                   query_time: Utc::now(),
                   execution_time_ms: 0,
               });
           }
           
           let start_time = std::time::Instant::now();
           let mut result_series = Vec::new();
           
           for series in matching_series {
               let filtered_points = self.filter_by_time_range(
                   &series.points,
                   query.start_time,
                   query.end_time,
               );
               
               let aggregated_data = match query.aggregation {
                   AggregationType::Raw => self.raw_aggregation(filtered_points),
                   AggregationType::Sum => self.sum_aggregation(filtered_points),
                   AggregationType::Average => self.average_aggregation(filtered_points),
                   AggregationType::Min => self.min_aggregation(filtered_points),
                   AggregationType::Max => self.max_aggregation(filtered_points),
                   AggregationType::Count => self.count_aggregation(filtered_points),
                   AggregationType::Rate => self.rate_aggregation(filtered_points, query.end_time - query.start_time),
                   AggregationType::Percentile(p) => self.percentile_aggregation(filtered_points, p),
               };
               
               if let Some(grouped_data) = self.group_by_tags(aggregated_data, &query.group_by) {
                   let filtered_data = self.apply_having_condition(grouped_data, &query.having);
                   result_series.extend(filtered_data);
               }
           }
           
           let execution_time = start_time.elapsed();
           
           Ok(QueryResult {
               metric_name: query.metric_name.clone(),
               series: result_series,
               query_time: Utc::now(),
               execution_time_ms: execution_time.as_millis() as u64,
           })
       }
       
       fn rate_aggregation(&self, points: Vec<&MetricPoint>, duration: Duration) -> Vec<AggregationPoint> {
           if points.len() < 2 {
               return Vec::new();
           }
           
           let total_value_change = points.last().unwrap().value - points.first().unwrap().value;
           let rate = total_value_change / duration.num_seconds() as f64;
           
           vec![AggregationPoint {
               timestamp: points.first().unwrap().timestamp,
               value: rate,
               tags: points.first().unwrap().tags.clone(),
           }]
       }
   }
   ```

5. Implement real-time metrics dashboard data:
   ```rust
   #[derive(Debug, Clone, Serialize)]
   pub struct DashboardMetrics {
       pub algorithm_performance: AlgorithmPerformanceMetrics,
       pub system_health: SystemHealthMetrics,
       pub resource_utilization: ResourceUtilizationMetrics,
       pub error_rates: ErrorRateMetrics,
       pub throughput: ThroughputMetrics,
   }
   
   #[derive(Debug, Clone, Serialize)]
   pub struct AlgorithmPerformanceMetrics {
       pub active_algorithms: u64,
       pub average_execution_time: f64,
       pub success_rate: f64,
       pub algorithms_per_minute: f64,
       pub top_algorithms: Vec<AlgorithmStats>,
   }
   
   #[derive(Debug, Clone, Serialize)]
   pub struct AlgorithmStats {
       pub name: String,
       pub execution_count: u64,
       pub average_time_ms: f64,
       pub success_rate: f64,
       pub last_execution: DateTime<Utc>,
   }
   
   pub struct DashboardDataProvider {
       metrics_aggregator: Arc<MetricsAggregator>,
       query_engine: Arc<MetricsQueryEngine>,
       cache_ttl: Duration,
   }
   
   impl DashboardDataProvider {
       pub async fn get_real_time_metrics(&self) -> Result<DashboardMetrics, MetricsError> {
           let now = Utc::now();
           let one_hour_ago = now - Duration::hours(1);
           let five_minutes_ago = now - Duration::minutes(5);
           
           // Get algorithm performance metrics
           let algorithm_perf = self.get_algorithm_performance_metrics(one_hour_ago, now).await?;
           
           // Get system health metrics
           let system_health = self.get_system_health_metrics(five_minutes_ago, now).await?;
           
           // Get resource utilization
           let resource_util = self.get_resource_utilization_metrics(five_minutes_ago, now).await?;
           
           // Get error rates
           let error_rates = self.get_error_rate_metrics(one_hour_ago, now).await?;
           
           // Get throughput metrics
           let throughput = self.get_throughput_metrics(one_hour_ago, now).await?;
           
           Ok(DashboardMetrics {
               algorithm_performance: algorithm_perf,
               system_health,
               resource_utilization: resource_util,
               error_rates,
               throughput,
           })
       }
       
       async fn get_algorithm_performance_metrics(
           &self,
           start_time: DateTime<Utc>,
           end_time: DateTime<Utc>,
       ) -> Result<AlgorithmPerformanceMetrics, MetricsError> {
           // Query active algorithms
           let active_query = MetricsQuery {
               metric_name: "active_algorithms".to_string(),
               start_time,
               end_time,
               tags: HashMap::new(),
               aggregation: AggregationType::Average,
               group_by: Vec::new(),
               having: None,
           };
           
           let active_result = self.query_engine.execute_query(&active_query).await?;
           let active_algorithms = active_result.series.first()
               .map(|s| s.points.last().map(|p| p.value as u64).unwrap_or(0))
               .unwrap_or(0);
           
           // Query execution times
           let exec_time_query = MetricsQuery {
               metric_name: "algorithm_execution_time".to_string(),
               start_time,
               end_time,
               tags: HashMap::new(),
               aggregation: AggregationType::Average,
               group_by: vec!["algorithm".to_string()],
               having: None,
           };
           
           let exec_time_result = self.query_engine.execute_query(&exec_time_query).await?;
           let average_execution_time = exec_time_result.series.iter()
               .map(|s| s.points.iter().map(|p| p.value).sum::<f64>() / s.points.len() as f64)
               .sum::<f64>() / exec_time_result.series.len().max(1) as f64;
           
           // Calculate success rate
           let success_rate = self.calculate_success_rate(start_time, end_time).await?;
           
           // Calculate algorithms per minute
           let algorithms_per_minute = self.calculate_throughput_rate(start_time, end_time).await?;
           
           // Get top algorithms
           let top_algorithms = self.get_top_algorithms(start_time, end_time, 10).await?;
           
           Ok(AlgorithmPerformanceMetrics {
               active_algorithms,
               average_execution_time,
               success_rate,
               algorithms_per_minute,
               top_algorithms,
           })
       }
   }
   ```

## Expected Output
```rust
pub trait MetricsAggregation {
    async fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<(), MetricsError>;
    async fn query_metrics(&self, query: &MetricsQuery) -> Result<QueryResult, MetricsError>;
    async fn get_aggregated_data(&self, interval: Duration, aggregation: AggregationType) -> Result<Vec<AggregatedMetrics>, MetricsError>;
    async fn get_dashboard_metrics(&self) -> Result<DashboardMetrics, MetricsError>;
}

#[derive(Debug)]
pub enum MetricsError {
    StorageError(String),
    QueryError(String),
    AggregationError(String),
    SerializationError(serde_json::Error),
    TimeRangeError(String),
}

pub struct MetricsExporter {
    prometheus_registry: prometheus::Registry,
    influxdb_client: Option<influxdb::Client>,
    export_interval: Duration,
}
```

## Verification Steps
1. Test metrics collection under high-frequency data ingestion
2. Verify aggregation accuracy across different time windows
3. Test query performance with large time-series datasets
4. Validate percentile calculations for various distributions
5. Test dashboard data refresh performance
6. Benchmark memory usage with extended retention periods

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- chrono: Date/time handling
- serde: Serialization framework
- prometheus: Metrics export
- influxdb: Time-series database (optional)
- lru: Caching support