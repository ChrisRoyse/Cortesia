# MP056: Logging & Tracing Integration

## Task Description
Implement comprehensive logging and distributed tracing system for monitoring graph algorithm execution, performance analysis, and debugging across neuromorphic components.

## Prerequisites
- MP001-MP050 completed
- Understanding of structured logging principles
- Knowledge of distributed tracing concepts
- Familiarity with observability patterns

## Detailed Steps

1. Create `src/neuromorphic/observability/logging.rs`

2. Implement structured logging with contextual information:
   ```rust
   use tracing::{info, warn, error, debug, trace, Instrument, Span};
   use tracing_subscriber::{
       layer::SubscriberExt,
       util::SubscriberInitExt,
       fmt,
       EnvFilter,
   };
   use serde::{Serialize, Deserialize};
   use uuid::Uuid;
   use std::collections::HashMap;
   use std::sync::Arc;
   use chrono::{DateTime, Utc};
   
   #[derive(Debug, Clone, Serialize)]
   pub struct GraphAlgorithmContext {
       pub algorithm_name: String,
       pub job_id: Uuid,
       pub graph_size: GraphSize,
       pub parameters: HashMap<String, serde_json::Value>,
       pub user_id: Option<String>,
       pub session_id: Option<String>,
       pub request_id: String,
   }
   
   #[derive(Debug, Clone, Serialize)]
   pub struct GraphSize {
       pub node_count: u64,
       pub edge_count: u64,
       pub memory_usage_bytes: u64,
   }
   
   #[derive(Debug, Clone, Serialize)]
   pub struct PerformanceMetrics {
       pub execution_time_ms: u64,
       pub memory_peak_mb: f64,
       pub cpu_usage_percent: f64,
       pub cache_hit_ratio: f64,
       pub nodes_processed: u64,
       pub edges_processed: u64,
   }
   
   pub struct StructuredLogger {
       context_storage: Arc<tokio::sync::RwLock<HashMap<String, serde_json::Value>>>,
       performance_tracker: PerformanceTracker,
       log_filters: LogFilters,
   }
   
   impl StructuredLogger {
       pub fn init() -> Result<Self, LoggingError> {
           // Initialize tracing subscriber with multiple layers
           let env_filter = EnvFilter::try_from_default_env()
               .unwrap_or_else(|_| EnvFilter::new("info"));
           
           let formatting_layer = fmt::layer()
               .with_target(true)
               .with_thread_ids(true)
               .with_file(true)
               .with_line_number(true)
               .json();
           
           let file_appender = tracing_appender::rolling::daily("logs", "neuromorphic.log");
           let file_layer = fmt::layer()
               .with_writer(file_appender)
               .json();
           
           tracing_subscriber::registry()
               .with(env_filter)
               .with(formatting_layer)
               .with(file_layer)
               .init();
           
           Ok(Self {
               context_storage: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
               performance_tracker: PerformanceTracker::new(),
               log_filters: LogFilters::default(),
           })
       }
       
       pub async fn log_algorithm_start(&self, context: &GraphAlgorithmContext) {
           let span = tracing::info_span!(
               "algorithm_execution",
               algorithm = %context.algorithm_name,
               job_id = %context.job_id,
               node_count = context.graph_size.node_count,
               edge_count = context.graph_size.edge_count
           );
           
           let _enter = span.enter();
           
           info!(
               algorithm = %context.algorithm_name,
               job_id = %context.job_id,
               graph_size = ?context.graph_size,
               parameters = ?context.parameters,
               user_id = ?context.user_id,
               "Algorithm execution started"
           );
           
           // Store context for later correlation
           {
               let mut storage = self.context_storage.write().await;
               storage.insert(
                   context.job_id.to_string(),
                   serde_json::to_value(context).unwrap_or_default(),
               );
           }
           
           self.performance_tracker.start_tracking(&context.job_id).await;
       }
       
       pub async fn log_algorithm_progress(
           &self,
           job_id: Uuid,
           progress: f32,
           current_step: &str,
           metrics: Option<PerformanceMetrics>,
       ) {
           let span = tracing::debug_span!(
               "algorithm_progress",
               job_id = %job_id,
               progress = progress,
               step = current_step
           );
           
           let _enter = span.enter();
           
           debug!(
               job_id = %job_id,
               progress = progress,
               current_step = current_step,
               metrics = ?metrics,
               "Algorithm progress update"
           );
           
           if let Some(metrics) = metrics {
               self.log_performance_metrics(job_id, &metrics).await;
           }
       }
       
       pub async fn log_algorithm_completion(
           &self,
           job_id: Uuid,
           result: &AlgorithmResult,
           final_metrics: PerformanceMetrics,
       ) {
           let span = tracing::info_span!(
               "algorithm_completion",
               job_id = %job_id,
               success = result.is_success(),
               execution_time_ms = final_metrics.execution_time_ms
           );
           
           let _enter = span.enter();
           
           match result {
               AlgorithmResult::Success { data, .. } => {
                   info!(
                       job_id = %job_id,
                       result_size = data.len(),
                       metrics = ?final_metrics,
                       "Algorithm execution completed successfully"
                   );
               }
               AlgorithmResult::Error { error, .. } => {
                   error!(
                       job_id = %job_id,
                       error = %error,
                       metrics = ?final_metrics,
                       "Algorithm execution failed"
                   );
               }
           }
           
           // Clean up context storage
           {
               let mut storage = self.context_storage.write().await;
               storage.remove(&job_id.to_string());
           }
           
           self.performance_tracker.stop_tracking(&job_id).await;
       }
   }
   ```

3. Implement distributed tracing with OpenTelemetry:
   ```rust
   use opentelemetry::{
       global,
       sdk::{
           trace::{self, RandomIdGenerator, Sampler},
           Resource,
       },
       trace::{TraceError, Tracer, TracerProvider},
       KeyValue,
   };
   use opentelemetry_jaeger::new_agent_pipeline;
   use tracing_opentelemetry::OpenTelemetryLayer;
   
   pub struct DistributedTracing {
       tracer: Box<dyn Tracer + Send + Sync>,
       span_processor: SpanProcessor,
       trace_config: TraceConfig,
   }
   
   #[derive(Debug, Clone)]
   pub struct TraceConfig {
       pub service_name: String,
       pub service_version: String,
       pub jaeger_endpoint: Option<String>,
       pub sampling_ratio: f64,
       pub max_span_attributes: u32,
       pub max_span_events: u32,
   }
   
   impl DistributedTracing {
       pub fn init(config: TraceConfig) -> Result<Self, TraceError> {
           // Configure OpenTelemetry resource
           let resource = Resource::new(vec![
               KeyValue::new("service.name", config.service_name.clone()),
               KeyValue::new("service.version", config.service_version.clone()),
               KeyValue::new("service.namespace", "neuromorphic"),
           ]);
           
           // Configure tracer provider
           let tracer_provider = if let Some(jaeger_endpoint) = &config.jaeger_endpoint {
               // Export to Jaeger
               new_agent_pipeline()
                   .with_service_name(&config.service_name)
                   .with_endpoint(jaeger_endpoint)
                   .with_trace_config(
                       trace::config()
                           .with_resource(resource)
                           .with_sampler(Sampler::TraceIdRatioBased(config.sampling_ratio))
                           .with_id_generator(RandomIdGenerator::default())
                   )
                   .install_batch(opentelemetry::runtime::Tokio)?
           } else {
               // Use stdout exporter for development
               opentelemetry::sdk::trace::TracerProvider::builder()
                   .with_resource(resource)
                   .with_batch_exporter(
                       opentelemetry_stdout::SpanExporter::default(),
                       opentelemetry::runtime::Tokio,
                   )
                   .build()
           };
           
           global::set_tracer_provider(tracer_provider.clone());
           let tracer = tracer_provider.tracer("neuromorphic-graph-algorithms");
           
           // Add OpenTelemetry layer to tracing subscriber
           let opentelemetry_layer = OpenTelemetryLayer::new(tracer.clone());
           
           tracing_subscriber::registry()
               .with(opentelemetry_layer)
               .init();
           
           Ok(Self {
               tracer: Box::new(tracer),
               span_processor: SpanProcessor::new(),
               trace_config: config,
           })
       }
       
       pub async fn trace_algorithm_execution<F, R>(
           &self,
           context: &GraphAlgorithmContext,
           operation: F,
       ) -> R
       where
           F: Future<Output = R> + Send + 'static,
           R: Send + 'static,
       {
           let span = self.tracer.start_with_context(
               &format!("execute_{}", context.algorithm_name),
               &opentelemetry::Context::current(),
           );
           
           // Add span attributes
           let mut span = span;
           span.set_attribute(KeyValue::new("algorithm.name", context.algorithm_name.clone()));
           span.set_attribute(KeyValue::new("job.id", context.job_id.to_string()));
           span.set_attribute(KeyValue::new("graph.node_count", context.graph_size.node_count as i64));
           span.set_attribute(KeyValue::new("graph.edge_count", context.graph_size.edge_count as i64));
           
           if let Some(user_id) = &context.user_id {
               span.set_attribute(KeyValue::new("user.id", user_id.clone()));
           }
           
           // Execute operation within span context
           let result = {
               let _guard = span.enter();
               operation.await
           };
           
           span.end();
           result
       }
       
       pub fn create_child_span(&self, name: &str, parent_context: &opentelemetry::Context) -> opentelemetry::trace::Span {
           self.tracer.start_with_context(name, parent_context)
       }
   }
   ```

4. Implement performance monitoring and alerting:
   ```rust
   use metrics::{counter, gauge, histogram, register_counter, register_gauge, register_histogram};
   use metrics_exporter_prometheus::PrometheusBuilder;
   use std::time::{Duration, Instant};
   
   pub struct PerformanceMonitor {
       metrics_registry: MetricsRegistry,
       alert_manager: AlertManager,
       thresholds: PerformanceThresholds,
   }
   
   #[derive(Debug, Clone)]
   pub struct PerformanceThresholds {
       pub max_execution_time_ms: u64,
       pub max_memory_usage_mb: f64,
       pub min_cache_hit_ratio: f64,
       pub max_error_rate: f64,
   }
   
   impl PerformanceMonitor {
       pub fn init() -> Result<Self, MonitoringError> {
           // Initialize Prometheus metrics exporter
           let builder = PrometheusBuilder::new();
           builder
               .with_http_listener(([0, 0, 0, 0], 9090))
               .install()
               .expect("Failed to install Prometheus recorder");
           
           // Register metrics
           register_counter!("algorithm_executions_total", "Total number of algorithm executions");
           register_histogram!("algorithm_execution_duration_seconds", "Algorithm execution time");
           register_gauge!("algorithm_memory_usage_bytes", "Memory usage during algorithm execution");
           register_counter!("algorithm_errors_total", "Total number of algorithm errors");
           register_gauge!("active_algorithms", "Number of currently running algorithms");
           
           Ok(Self {
               metrics_registry: MetricsRegistry::new(),
               alert_manager: AlertManager::new(),
               thresholds: PerformanceThresholds::default(),
           })
       }
       
       pub async fn track_algorithm_execution<F, R>(
           &self,
           algorithm_name: &str,
           operation: F,
       ) -> R
       where
           F: Future<Output = R> + Send,
       {
           let start_time = Instant::now();
           
           // Increment active algorithms gauge
           gauge!("active_algorithms").increment(1.0);
           
           // Execute operation
           let result = operation.await;
           
           let execution_time = start_time.elapsed();
           
           // Record metrics
           counter!("algorithm_executions_total", "algorithm" => algorithm_name.to_string()).increment(1);
           histogram!("algorithm_execution_duration_seconds", "algorithm" => algorithm_name.to_string())
               .record(execution_time.as_secs_f64());
           
           // Decrement active algorithms gauge
           gauge!("active_algorithms").decrement(1.0);
           
           // Check thresholds and trigger alerts if necessary
           self.check_performance_thresholds(algorithm_name, execution_time).await;
           
           result
       }
       
       pub async fn record_memory_usage(&self, algorithm_name: &str, memory_bytes: u64) {
           gauge!("algorithm_memory_usage_bytes", "algorithm" => algorithm_name.to_string())
               .set(memory_bytes as f64);
           
           let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;
           if memory_mb > self.thresholds.max_memory_usage_mb {
               self.alert_manager.trigger_alert(Alert {
                   level: AlertLevel::Warning,
                   title: "High memory usage".to_string(),
                   description: format!(
                       "Algorithm {} is using {:.2} MB of memory, exceeding threshold of {:.2} MB",
                       algorithm_name, memory_mb, self.thresholds.max_memory_usage_mb
                   ),
                   algorithm: Some(algorithm_name.to_string()),
                   timestamp: Utc::now(),
               }).await;
           }
       }
       
       pub async fn record_error(&self, algorithm_name: &str, error: &str) {
           counter!("algorithm_errors_total", "algorithm" => algorithm_name.to_string(), "error" => error.to_string())
               .increment(1);
           
           error!(
               algorithm = algorithm_name,
               error = error,
               "Algorithm execution error"
           );
           
           self.alert_manager.trigger_alert(Alert {
               level: AlertLevel::Error,
               title: "Algorithm execution error".to_string(),
               description: format!("Algorithm {} failed with error: {}", algorithm_name, error),
               algorithm: Some(algorithm_name.to_string()),
               timestamp: Utc::now(),
           }).await;
       }
   }
   ```

5. Implement log aggregation and analysis:
   ```rust
   use elasticsearch::{Elasticsearch, SearchParts, BulkParts};
   use serde_json::{json, Value};
   
   pub struct LogAggregator {
       elasticsearch_client: Elasticsearch,
       log_buffer: Arc<Mutex<Vec<LogEntry>>>,
       buffer_size: usize,
       flush_interval: Duration,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct LogEntry {
       pub timestamp: DateTime<Utc>,
       pub level: String,
       pub message: String,
       pub algorithm: Option<String>,
       pub job_id: Option<Uuid>,
       pub user_id: Option<String>,
       pub trace_id: Option<String>,
       pub span_id: Option<String>,
       pub fields: HashMap<String, serde_json::Value>,
   }
   
   impl LogAggregator {
       pub fn new(elasticsearch_url: &str, buffer_size: usize) -> Result<Self, LoggingError> {
           let elasticsearch_client = Elasticsearch::default();
           
           Ok(Self {
               elasticsearch_client,
               log_buffer: Arc::new(Mutex::new(Vec::with_capacity(buffer_size))),
               buffer_size,
               flush_interval: Duration::from_secs(30),
           })
       }
       
       pub async fn start_log_shipping(&self) {
           let mut interval = tokio::time::interval(self.flush_interval);
           
           loop {
               interval.tick().await;
               
               if let Err(e) = self.flush_logs().await {
                   error!("Failed to flush logs to Elasticsearch: {}", e);
               }
           }
       }
       
       async fn flush_logs(&self) -> Result<(), LoggingError> {
           let logs = {
               let mut buffer = self.log_buffer.lock().await;
               if buffer.is_empty() {
                   return Ok(());
               }
               std::mem::take(&mut *buffer)
           };
           
           let mut bulk_body = Vec::new();
           
           for log_entry in logs {
               // Add index action
               bulk_body.push(json!({
                   "index": {
                       "_index": format!("neuromorphic-logs-{}", log_entry.timestamp.format("%Y-%m-%d")),
                       "_type": "_doc"
                   }
               }));
               
               // Add document
               bulk_body.push(serde_json::to_value(&log_entry)?);
           }
           
           if !bulk_body.is_empty() {
               let response = self.elasticsearch_client
                   .bulk(BulkParts::None)
                   .body(bulk_body)
                   .send()
                   .await?;
               
               if response.status_code().is_success() {
                   debug!("Successfully shipped logs to Elasticsearch");
               } else {
                   warn!("Failed to ship logs: {}", response.status_code());
               }
           }
           
           Ok(())
       }
       
       pub async fn search_logs(&self, query: LogQuery) -> Result<Vec<LogEntry>, LoggingError> {
           let search_body = json!({
               "query": {
                   "bool": {
                       "must": self.build_query_filters(&query),
                       "filter": [
                           {
                               "range": {
                                   "timestamp": {
                                       "gte": query.start_time.map(|t| t.to_rfc3339()),
                                       "lte": query.end_time.map(|t| t.to_rfc3339())
                                   }
                               }
                           }
                       ]
                   }
               },
               "sort": [
                   { "timestamp": { "order": "desc" } }
               ],
               "size": query.limit.unwrap_or(100)
           });
           
           let response = self.elasticsearch_client
               .search(SearchParts::Index(&["neuromorphic-logs-*"]))
               .body(search_body)
               .send()
               .await?;
           
           let response_body: Value = response.json().await?;
           let hits = response_body["hits"]["hits"].as_array().unwrap_or(&vec![]);
           
           let mut log_entries = Vec::new();
           for hit in hits {
               if let Ok(log_entry) = serde_json::from_value::<LogEntry>(hit["_source"].clone()) {
                   log_entries.push(log_entry);
               }
           }
           
           Ok(log_entries)
       }
   }
   ```

## Expected Output
```rust
pub trait Logging {
    async fn log_algorithm_start(&self, context: &GraphAlgorithmContext);
    async fn log_algorithm_progress(&self, job_id: Uuid, progress: f32, step: &str);
    async fn log_algorithm_completion(&self, job_id: Uuid, result: &AlgorithmResult);
    async fn log_error(&self, error: &dyn std::error::Error, context: Option<&GraphAlgorithmContext>);
}

pub trait Tracing {
    async fn trace_operation<F, R>(&self, name: &str, context: &TraceContext, operation: F) -> R
    where
        F: Future<Output = R> + Send,
        R: Send;
    fn create_span(&self, name: &str, parent: Option<&SpanContext>) -> Span;
}

#[derive(Debug)]
pub enum LoggingError {
    InitializationError(String),
    SerializationError(serde_json::Error),
    ElasticsearchError(elasticsearch::Error),
    TracingError(TraceError),
}
```

## Verification Steps
1. Test structured logging output format and fields
2. Verify distributed trace propagation across services
3. Test log aggregation and search functionality
4. Validate performance metrics collection and accuracy
5. Test alert triggering and notification delivery
6. Benchmark logging overhead on algorithm performance

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- tracing: Structured logging framework
- opentelemetry: Distributed tracing
- metrics: Performance metrics collection
- elasticsearch: Log aggregation
- serde: Serialization support