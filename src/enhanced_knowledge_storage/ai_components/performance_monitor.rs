//! Performance Monitoring for AI Components
//! 
//! Provides comprehensive performance monitoring, metrics collection, and analysis
//! for all AI model operations in the Enhanced Knowledge Storage System.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::types::*;

/// Performance metrics for model operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub model_id: String,
    pub operation_type: OperationType,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration: Duration,
    pub tokens_processed: usize,
    pub memory_usage: MemoryUsage,
    pub throughput: f64,
    pub latency_percentiles: LatencyPercentiles,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Inference,
    ModelLoad,
    ModelUnload,
    Tokenization,
    Embedding,
    EntityExtraction,
    SemanticChunking,
    Reasoning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub before_mb: f64,
    pub after_mb: f64,
    pub peak_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Main performance monitor for AI components
pub struct PerformanceMonitor {
    metrics_store: Arc<RwLock<Vec<ModelPerformanceMetrics>>>,
    active_operations: Arc<RwLock<HashMap<String, OperationTracker>>>,
    aggregated_stats: Arc<RwLock<AggregatedStats>>,
}

struct OperationTracker {
    operation_id: String,
    model_id: String,
    operation_type: OperationType,
    start_time: Instant,
    start_memory: f64,
    tokens_processed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AggregatedStats {
    total_operations: u64,
    successful_operations: u64,
    failed_operations: u64,
    total_tokens_processed: u64,
    average_throughput: f64,
    model_stats: HashMap<String, ModelStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelStats {
    operation_count: u64,
    total_duration: Duration,
    average_latency: Duration,
    success_rate: f64,
    total_tokens: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics_store: Arc::new(RwLock::new(Vec::new())),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            aggregated_stats: Arc::new(RwLock::new(AggregatedStats {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                total_tokens_processed: 0,
                average_throughput: 0.0,
                model_stats: HashMap::new(),
            })),
        }
    }

    /// Start tracking a model operation
    pub async fn start_operation(
        &self,
        operation_id: String,
        model_id: String,
        operation_type: OperationType,
    ) -> Result<(), PerformanceError> {
        let tracker = OperationTracker {
            operation_id: operation_id.clone(),
            model_id,
            operation_type,
            start_time: Instant::now(),
            start_memory: self.get_current_memory_mb(),
            tokens_processed: 0,
        };

        self.active_operations.write().await.insert(operation_id, tracker);
        Ok(())
    }

    /// Update token count for an active operation
    pub async fn update_tokens(
        &self,
        operation_id: &str,
        tokens: usize,
    ) -> Result<(), PerformanceError> {
        if let Some(tracker) = self.active_operations.write().await.get_mut(operation_id) {
            tracker.tokens_processed = tokens;
            Ok(())
        } else {
            Err(PerformanceError::OperationNotFound(operation_id.to_string()))
        }
    }

    /// Complete tracking for an operation
    pub async fn complete_operation(
        &self,
        operation_id: &str,
        success: bool,
        error_message: Option<String>,
    ) -> Result<ModelPerformanceMetrics, PerformanceError> {
        let tracker = self.active_operations.write().await.remove(operation_id)
            .ok_or_else(|| PerformanceError::OperationNotFound(operation_id.to_string()))?;

        let end_time = Instant::now();
        let duration = end_time - tracker.start_time;
        let end_memory = self.get_current_memory_mb();

        let metrics = ModelPerformanceMetrics {
            model_id: tracker.model_id.clone(),
            operation_type: tracker.operation_type,
            start_time: Utc::now() - chrono::Duration::from_std(duration).unwrap(),
            end_time: Utc::now(),
            duration,
            tokens_processed: tracker.tokens_processed,
            memory_usage: MemoryUsage {
                before_mb: tracker.start_memory,
                after_mb: end_memory,
                peak_mb: tracker.start_memory.max(end_memory),
            },
            throughput: if duration.as_secs_f64() > 0.0 {
                tracker.tokens_processed as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
            latency_percentiles: self.calculate_latency_percentiles(duration),
            success,
            error_message,
        };

        // Store metrics
        self.metrics_store.write().await.push(metrics.clone());

        // Update aggregated stats
        self.update_aggregated_stats(&metrics).await;

        Ok(metrics)
    }

    /// Get performance summary for a specific model
    pub async fn get_model_summary(&self, model_id: &str) -> Option<ModelStats> {
        self.aggregated_stats.read().await.model_stats.get(model_id).cloned()
    }

    /// Get recent metrics within time window
    pub async fn get_recent_metrics(&self, window: Duration) -> Vec<ModelPerformanceMetrics> {
        let cutoff = Utc::now() - chrono::Duration::from_std(window).unwrap();
        self.metrics_store
            .read()
            .await
            .iter()
            .filter(|m| m.start_time > cutoff)
            .cloned()
            .collect()
    }

    /// Export metrics for analysis
    pub async fn export_metrics(&self) -> Result<String, PerformanceError> {
        let metrics = self.metrics_store.read().await;
        serde_json::to_string_pretty(&*metrics)
            .map_err(|e| PerformanceError::SerializationError(e.to_string()))
    }

    /// Clear old metrics to prevent memory growth
    pub async fn cleanup_old_metrics(&self, retention: Duration) {
        let cutoff = Utc::now() - chrono::Duration::from_std(retention).unwrap();
        let mut metrics = self.metrics_store.write().await;
        metrics.retain(|m| m.start_time > cutoff);
    }

    // Private helper methods

    fn get_current_memory_mb(&self) -> f64 {
        // Placeholder - in production, use actual memory measurement
        // Could integrate with system metrics or Rust memory allocator stats
        100.0
    }

    fn calculate_latency_percentiles(&self, duration: Duration) -> LatencyPercentiles {
        // Simplified percentile calculation
        // In production, maintain histogram of latencies for accurate percentiles
        LatencyPercentiles {
            p50: duration,
            p90: duration.mul_f64(1.1),
            p95: duration.mul_f64(1.2),
            p99: duration.mul_f64(1.5),
        }
    }

    async fn update_aggregated_stats(&self, metrics: &ModelPerformanceMetrics) {
        let mut stats = self.aggregated_stats.write().await;
        
        stats.total_operations += 1;
        if metrics.success {
            stats.successful_operations += 1;
        } else {
            stats.failed_operations += 1;
        }
        
        stats.total_tokens_processed += metrics.tokens_processed as u64;
        
        // Update average throughput
        let total_ops = stats.total_operations as f64;
        stats.average_throughput = 
            (stats.average_throughput * (total_ops - 1.0) + metrics.throughput) / total_ops;
        
        // Update model-specific stats
        let model_stat = stats.model_stats.entry(metrics.model_id.clone()).or_insert(ModelStats {
            operation_count: 0,
            total_duration: Duration::default(),
            average_latency: Duration::default(),
            success_rate: 0.0,
            total_tokens: 0,
        });
        
        model_stat.operation_count += 1;
        model_stat.total_duration += metrics.duration;
        model_stat.average_latency = model_stat.total_duration / model_stat.operation_count as u32;
        model_stat.total_tokens += metrics.tokens_processed as u64;
        
        let successful = if metrics.success { 1.0 } else { 0.0 };
        model_stat.success_rate = 
            (model_stat.success_rate * (model_stat.operation_count - 1) as f64 + successful) 
            / model_stat.operation_count as f64;
    }
}

/// Performance monitoring error types
#[derive(Debug, thiserror::Error)]
pub enum PerformanceError {
    #[error("Operation not found: {0}")]
    OperationNotFound(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Metrics collection error: {0}")]
    MetricsError(String),
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_tracking() {
        let monitor = PerformanceMonitor::new();
        
        // Start operation
        monitor.start_operation(
            "test_op_1".to_string(),
            "model_135M".to_string(),
            OperationType::Inference,
        ).await.unwrap();
        
        // Update tokens
        monitor.update_tokens("test_op_1", 1000).await.unwrap();
        
        // Complete operation
        let metrics = monitor.complete_operation("test_op_1", true, None).await.unwrap();
        
        assert_eq!(metrics.model_id, "model_135M");
        assert_eq!(metrics.tokens_processed, 1000);
        assert!(metrics.success);
    }

    #[tokio::test]
    async fn test_model_summary() {
        let monitor = PerformanceMonitor::new();
        
        // Track multiple operations
        for i in 0..5 {
            let op_id = format!("op_{}", i);
            monitor.start_operation(
                op_id.clone(),
                "test_model".to_string(),
                OperationType::Inference,
            ).await.unwrap();
            
            monitor.update_tokens(&op_id, 500).await.unwrap();
            monitor.complete_operation(&op_id, true, None).await.unwrap();
        }
        
        // Get summary
        let summary = monitor.get_model_summary("test_model").await.unwrap();
        assert_eq!(summary.operation_count, 5);
        assert_eq!(summary.success_rate, 1.0);
        assert_eq!(summary.total_tokens, 2500);
    }
}