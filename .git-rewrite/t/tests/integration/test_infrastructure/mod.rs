// Integration Test Infrastructure
// Provides common utilities and helpers for integration testing

use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use rand::{Rng, thread_rng, SeedableRng};
use rand::rngs::StdRng;

use crate::entity::{Entity, EntityKey};
use crate::relationship::Relationship;
use crate::knowledge_graph::KnowledgeGraph;
use crate::embedding::{EmbeddingStore, EmbeddingVector};
use crate::query::{RagEngine, RagParameters};
use crate::storage::{StorageError, StorageResult};

pub mod data_generator;
pub mod performance_monitor;
pub mod test_scenarios;
pub mod validation_utils;

pub use data_generator::*;
pub use performance_monitor::*;
pub use test_scenarios::*;
pub use validation_utils::*;

/// Integration test environment that manages test lifecycle
#[derive(Debug)]
pub struct IntegrationTestEnvironment {
    pub test_name: String,
    pub test_id: String,
    pub start_time: Instant,
    pub data_generator: TestDataGenerator,
    pub performance_monitor: PerformanceMonitor,
    pub metrics: Arc<Mutex<HashMap<String, f64>>>,
    pub logs: Arc<Mutex<Vec<TestLog>>>,
    pub temp_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestLog {
    pub timestamp: Duration,
    pub level: LogLevel,
    pub message: String,
    pub context: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl IntegrationTestEnvironment {
    /// Create a new integration test environment
    pub fn new(test_name: &str) -> Self {
        let test_id = format!("{}_{}", test_name, uuid::Uuid::new_v4());
        let temp_dir = std::env::temp_dir().join("llmkg_integration_tests").join(&test_id);
        std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

        Self {
            test_name: test_name.to_string(),
            test_id,
            start_time: Instant::now(),
            data_generator: TestDataGenerator::new(),
            performance_monitor: PerformanceMonitor::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
            logs: Arc::new(Mutex::new(Vec::new())),
            temp_dir,
        }
    }

    /// Check if environment is properly initialized
    pub fn is_initialized(&self) -> bool {
        self.temp_dir.exists()
    }

    /// Record a performance measurement
    pub fn record_performance(&mut self, metric_name: &str, duration: Duration) {
        self.performance_monitor.record_duration(metric_name, duration);
        self.log(LogLevel::Info, &format!("Performance metric '{}': {:?}", metric_name, duration));
    }

    /// Record a metric value
    pub fn record_metric(&self, metric_name: &str, value: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.insert(metric_name.to_string(), value);
        self.log(LogLevel::Info, &format!("Metric '{}': {}", metric_name, value));
    }

    /// Record performance for a specific size
    pub fn record_performance_for_size(&mut self, size: u64, metric_name: &str, duration: Duration) {
        let key = format!("{}_{}", metric_name, size);
        self.record_performance(&key, duration);
    }

    /// Record memory usage for a specific size
    pub fn record_memory_usage(&self, size: u64, metric_name: &str, bytes: u64) {
        let key = format!("{}_{}", metric_name, size);
        self.record_metric(&key, bytes as f64);
    }

    /// Record compression ratio
    pub fn record_compression_ratio(&self, size: usize, dim: usize, ratio: f64) {
        let key = format!("compression_{}_{}", size, dim);
        self.record_metric(&key, ratio);
    }

    /// Record reconstruction error
    pub fn record_reconstruction_error(&self, size: usize, dim: usize, error: f64) {
        let key = format!("reconstruction_error_{}_{}", size, dim);
        self.record_metric(&key, error);
    }

    /// Analyze scaling behavior of metrics
    pub fn analyze_scaling_behavior(&self, metric_names: &[&str]) {
        let metrics = self.metrics.lock().unwrap();
        
        for metric_name in metric_names {
            let mut size_value_pairs: Vec<(u64, f64)> = Vec::new();
            
            // Collect all metrics with this name
            for (key, value) in metrics.iter() {
                if key.contains(metric_name) {
                    // Extract size from key (assuming format: metric_name_size)
                    if let Some(size_str) = key.split('_').last() {
                        if let Ok(size) = size_str.parse::<u64>() {
                            size_value_pairs.push((size, *value));
                        }
                    }
                }
            }
            
            if size_value_pairs.len() >= 2 {
                size_value_pairs.sort_by_key(|&(size, _)| size);
                
                // Calculate scaling factor
                let first = size_value_pairs.first().unwrap();
                let last = size_value_pairs.last().unwrap();
                
                let size_ratio = last.0 as f64 / first.0 as f64;
                let value_ratio = last.1 / first.1;
                
                let scaling_factor = value_ratio.log2() / size_ratio.log2();
                
                self.log(LogLevel::Info, &format!(
                    "Scaling analysis for '{}': O(n^{:.2})",
                    metric_name, scaling_factor
                ));
            }
        }
    }

    /// Log a message
    pub fn log(&self, level: LogLevel, message: &str) {
        let mut logs = self.logs.lock().unwrap();
        logs.push(TestLog {
            timestamp: self.start_time.elapsed(),
            level,
            message: message.to_string(),
            context: None,
        });
    }

    /// Generate a test report
    pub fn generate_report(&self) -> TestReport {
        let metrics = self.metrics.lock().unwrap().clone();
        let logs = self.logs.lock().unwrap().clone();
        let performance_summary = self.performance_monitor.get_summary();

        TestReport {
            test_name: self.test_name.clone(),
            test_id: self.test_id.clone(),
            duration: self.start_time.elapsed(),
            metrics,
            performance_summary,
            logs,
            success: true, // Can be updated based on test results
        }
    }

    /// Clean up test environment
    pub fn cleanup(&self) {
        if self.temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.temp_dir);
        }
    }
}

impl Drop for IntegrationTestEnvironment {
    fn drop(&mut self) {
        // Generate and save report
        let report = self.generate_report();
        let report_path = self.temp_dir.join("test_report.json");
        
        if let Ok(report_json) = serde_json::to_string_pretty(&report) {
            let _ = std::fs::write(&report_path, report_json);
        }
        
        // Clean up if not in debug mode
        if !std::env::var("LLMKG_KEEP_TEST_DATA").is_ok() {
            self.cleanup();
        }
    }
}

/// Test report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub test_name: String,
    pub test_id: String,
    pub duration: Duration,
    pub metrics: HashMap<String, f64>,
    pub performance_summary: PerformanceSummary,
    pub logs: Vec<TestLog>,
    pub success: bool,
}

/// Performance summary for a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub measurements: HashMap<String, DurationStats>,
    pub memory_usage: Option<MemoryStats>,
}

/// Duration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStats {
    pub count: usize,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub initial: u64,
    pub peak: u64,
    pub final_value: u64,
    pub allocations: u64,
}

/// Helper functions for integration tests

/// Calculate overlap between two entity sets
pub fn calculate_entity_set_overlap(set1: &HashSet<EntityKey>, set2: &HashSet<EntityKey>) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let union_size = set1.union(set2).count();
    
    if union_size == 0 {
        1.0
    } else {
        intersection_size as f64 / union_size as f64
    }
}

/// Calculate overlap ratio between two sets
pub fn calculate_set_overlap_ratio<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let smaller_size = set1.len().min(set2.len());
    
    if smaller_size == 0 {
        0.0
    } else {
        intersection_size as f64 / smaller_size as f64
    }
}

/// Calculate ranked overlap for similarity results
pub fn calculate_ranked_overlap<T: Eq>(results1: &[T], results2: &[T], top_k: usize) -> f64 {
    let top_k = top_k.min(results1.len()).min(results2.len());
    
    let mut matches = 0;
    for i in 0..top_k {
        for j in 0..top_k {
            if results1[i] == results2[j] {
                matches += 1;
                break;
            }
        }
    }
    
    if top_k == 0 {
        1.0
    } else {
        matches as f64 / top_k as f64
    }
}

/// Calculate euclidean distance between vectors
pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len());
    
    let mut sum = 0.0;
    for i in 0..vec1.len() {
        let diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// Calculate correlation coefficient
pub fn calculate_correlation(data: &[(f32, f32)]) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let n = data.len() as f32;
    let sum_x: f32 = data.iter().map(|(x, _)| x).sum();
    let sum_y: f32 = data.iter().map(|(_, y)| y).sum();
    let sum_xy: f32 = data.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f32 = data.iter().map(|(x, _)| x * x).sum();
    let sum_y2: f32 = data.iter().map(|(_, y)| y * y).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Test result for similarity search
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    pub entity: EntityKey,
    pub distance: f32,
}

/// Mock quantizer for testing
pub struct NoQuantizer;

impl NoQuantizer {
    pub fn quantize(&self, _embedding: &[f32]) -> Vec<u8> {
        Vec::new()
    }
    
    pub fn reconstruct(&self, _quantized: &[u8]) -> Vec<f32> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_environment() {
        let env = IntegrationTestEnvironment::new("test_env");
        assert!(env.is_initialized());
        
        env.record_metric("test_metric", 42.0);
        env.log(LogLevel::Info, "Test log message");
        
        let report = env.generate_report();
        assert_eq!(report.test_name, "test_env");
        assert_eq!(report.metrics.get("test_metric"), Some(&42.0));
        assert_eq!(report.logs.len(), 2); // Including the metric log
    }

    #[test]
    fn test_overlap_calculations() {
        let set1: HashSet<EntityKey> = vec![
            EntityKey::from_hash("a"),
            EntityKey::from_hash("b"),
            EntityKey::from_hash("c"),
        ].into_iter().collect();
        
        let set2: HashSet<EntityKey> = vec![
            EntityKey::from_hash("b"),
            EntityKey::from_hash("c"),
            EntityKey::from_hash("d"),
        ].into_iter().collect();
        
        let overlap = calculate_entity_set_overlap(&set1, &set2);
        assert!((overlap - 0.5).abs() < 0.01); // 2/4 = 0.5
        
        let ratio = calculate_set_overlap_ratio(&set1, &set2);
        assert!((ratio - 0.667).abs() < 0.01); // 2/3 ≈ 0.667
    }

    #[test]
    fn test_euclidean_distance() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        
        let distance = euclidean_distance(&vec1, &vec2);
        assert!((distance - 5.196).abs() < 0.01); // sqrt(27) ≈ 5.196
    }

    #[test]
    fn test_correlation() {
        let data = vec![
            (1.0, 2.0),
            (2.0, 4.0),
            (3.0, 6.0),
            (4.0, 8.0),
        ];
        
        let correlation = calculate_correlation(&data);
        assert!((correlation - 1.0).abs() < 0.01); // Perfect positive correlation
    }
}