//! E2E Simulation Environment
//! 
//! Core simulation environment that provides the foundation for all end-to-end testing scenarios.

use crate::data_generation::ComprehensiveDataGenerator;
use crate::infrastructure::{PerformanceMonitor, TestConfig};
use super::data_generators::E2EDataGenerator;
use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main simulation environment for E2E testing
pub struct E2ESimulationEnvironment {
    pub name: String,
    pub data_generator: E2EDataGenerator,
    pub performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    pub workflow_results: HashMap<String, WorkflowResult>,
    pub start_time: Instant,
    pub config: SimulationConfig,
}

impl E2ESimulationEnvironment {
    /// Create a new simulation environment
    pub async fn new(name: &str) -> Result<Self> {
        let seed = 42; // Deterministic seed for reproducible results
        let data_generator = E2EDataGenerator::new(seed);
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));
        
        Ok(Self {
            name: name.to_string(),
            data_generator,
            performance_monitor,
            workflow_results: HashMap::new(),
            start_time: Instant::now(),
            config: SimulationConfig::default(),
        })
    }

    /// Create a new simulation environment from String name (for convenience)
    pub fn new(name: String) -> Self {
        let seed = 42; // Deterministic seed for reproducible results
        let data_generator = E2EDataGenerator::new(seed);
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));
        
        Self {
            name,
            data_generator,
            performance_monitor,
            workflow_results: HashMap::new(),
            start_time: Instant::now(),
            config: SimulationConfig::default(),
        }
    }

    /// Record the result of a workflow simulation
    pub fn record_workflow_result(&mut self, workflow_name: &str, result: WorkflowResult) {
        self.workflow_results.insert(workflow_name.to_string(), result);
    }

    /// Get performance summary for the simulation environment
    pub async fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        let monitor = self.performance_monitor.read().await;
        let total_duration = self.start_time.elapsed();
        
        Ok(PerformanceSummary {
            total_simulation_time: total_duration,
            total_workflows: self.workflow_results.len(),
            average_workflow_time: self.calculate_average_workflow_time(),
            memory_usage_mb: monitor.get_memory_usage_mb(),
            success_rate: self.calculate_success_rate(),
        })
    }

    /// Generate test report for this simulation environment
    pub fn generate_test_report(&self) -> E2ETestReport {
        E2ETestReport {
            environment_name: self.name.clone(),
            total_duration: self.start_time.elapsed(),
            workflow_results: self.workflow_results.clone(),
            overall_success: self.workflow_results.values().all(|r| r.success),
            quality_scores: self.calculate_quality_scores(),
            performance_metrics: self.calculate_performance_metrics(),
        }
    }

    /// Clean up simulation environment resources
    pub async fn cleanup(&mut self) -> Result<()> {
        // Reset workflow results
        self.workflow_results.clear();
        
        // Reset performance monitor
        let mut monitor = self.performance_monitor.write().await;
        monitor.reset();
        
        Ok(())
    }

    // Private helper methods

    fn calculate_average_workflow_time(&self) -> Duration {
        if self.workflow_results.is_empty() {
            return Duration::from_secs(0);
        }

        let total_time: Duration = self.workflow_results.values()
            .map(|r| r.total_time)
            .sum();
        
        total_time / self.workflow_results.len() as u32
    }

    fn calculate_success_rate(&self) -> f64 {
        if self.workflow_results.is_empty() {
            return 1.0;
        }

        let successful_workflows = self.workflow_results.values()
            .filter(|r| r.success)
            .count();
        
        successful_workflows as f64 / self.workflow_results.len() as f64
    }

    fn calculate_quality_scores(&self) -> HashMap<String, f64> {
        let mut quality_scores = HashMap::new();
        
        for (workflow_name, result) in &self.workflow_results {
            let avg_quality = if result.quality_scores.is_empty() {
                0.0
            } else {
                result.quality_scores.iter()
                    .map(|(_, score)| *score)
                    .sum::<f64>() / result.quality_scores.len() as f64
            };
            
            quality_scores.insert(workflow_name.clone(), avg_quality);
        }
        
        quality_scores
    }

    fn calculate_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Calculate average workflow execution time
        if !self.workflow_results.is_empty() {
            let avg_time = self.calculate_average_workflow_time().as_secs_f64();
            metrics.insert("avg_workflow_time_seconds".to_string(), avg_time);
        }

        // Calculate success rate
        metrics.insert("success_rate".to_string(), self.calculate_success_rate());

        // Calculate workflows per minute
        let total_minutes = self.start_time.elapsed().as_secs_f64() / 60.0;
        if total_minutes > 0.0 {
            let workflows_per_minute = self.workflow_results.len() as f64 / total_minutes;
            metrics.insert("workflows_per_minute".to_string(), workflows_per_minute);
        }

        metrics
    }
}

/// Configuration for simulation environment
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub timeout_duration: Duration,
    pub max_memory_usage_mb: f64,
    pub performance_targets: PerformanceTargets,
    pub quality_thresholds: QualityThresholds,
}

/// Performance targets for validation
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub max_workflow_time: Duration,
    pub min_throughput: f64,
    pub max_memory_usage: f64,
    pub min_success_rate: f64,
}

/// Quality thresholds for validation
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_workflow_quality: f64,
    pub min_content_coherence: f64,
    pub min_fact_accuracy: f64,
    pub min_relevance_score: f64,
}

/// Result of a workflow execution
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    pub success: bool,
    pub total_time: Duration,
    pub quality_scores: Vec<(String, f64)>,
    pub performance_metrics: Vec<(String, f64)>,
}

/// Performance summary for simulation environment
#[derive(Debug)]
pub struct PerformanceSummary {
    pub total_simulation_time: Duration,
    pub total_workflows: usize,
    pub average_workflow_time: Duration,
    pub memory_usage_mb: f64,
    pub success_rate: f64,
}

/// Complete test report for E2E simulation
#[derive(Debug)]
pub struct E2ETestReport {
    pub environment_name: String,
    pub total_duration: Duration,
    pub workflow_results: HashMap<String, WorkflowResult>,
    pub overall_success: bool,
    pub quality_scores: HashMap<String, f64>,
    pub performance_metrics: HashMap<String, f64>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            timeout_duration: Duration::from_minutes(30),
            max_memory_usage_mb: 4000.0,
            performance_targets: PerformanceTargets {
                max_workflow_time: Duration::from_minutes(5),
                min_throughput: 10.0,
                max_memory_usage: 2000.0,
                min_success_rate: 0.95,
            },
            quality_thresholds: QualityThresholds {
                min_workflow_quality: 0.8,
                min_content_coherence: 0.7,
                min_fact_accuracy: 0.85,
                min_relevance_score: 0.75,
            },
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_workflow_time: Duration::from_minutes(5),
            min_throughput: 10.0,
            max_memory_usage: 2000.0,
            min_success_rate: 0.95,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_workflow_quality: 0.8,
            min_content_coherence: 0.7,
            min_fact_accuracy: 0.85,
            min_relevance_score: 0.75,
        }
    }
}

impl E2ETestReport {
    /// Check if the simulation meets performance targets
    pub fn meets_performance_targets(&self, targets: &PerformanceTargets) -> bool {
        let avg_workflow_time = self.calculate_average_workflow_time();
        let success_rate = self.calculate_success_rate();
        
        avg_workflow_time <= targets.max_workflow_time &&
        success_rate >= targets.min_success_rate
    }

    /// Check if the simulation meets quality thresholds
    pub fn meets_quality_thresholds(&self, thresholds: &QualityThresholds) -> bool {
        let avg_quality = self.calculate_average_quality_score();
        avg_quality >= thresholds.min_workflow_quality
    }

    /// Generate summary text for the report
    pub fn generate_summary(&self) -> String {
        format!(
            "E2E Simulation Report: {}\n\
            Duration: {:?}\n\
            Workflows: {}\n\
            Success Rate: {:.2}%\n\
            Average Quality: {:.3}\n\
            Overall Success: {}",
            self.environment_name,
            self.total_duration,
            self.workflow_results.len(),
            self.calculate_success_rate() * 100.0,
            self.calculate_average_quality_score(),
            self.overall_success
        )
    }

    // Private helper methods

    fn calculate_average_workflow_time(&self) -> Duration {
        if self.workflow_results.is_empty() {
            return Duration::from_secs(0);
        }

        let total_time: Duration = self.workflow_results.values()
            .map(|r| r.total_time)
            .sum();
        
        total_time / self.workflow_results.len() as u32
    }

    fn calculate_success_rate(&self) -> f64 {
        if self.workflow_results.is_empty() {
            return 1.0;
        }

        let successful_workflows = self.workflow_results.values()
            .filter(|r| r.success)
            .count();
        
        successful_workflows as f64 / self.workflow_results.len() as f64
    }

    fn calculate_average_quality_score(&self) -> f64 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }

        self.quality_scores.values().sum::<f64>() / self.quality_scores.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simulation_environment_creation() {
        let env = E2ESimulationEnvironment::new("test_env").await;
        assert!(env.is_ok());
        
        let env = env.unwrap();
        assert_eq!(env.name, "test_env");
        assert!(env.workflow_results.is_empty());
    }

    #[tokio::test]
    async fn test_workflow_result_recording() {
        let mut env = E2ESimulationEnvironment::new("test_env").await.unwrap();
        
        let workflow_result = WorkflowResult {
            success: true,
            total_time: Duration::from_secs(10),
            quality_scores: vec![("coherence".to_string(), 0.8)],
            performance_metrics: vec![("latency".to_string(), 5.0)],
        };

        env.record_workflow_result("test_workflow", workflow_result);
        
        assert_eq!(env.workflow_results.len(), 1);
        assert!(env.workflow_results.contains_key("test_workflow"));
    }

    #[tokio::test]
    async fn test_performance_summary_generation() {
        let mut env = E2ESimulationEnvironment::new("test_env").await.unwrap();
        
        let workflow_result = WorkflowResult {
            success: true,
            total_time: Duration::from_secs(5),
            quality_scores: vec![("quality".to_string(), 0.9)],
            performance_metrics: vec![("speed".to_string(), 10.0)],
        };

        env.record_workflow_result("test_workflow", workflow_result);
        
        let summary = env.get_performance_summary().await.unwrap();
        assert_eq!(summary.total_workflows, 1);
        assert_eq!(summary.success_rate, 1.0);
    }

    #[tokio::test]
    async fn test_test_report_generation() {
        let mut env = E2ESimulationEnvironment::new("test_env").await.unwrap();
        
        let workflow_result = WorkflowResult {
            success: true,
            total_time: Duration::from_secs(3),
            quality_scores: vec![("coherence".to_string(), 0.85)],
            performance_metrics: vec![("throughput".to_string(), 15.0)],
        };

        env.record_workflow_result("test_workflow", workflow_result);
        
        let report = env.generate_test_report();
        assert_eq!(report.environment_name, "test_env");
        assert!(report.overall_success);
        assert_eq!(report.workflow_results.len(), 1);
    }

    #[test]
    fn test_default_configurations() {
        let config = SimulationConfig::default();
        assert_eq!(config.timeout_duration, Duration::from_minutes(30));
        assert_eq!(config.max_memory_usage_mb, 4000.0);
        
        let targets = PerformanceTargets::default();
        assert_eq!(targets.max_workflow_time, Duration::from_minutes(5));
        assert_eq!(targets.min_success_rate, 0.95);
        
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.min_workflow_quality, 0.8);
        assert_eq!(thresholds.min_fact_accuracy, 0.85);
    }

    #[test]
    fn test_report_validation() {
        let mut workflow_results = HashMap::new();
        workflow_results.insert("test1".to_string(), WorkflowResult {
            success: true,
            total_time: Duration::from_secs(2),
            quality_scores: vec![("quality".to_string(), 0.9)],
            performance_metrics: vec![("metric".to_string(), 1.0)],
        });

        let report = E2ETestReport {
            environment_name: "test".to_string(),
            total_duration: Duration::from_secs(10),
            workflow_results,
            overall_success: true,
            quality_scores: [("test1".to_string(), 0.9)].into_iter().collect(),
            performance_metrics: [("avg_time".to_string(), 2.0)].into_iter().collect(),
        };

        let targets = PerformanceTargets::default();
        let thresholds = QualityThresholds::default();

        assert!(report.meets_performance_targets(&targets));
        assert!(report.meets_quality_thresholds(&thresholds));
    }

    #[test]
    fn test_report_summary_generation() {
        let workflow_results = HashMap::new();
        let report = E2ETestReport {
            environment_name: "summary_test".to_string(),
            total_duration: Duration::from_secs(60),
            workflow_results,
            overall_success: true,
            quality_scores: HashMap::new(),
            performance_metrics: HashMap::new(),
        };

        let summary = report.generate_summary();
        assert!(summary.contains("summary_test"));
        assert!(summary.contains("Duration"));
        assert!(summary.contains("Success Rate"));
    }
}