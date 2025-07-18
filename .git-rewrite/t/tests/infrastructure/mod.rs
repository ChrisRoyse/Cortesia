//! LLMKG Simulation Infrastructure
//! 
//! This module provides comprehensive testing infrastructure for the LLMKG system,
//! including test orchestration, deterministic environment setup, performance 
//! measurement, data management, and reporting capabilities.

pub mod test_registry;
pub mod execution_engine;
pub mod config;
pub mod deterministic_rng;
pub mod time_control;
pub mod isolation;
pub mod metrics;
pub mod performance_monitor;
pub mod performance_db;
pub mod data_registry;
pub mod data_generation;
pub mod data_cache;
pub mod reporting;
pub mod report_writers;
pub mod dashboard;
pub mod ci_integration;

// Re-export core types for easy access
pub use test_registry::{TestRegistry, TestDescriptor, TestCategory};
pub use execution_engine::{TestExecutionEngine, TestResult, TestStatus};
pub use config::{TestConfig, PerformanceTargets};
pub use deterministic_rng::DeterministicRng;
pub use time_control::ControlledTime;
pub use isolation::{TestEnvironment, ResourceLimits};
pub use metrics::{PerformanceMetrics, LatencyStats, MemoryStats};
pub use performance_monitor::PerformanceMonitor;
pub use performance_db::{PerformanceDatabase, BaselineMetric};
pub use data_registry::{TestDataRegistry, DatasetDescriptor, DataProperties};
pub use data_generation::{DataGenerator, GenerationParams, DataSize};
pub use data_cache::{DataCache, CacheEntry};
pub use reporting::{TestReporter, TestReport, TestSummary};
pub use report_writers::{ReportWriter, HtmlReportWriter, JsonReportWriter, JunitXmlWriter};
pub use dashboard::{TestDashboard, DashboardMetrics};
pub use ci_integration::{CiIntegration, CiPlatform};

use anyhow::Result;
use std::sync::Arc;

/// Main simulation infrastructure manager
pub struct SimulationInfrastructure {
    registry: Arc<TestRegistry>,
    executor: Arc<TestExecutionEngine>,
    config: TestConfig,
    data_registry: Arc<TestDataRegistry>,
    performance_db: Arc<PerformanceDatabase>,
    reporter: Arc<TestReporter>,
}

impl SimulationInfrastructure {
    /// Initialize the complete simulation infrastructure
    pub async fn new(config: TestConfig) -> Result<Self> {
        let registry = Arc::new(TestRegistry::new()?);
        let data_registry = Arc::new(TestDataRegistry::new(&config).await?);
        let performance_db = Arc::new(PerformanceDatabase::new(&config.performance_db_path).await?);
        let reporter = Arc::new(TestReporter::new(&config.reporting_config)?);
        
        let executor = Arc::new(TestExecutionEngine::new(
            registry.clone(),
            data_registry.clone(),
            performance_db.clone(),
            reporter.clone(),
            config.clone(),
        )?);

        Ok(Self {
            registry,
            executor,
            config,
            data_registry,
            performance_db,
            reporter,
        })
    }

    /// Discover all available tests
    pub async fn discover_tests(&self) -> Result<Vec<TestDescriptor>> {
        self.registry.discover_tests().await
    }

    /// Execute a specific test suite
    pub async fn execute_test_suite(&self, suite_name: &str) -> Result<TestReport> {
        self.executor.execute_suite(suite_name).await
    }

    /// Execute all tests
    pub async fn execute_all_tests(&self) -> Result<TestReport> {
        self.executor.execute_all().await
    }

    /// Generate test data for specific scenarios
    pub async fn generate_test_data(&self, params: &GenerationParams) -> Result<String> {
        self.data_registry.generate_data(params).await
    }

    /// Get performance baselines for comparison
    pub async fn get_performance_baselines(&self) -> Result<Vec<BaselineMetric>> {
        self.performance_db.get_all_baselines().await
    }

    /// Update performance baselines with new results
    pub async fn update_baselines(&self, results: &TestReport) -> Result<()> {
        self.performance_db.update_baselines(results).await
    }

    /// Generate comprehensive test report
    pub async fn generate_report(&self, results: &TestReport) -> Result<()> {
        self.reporter.generate_all_reports(results).await
    }

    /// Start real-time dashboard (if enabled)
    pub async fn start_dashboard(&self) -> Result<()> {
        if self.config.dashboard_enabled {
            let dashboard = TestDashboard::new(&self.config.dashboard_config)?;
            dashboard.start().await?;
        }
        Ok(())
    }

    /// Cleanup temporary resources
    pub async fn cleanup(&self) -> Result<()> {
        self.data_registry.cleanup().await?;
        self.performance_db.close().await?;
        Ok(())
    }
}

/// Convenience function to initialize infrastructure with default config
pub async fn init_default_infrastructure() -> Result<SimulationInfrastructure> {
    let config = TestConfig::default();
    SimulationInfrastructure::new(config).await
}

/// Convenience function to run a complete test simulation
pub async fn run_simulation(config: TestConfig) -> Result<TestReport> {
    let infrastructure = SimulationInfrastructure::new(config).await?;
    
    // Start dashboard if enabled
    infrastructure.start_dashboard().await?;
    
    // Discover tests
    let _tests = infrastructure.discover_tests().await?;
    
    // Execute all tests
    let report = infrastructure.execute_all_tests().await?;
    
    // Update baselines
    infrastructure.update_baselines(&report).await?;
    
    // Generate reports
    infrastructure.generate_report(&report).await?;
    
    // Cleanup
    infrastructure.cleanup().await?;
    
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_infrastructure_initialization() {
        let config = TestConfig::default();
        let infrastructure = SimulationInfrastructure::new(config).await;
        assert!(infrastructure.is_ok());
    }

    #[tokio::test]
    async fn test_default_infrastructure() {
        let infrastructure = init_default_infrastructure().await;
        assert!(infrastructure.is_ok());
    }
}