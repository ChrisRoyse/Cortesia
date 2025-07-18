//! Test Reporting System
//! 
//! Comprehensive test result reporting and analysis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use crate::infrastructure::{TestResult, PerformanceMetrics, ReportWriter};

/// Individual test summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub name: String,
    pub status: String,
    pub duration: Duration,
    pub memory_usage: u64,
    pub error_message: Option<String>,
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Complete test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub total_duration: Duration,
    pub coverage_percentage: f64,
    pub test_summaries: Vec<TestSummary>,
    pub performance_summary: PerformanceSummary,
    pub environment_info: EnvironmentInfo,
}

/// Performance summary across all tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub average_test_duration: Duration,
    pub slowest_test: Option<String>,
    pub fastest_test: Option<String>,
    pub total_memory_usage: u64,
    pub peak_memory_usage: u64,
    pub cpu_usage_percentage: f64,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub platform: String,
    pub rust_version: String,
    pub cpu_count: usize,
    pub total_memory: u64,
    pub test_runner_version: String,
}

/// Configuration for report generation
#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub output_directory: std::path::PathBuf,
    pub generate_html: bool,
    pub generate_json: bool,
    pub generate_junit_xml: bool,
    pub include_performance_details: bool,
    pub include_coverage_details: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            output_directory: std::path::PathBuf::from("test_reports"),
            generate_html: true,
            generate_json: true,
            generate_junit_xml: true,
            include_performance_details: true,
            include_coverage_details: true,
        }
    }
}

/// Main test reporter
pub struct TestReporter {
    config: ReportConfig,
    writers: Vec<Box<dyn ReportWriter>>,
}

impl TestReporter {
    /// Create a new test reporter
    pub fn new(config: &ReportConfig) -> Result<Self> {
        let mut writers: Vec<Box<dyn ReportWriter>> = Vec::new();

        // Add configured report writers
        if config.generate_html {
            writers.push(Box::new(crate::infrastructure::HtmlReportWriter::new(
                config.output_directory.join("report.html")
            )?));
        }

        if config.generate_json {
            writers.push(Box::new(crate::infrastructure::JsonReportWriter::new(
                config.output_directory.join("report.json")
            )?));
        }

        if config.generate_junit_xml {
            writers.push(Box::new(crate::infrastructure::JunitXmlWriter::new(
                config.output_directory.join("junit.xml")
            )?));
        }

        Ok(Self {
            config: config.clone(),
            writers,
        })
    }

    /// Generate a comprehensive test report from results
    pub async fn generate_report(&self, results: &[TestResult]) -> Result<TestReport> {
        let timestamp = chrono::Utc::now();
        
        // Calculate basic statistics
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = results.iter().filter(|r| !r.passed).count();
        let skipped_tests = 0; // TODO: Add skipped test support
        
        let total_duration = results.iter()
            .map(|r| r.duration)
            .sum();

        // Calculate coverage
        let coverage_percentage = if total_tests > 0 {
            results.iter()
                .filter_map(|r| r.coverage_percentage)
                .sum::<f64>() / total_tests as f64
        } else {
            0.0
        };

        // Create test summaries
        let test_summaries: Vec<TestSummary> = results.iter()
            .map(|r| TestSummary {
                name: r.name.clone(),
                status: if r.passed { "PASSED".to_string() } else { "FAILED".to_string() },
                duration: r.duration,
                memory_usage: r.memory_usage_bytes,
                error_message: r.error_message.clone(),
                performance_metrics: r.performance_metrics.clone(),
            })
            .collect();

        // Calculate performance summary
        let performance_summary = self.calculate_performance_summary(results);

        // Gather environment info
        let environment_info = self.gather_environment_info();

        let report = TestReport {
            timestamp,
            total_tests,
            passed_tests,
            failed_tests,
            skipped_tests,
            total_duration,
            coverage_percentage,
            test_summaries,
            performance_summary,
            environment_info,
        };

        Ok(report)
    }

    /// Generate all configured report formats
    pub async fn generate_all_reports(&self, results: &[TestResult]) -> Result<()> {
        let report = self.generate_report(results).await?;

        // Ensure output directory exists
        tokio::fs::create_dir_all(&self.config.output_directory).await?;

        // Generate reports using all configured writers
        for writer in &self.writers {
            if let Err(e) = writer.write_report(&report).await {
                log::error!("Failed to generate report: {}", e);
            }
        }

        Ok(())
    }

    /// Calculate performance summary from results
    fn calculate_performance_summary(&self, results: &[TestResult]) -> PerformanceSummary {
        if results.is_empty() {
            return PerformanceSummary {
                average_test_duration: Duration::from_secs(0),
                slowest_test: None,
                fastest_test: None,
                total_memory_usage: 0,
                peak_memory_usage: 0,
                cpu_usage_percentage: 0.0,
            };
        }

        let total_duration: Duration = results.iter().map(|r| r.duration).sum();
        let average_test_duration = total_duration / results.len() as u32;

        let slowest_test = results.iter()
            .max_by_key(|r| r.duration)
            .map(|r| r.name.clone());

        let fastest_test = results.iter()
            .min_by_key(|r| r.duration)
            .map(|r| r.name.clone());

        let total_memory_usage: u64 = results.iter()
            .map(|r| r.memory_usage_bytes)
            .sum();

        let peak_memory_usage = results.iter()
            .map(|r| r.memory_usage_bytes)
            .max()
            .unwrap_or(0);

        let cpu_usage_percentage = results.iter()
            .filter_map(|r| r.performance_metrics.as_ref())
            .map(|m| m.cpu_usage_percent)
            .sum::<f64>() / results.len() as f64;

        PerformanceSummary {
            average_test_duration,
            slowest_test,
            fastest_test,
            total_memory_usage,
            peak_memory_usage,
            cpu_usage_percentage,
        }
    }

    /// Gather environment information
    fn gather_environment_info(&self) -> EnvironmentInfo {
        use sysinfo::{System, SystemExt};
        
        let system = System::new_all();
        
        EnvironmentInfo {
            platform: std::env::consts::OS.to_string(),
            rust_version: env!("RUSTC_VERSION").to_string(),
            cpu_count: num_cpus::get(),
            total_memory: system.total_memory(),
            test_runner_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{TestResult, TestStatus};

    #[tokio::test]
    async fn test_report_generation() {
        let results = vec![
            TestResult {
                name: "test1".to_string(),
                status: TestStatus::Passed,
                passed: true,
                duration: Duration::from_millis(100),
                memory_usage_bytes: 1024,
                error_message: None,
                coverage_percentage: Some(95.0),
                performance_metrics: None,
            },
            TestResult {
                name: "test2".to_string(),
                status: TestStatus::Failed,
                passed: false,
                duration: Duration::from_millis(200),
                memory_usage_bytes: 2048,
                error_message: Some("Test failed".to_string()),
                coverage_percentage: Some(80.0),
                performance_metrics: None,
            },
        ];

        let config = ReportConfig::default();
        let reporter = TestReporter::new(&config).unwrap();
        
        let report = reporter.generate_report(&results).await.unwrap();
        
        assert_eq!(report.total_tests, 2);
        assert_eq!(report.passed_tests, 1);
        assert_eq!(report.failed_tests, 1);
        assert_eq!(report.coverage_percentage, 87.5);
    }

    #[test]
    fn test_performance_summary_calculation() {
        let config = ReportConfig::default();
        let reporter = TestReporter::new(&config).unwrap();

        let results = vec![
            TestResult {
                name: "fast_test".to_string(),
                status: TestStatus::Passed,
                passed: true,
                duration: Duration::from_millis(50),
                memory_usage_bytes: 1000,
                error_message: None,
                coverage_percentage: Some(90.0),
                performance_metrics: None,
            },
            TestResult {
                name: "slow_test".to_string(),
                status: TestStatus::Passed,
                passed: true,
                duration: Duration::from_millis(300),
                memory_usage_bytes: 3000,
                error_message: None,
                coverage_percentage: Some(85.0),
                performance_metrics: None,
            },
        ];

        let summary = reporter.calculate_performance_summary(&results);
        
        assert_eq!(summary.average_test_duration, Duration::from_millis(175));
        assert_eq!(summary.slowest_test, Some("slow_test".to_string()));
        assert_eq!(summary.fastest_test, Some("fast_test".to_string()));
        assert_eq!(summary.total_memory_usage, 4000);
        assert_eq!(summary.peak_memory_usage, 3000);
    }
}