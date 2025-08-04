# Task 081: Create CI/CD Integration Hooks

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The CI/CD Integration Hooks provide seamless integration with continuous integration systems, automated testing workflows, and deployment pipelines.

## Project Structure
```
src/
  validation/
    ci_hooks.rs        <- Create this file
  lib.rs
.github/
  workflows/
    validation.yml     <- Create this file
scripts/
  validate.sh          <- Create this file  
  validate.ps1         <- Create this file
```

## Task Description
Create CI/CD integration components that enable automated validation in continuous integration pipelines with proper exit codes, artifact generation, and integration with popular CI systems.

## Requirements
1. Create `src/validation/ci_hooks.rs` with CI integration logic
2. Create GitHub Actions workflow configuration
3. Create shell scripts for easy CI integration
4. Implement proper exit codes and status reporting
5. Add artifact generation for CI systems

## Expected Code Structure

### `src/validation/ci_hooks.rs`
```rust
use anyhow::{Result, Context};
use std::process::{Command, ExitCode};
use std::path::{Path, PathBuf};
use std::env;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

use crate::validation::{
    pipeline::{ValidationPipeline, PipelineConfig},
    report::ValidationReport,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiConfig {
    pub fail_on_accuracy_threshold: f64,
    pub fail_on_performance_threshold: f64,
    pub fail_on_security_issues: bool,
    pub generate_artifacts: bool,
    pub artifact_retention_days: u32,
    pub parallel_jobs: Option<usize>,
    pub timeout_minutes: u64,
    pub quiet_mode: bool,
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            fail_on_accuracy_threshold: 95.0,
            fail_on_performance_threshold: 80.0,
            fail_on_security_issues: true,
            generate_artifacts: true,
            artifact_retention_days: 30,
            parallel_jobs: None, // Use system default
            timeout_minutes: 60,
            quiet_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiResult {
    pub success: bool,
    pub exit_code: i32,
    pub summary_message: String,
    pub detailed_report_path: Option<PathBuf>,
    pub artifacts: Vec<PathBuf>,
    pub metrics: CiMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiMetrics {
    pub overall_score: f64,
    pub accuracy_percentage: f64,
    pub performance_score: f64,
    pub security_passed: bool,
    pub total_test_cases: usize,
    pub failed_test_cases: usize,
    pub execution_time_seconds: f64,
}

#[derive(Debug)]
pub enum CiExitCode {
    Success = 0,
    ValidationFailed = 1,
    ConfigurationError = 2,
    SystemError = 3,
    TimeoutError = 4,
    SecurityFailure = 5,
    PerformanceFailure = 6,
    AccuracyFailure = 7,
}

impl From<CiExitCode> for ExitCode {
    fn from(code: CiExitCode) -> Self {
        ExitCode::from(code as u8)
    }
}

pub struct CiHooks {
    config: CiConfig,
    pipeline_config: PipelineConfig,
}

impl CiHooks {
    pub fn new(ci_config: CiConfig, pipeline_config: PipelineConfig) -> Self {
        Self {
            config: ci_config,
            pipeline_config,
        }
    }
    
    pub fn from_environment() -> Result<Self> {
        let ci_config = Self::load_ci_config_from_env()?;
        let pipeline_config = Self::load_pipeline_config_from_env()?;
        
        Ok(Self::new(ci_config, pipeline_config))
    }
    
    fn load_ci_config_from_env() -> Result<CiConfig> {
        let mut config = CiConfig::default();
        
        if let Ok(threshold) = env::var("LLMKG_ACCURACY_THRESHOLD") {
            config.fail_on_accuracy_threshold = threshold.parse()
                .context("Invalid LLMKG_ACCURACY_THRESHOLD value")?;
        }
        
        if let Ok(threshold) = env::var("LLMKG_PERFORMANCE_THRESHOLD") {
            config.fail_on_performance_threshold = threshold.parse()
                .context("Invalid LLMKG_PERFORMANCE_THRESHOLD value")?;
        }
        
        if let Ok(value) = env::var("LLMKG_FAIL_ON_SECURITY") {
            config.fail_on_security_issues = value.parse()
                .context("Invalid LLMKG_FAIL_ON_SECURITY value")?;
        }
        
        if let Ok(jobs) = env::var("LLMKG_PARALLEL_JOBS") {
            config.parallel_jobs = Some(jobs.parse()
                .context("Invalid LLMKG_PARALLEL_JOBS value")?);
        }
        
        if let Ok(timeout) = env::var("LLMKG_TIMEOUT_MINUTES") {
            config.timeout_minutes = timeout.parse()
                .context("Invalid LLMKG_TIMEOUT_MINUTES value")?;
        }
        
        config.quiet_mode = env::var("LLMKG_QUIET").is_ok();
        
        Ok(config)
    }
    
    fn load_pipeline_config_from_env() -> Result<PipelineConfig> {
        let mut config = PipelineConfig::default();
        
        // Load pipeline configuration from environment variables
        if let Ok(output_dir) = env::var("LLMKG_OUTPUT_DIR") {
            config.validation_config.output_dir = PathBuf::from(output_dir);
        }
        
        if let Ok(ground_truth) = env::var("LLMKG_GROUND_TRUTH_PATH") {
            config.validation_config.ground_truth_path = PathBuf::from(ground_truth);
        }
        
        if let Ok(test_data) = env::var("LLMKG_TEST_DATA_PATH") {
            config.validation_config.test_data_path = PathBuf::from(test_data);
        }
        
        // Override parallel jobs if specified in CI config
        if let Ok(jobs) = env::var("LLMKG_PARALLEL_JOBS") {
            if let Ok(job_count) = jobs.parse::<usize>() {
                config.parallel_config.max_concurrent_tests = job_count;
            }
        }
        
        Ok(config)
    }
    
    pub async fn run_validation(&mut self) -> Result<CiResult> {
        info!("Starting CI validation run");
        
        let start_time = std::time::Instant::now();
        
        // Set up pipeline timeout
        let timeout_duration = std::time::Duration::from_secs(self.config.timeout_minutes * 60);
        
        // Run validation pipeline with timeout
        let validation_result = tokio::time::timeout(
            timeout_duration,
            self.execute_validation_pipeline()
        ).await;
        
        let execution_time = start_time.elapsed();
        
        match validation_result {
            Ok(Ok(report)) => {
                self.process_successful_validation(report, execution_time).await
            }
            Ok(Err(e)) => {
                error!("Validation pipeline failed: {}", e);
                Ok(CiResult {
                    success: false,
                    exit_code: CiExitCode::ValidationFailed as i32,
                    summary_message: format!("Validation failed: {}", e),
                    detailed_report_path: None,
                    artifacts: Vec::new(),
                    metrics: CiMetrics::default_failed(),
                })
            }
            Err(_) => {
                error!("Validation pipeline timed out after {} minutes", self.config.timeout_minutes);
                Ok(CiResult {
                    success: false,
                    exit_code: CiExitCode::TimeoutError as i32,
                    summary_message: format!("Validation timed out after {} minutes", self.config.timeout_minutes),
                    detailed_report_path: None,
                    artifacts: Vec::new(),
                    metrics: CiMetrics::default_failed(),
                })
            }
        }
    }
    
    async fn execute_validation_pipeline(&mut self) -> Result<ValidationReport> {
        let mut pipeline = ValidationPipeline::new(self.pipeline_config.clone());
        pipeline.execute().await
    }
    
    async fn process_successful_validation(
        &self,
        report: ValidationReport,
        execution_time: std::time::Duration,
    ) -> Result<CiResult> {
        let metrics = CiMetrics {
            overall_score: report.overall_score,
            accuracy_percentage: report.accuracy_metrics.overall_accuracy,
            performance_score: if report.performance_metrics.meets_targets { 100.0 } else { 50.0 },
            security_passed: self.all_security_tests_passed(&report),
            total_test_cases: report.metadata.total_test_cases,
            failed_test_cases: self.count_failed_tests(&report),
            execution_time_seconds: execution_time.as_secs_f64(),
        };
        
        // Determine success based on thresholds
        let success = self.evaluate_success_criteria(&metrics);
        let exit_code = self.determine_exit_code(&metrics, success);
        let summary_message = self.generate_summary_message(&metrics, success);
        
        // Generate artifacts if enabled
        let mut artifacts = Vec::new();
        if self.config.generate_artifacts {
            artifacts = self.generate_ci_artifacts(&report).await?;
        }
        
        // Output CI-friendly summary
        self.output_ci_summary(&metrics, success);
        
        Ok(CiResult {
            success,
            exit_code: exit_code as i32,
            summary_message,
            detailed_report_path: Some(self.pipeline_config.validation_config.output_dir.join("validation_report.md")),
            artifacts,
            metrics,
        })
    }
    
    fn evaluate_success_criteria(&self, metrics: &CiMetrics) -> bool {
        let accuracy_pass = metrics.accuracy_percentage >= self.config.fail_on_accuracy_threshold;
        let performance_pass = metrics.performance_score >= self.config.fail_on_performance_threshold;
        let security_pass = !self.config.fail_on_security_issues || metrics.security_passed;
        
        accuracy_pass && performance_pass && security_pass
    }
    
    fn determine_exit_code(&self, metrics: &CiMetrics, success: bool) -> CiExitCode {
        if success {
            return CiExitCode::Success;
        }
        
        // Prioritize specific failure types
        if self.config.fail_on_security_issues && !metrics.security_passed {
            CiExitCode::SecurityFailure
        } else if metrics.accuracy_percentage < self.config.fail_on_accuracy_threshold {
            CiExitCode::AccuracyFailure
        } else if metrics.performance_score < self.config.fail_on_performance_threshold {
            CiExitCode::PerformanceFailure
        } else {
            CiExitCode::ValidationFailed
        }
    }
    
    fn generate_summary_message(&self, metrics: &CiMetrics, success: bool) -> String {
        if success {
            format!(
                "✅ Validation PASSED - Score: {:.1}/100, Accuracy: {:.1}%, Performance: {:.1}%, Security: {}",
                metrics.overall_score,
                metrics.accuracy_percentage,
                metrics.performance_score,
                if metrics.security_passed { "PASS" } else { "FAIL" }
            )
        } else {
            let mut issues = Vec::new();
            
            if metrics.accuracy_percentage < self.config.fail_on_accuracy_threshold {
                issues.push(format!("Accuracy {:.1}% < {:.1}%", metrics.accuracy_percentage, self.config.fail_on_accuracy_threshold));
            }
            
            if metrics.performance_score < self.config.fail_on_performance_threshold {
                issues.push(format!("Performance {:.1}% < {:.1}%", metrics.performance_score, self.config.fail_on_performance_threshold));
            }
            
            if self.config.fail_on_security_issues && !metrics.security_passed {
                issues.push("Security tests failed".to_string());
            }
            
            format!(
                "❌ Validation FAILED - Score: {:.1}/100, Issues: {}",
                metrics.overall_score,
                issues.join(", ")
            )
        }
    }
    
    fn output_ci_summary(&self, metrics: &CiMetrics, success: bool) {
        if self.config.quiet_mode {
            return;
        }
        
        println!("=== LLMKG Validation Summary ===");
        println!("Overall Score: {:.1}/100", metrics.overall_score);
        println!("Accuracy: {:.1}%", metrics.accuracy_percentage);
        println!("Performance: {:.1}%", metrics.performance_score);
        println!("Security: {}", if metrics.security_passed { "PASS" } else { "FAIL" });
        println!("Test Cases: {} total, {} failed", metrics.total_test_cases, metrics.failed_test_cases);
        println!("Execution Time: {:.1}s", metrics.execution_time_seconds);
        println!("Result: {}", if success { "PASS ✅" } else { "FAIL ❌" });
        
        // GitHub Actions annotations
        if env::var("GITHUB_ACTIONS").is_ok() {
            if success {
                println!("::notice title=Validation Success::LLMKG validation passed with score {:.1}/100", metrics.overall_score);
            } else {
                println!("::error title=Validation Failed::LLMKG validation failed with score {:.1}/100", metrics.overall_score);
            }
        }
        
        // Azure Pipelines task summaries
        if env::var("AZURE_HTTP_USER_AGENT").is_ok() {
            println!("##vso[task.setvariable variable=ValidationScore]{:.1}", metrics.overall_score);
            println!("##vso[task.setvariable variable=ValidationPassed]{}", success);
        }
        
        // Jenkins build result
        if env::var("JENKINS_URL").is_ok() {
            if !success {
                println!("BUILD UNSTABLE: Validation failed");
            }
        }
    }
    
    async fn generate_ci_artifacts(&self, report: &ValidationReport) -> Result<Vec<PathBuf>> {
        let mut artifacts = Vec::new();
        let output_dir = &self.pipeline_config.validation_config.output_dir;
        
        // Generate JUnit XML report for CI systems
        let junit_path = output_dir.join("validation-results.xml");
        self.generate_junit_report(report, &junit_path).await?;
        artifacts.push(junit_path);
        
        // Generate metrics JSON for CI dashboards
        let metrics_path = output_dir.join("validation-metrics.json");
        self.generate_metrics_json(report, &metrics_path).await?;
        artifacts.push(metrics_path);
        
        // Generate badge data for README
        let badge_path = output_dir.join("validation-badge.json");
        self.generate_badge_data(report, &badge_path).await?;
        artifacts.push(badge_path);
        
        // Generate CSV summary for spreadsheet analysis
        let csv_path = output_dir.join("validation-summary.csv");
        self.generate_csv_summary(report, &csv_path).await?;
        artifacts.push(csv_path);
        
        info!("Generated {} CI artifacts", artifacts.len());
        
        Ok(artifacts)
    }
    
    async fn generate_junit_report(&self, report: &ValidationReport, path: &Path) -> Result<()> {
        let junit_xml = format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="LLMKG Validation" tests="{total_tests}" failures="{failures}" errors="0" time="{execution_time}">
    <testsuite name="Accuracy Tests" tests="{accuracy_tests}" failures="{accuracy_failures}" errors="0" time="{accuracy_time}">
        <testcase name="Overall Accuracy" classname="validation.accuracy" time="1.0">
            {accuracy_result}
        </testcase>
        {query_type_tests}
    </testsuite>
    <testsuite name="Performance Tests" tests="4" failures="{performance_failures}" errors="0" time="{performance_time}">
        <testcase name="Latency P50" classname="validation.performance" time="1.0">
            {latency_p50_result}
        </testcase>
        <testcase name="Latency P95" classname="validation.performance" time="1.0">
            {latency_p95_result}
        </testcase>
        <testcase name="Latency P99" classname="validation.performance" time="1.0">
            {latency_p99_result}
        </testcase>
        <testcase name="Throughput" classname="validation.performance" time="1.0">
            {throughput_result}
        </testcase>
    </testsuite>
    <testsuite name="Security Tests" tests="4" failures="{security_failures}" errors="0" time="{security_time}">
        <testcase name="SQL Injection" classname="validation.security" time="1.0">
            {sql_injection_result}
        </testcase>
        <testcase name="Input Validation" classname="validation.security" time="1.0">
            {input_validation_result}
        </testcase>
        <testcase name="DoS Prevention" classname="validation.security" time="1.0">
            {dos_prevention_result}
        </testcase>
        <testcase name="Malicious Queries" classname="validation.security" time="1.0">
            {malicious_queries_result}
        </testcase>
    </testsuite>
</testsuites>"#,
            total_tests = report.metadata.total_test_cases + 8, // Add performance and security tests
            failures = self.count_failed_tests(report),
            execution_time = report.metadata.test_duration_minutes * 60.0,
            accuracy_tests = report.accuracy_metrics.query_type_results.len() + 1,
            accuracy_failures = if report.accuracy_metrics.overall_accuracy >= self.config.fail_on_accuracy_threshold { 0 } else { 1 },
            accuracy_time = report.metadata.test_duration_minutes * 20.0, // Estimate 20% of time
            accuracy_result = if report.accuracy_metrics.overall_accuracy >= self.config.fail_on_accuracy_threshold {
                "".to_string()
            } else {
                format!(r#"<failure message="Accuracy {:.1}% below threshold {:.1}%"/>"#, 
                    report.accuracy_metrics.overall_accuracy, self.config.fail_on_accuracy_threshold)
            },
            query_type_tests = self.generate_query_type_junit_tests(report),
            performance_failures = if report.performance_metrics.meets_targets { 0 } else { 1 },
            performance_time = report.metadata.test_duration_minutes * 30.0, // Estimate 30% of time
            latency_p50_result = self.generate_performance_junit_result("P50", report.performance_metrics.latency_metrics.p50_ms, report.performance_metrics.latency_metrics.target_p50_ms),
            latency_p95_result = self.generate_performance_junit_result("P95", report.performance_metrics.latency_metrics.p95_ms, report.performance_metrics.latency_metrics.target_p95_ms),
            latency_p99_result = self.generate_performance_junit_result("P99", report.performance_metrics.latency_metrics.p99_ms, report.performance_metrics.latency_metrics.target_p99_ms),
            throughput_result = self.generate_throughput_junit_result(&report.performance_metrics.throughput_metrics),
            security_failures = self.count_security_failures(report),
            security_time = report.metadata.test_duration_minutes * 10.0, // Estimate 10% of time
            sql_injection_result = self.generate_security_junit_result(&report.security_audit.sql_injection_tests),
            input_validation_result = self.generate_security_junit_result(&report.security_audit.input_validation_tests),
            dos_prevention_result = self.generate_security_junit_result(&report.security_audit.dos_prevention_tests),
            malicious_queries_result = self.generate_security_junit_result(&report.security_audit.malicious_query_tests),
        );
        
        std::fs::write(path, junit_xml)?;
        Ok(())
    }
    
    fn generate_query_type_junit_tests(&self, report: &ValidationReport) -> String {
        let mut tests = String::new();
        
        for (query_type, result) in &report.accuracy_metrics.query_type_results {
            let passed = result.accuracy_percentage >= 90.0; // Per-query-type threshold
            
            if passed {
                tests.push_str(&format!(
                    r#"        <testcase name="{}" classname="validation.accuracy.query_type" time="1.0"/>"#,
                    query_type
                ));
            } else {
                tests.push_str(&format!(
                    r#"        <testcase name="{}" classname="validation.accuracy.query_type" time="1.0">
            <failure message="Query type accuracy {:.1}% below 90%"/>
        </testcase>"#,
                    query_type, result.accuracy_percentage
                ));
            }
            tests.push('\n');
        }
        
        tests
    }
    
    fn generate_performance_junit_result(&self, metric_name: &str, actual: u64, target: u64) -> String {
        if actual <= target {
            "".to_string()
        } else {
            format!(r#"<failure message="{} latency {}ms exceeds target {}ms"/>"#, metric_name, actual, target)
        }
    }
    
    fn generate_throughput_junit_result(&self, throughput: &crate::validation::report::ThroughputMetrics) -> String {
        if throughput.queries_per_second >= throughput.target_qps {
            "".to_string()
        } else {
            format!(r#"<failure message="Throughput {:.1} QPS below target {:.1} QPS"/>"#, 
                throughput.queries_per_second, throughput.target_qps)
        }
    }
    
    fn generate_security_junit_result(&self, test_result: &crate::validation::report::TestResult) -> String {
        if test_result.passed {
            "".to_string()
        } else {
            format!(r#"<failure message="Security test failed: {}"/>"#, test_result.details)
        }
    }
    
    async fn generate_metrics_json(&self, report: &ValidationReport, path: &Path) -> Result<()> {
        let metrics = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "overall_score": report.overall_score,
            "accuracy": {
                "overall": report.accuracy_metrics.overall_accuracy,
                "by_query_type": report.accuracy_metrics.query_type_results,
                "false_positives": report.accuracy_metrics.false_positives_total,
                "false_negatives": report.accuracy_metrics.false_negatives_total
            },
            "performance": {
                "meets_targets": report.performance_metrics.meets_targets,
                "latency": {
                    "p50": report.performance_metrics.latency_metrics.p50_ms,
                    "p95": report.performance_metrics.latency_metrics.p95_ms,
                    "p99": report.performance_metrics.latency_metrics.p99_ms
                },
                "throughput": report.performance_metrics.throughput_metrics.queries_per_second
            },
            "security": {
                "sql_injection": report.security_audit.sql_injection_tests.passed,
                "input_validation": report.security_audit.input_validation_tests.passed,
                "dos_prevention": report.security_audit.dos_prevention_tests.passed,
                "malicious_queries": report.security_audit.malicious_query_tests.passed
            },
            "execution": {
                "duration_minutes": report.metadata.test_duration_minutes,
                "total_test_cases": report.metadata.total_test_cases
            }
        });
        
        std::fs::write(path, serde_json::to_string_pretty(&metrics)?)?;
        Ok(())
    }
    
    async fn generate_badge_data(&self, report: &ValidationReport, path: &Path) -> Result<()> {
        let (color, status) = if report.overall_score >= 95.0 {
            ("brightgreen", "excellent")
        } else if report.overall_score >= 80.0 {
            ("green", "good")
        } else if report.overall_score >= 60.0 {
            ("yellow", "fair")
        } else {
            ("red", "poor")
        };
        
        let badge_data = serde_json::json!({
            "schemaVersion": 1,
            "label": "validation",
            "message": format!("{:.1}/100 ({})", report.overall_score, status),
            "color": color
        });
        
        std::fs::write(path, serde_json::to_string_pretty(&badge_data)?)?;
        Ok(())
    }
    
    async fn generate_csv_summary(&self, report: &ValidationReport, path: &Path) -> Result<()> {
        let mut csv_content = String::new();
        csv_content.push_str("metric,value,unit,status\n");
        csv_content.push_str(&format!("overall_score,{:.1},points,{}\n", 
            report.overall_score, 
            if report.overall_score >= 80.0 { "pass" } else { "fail" }
        ));
        csv_content.push_str(&format!("accuracy,{:.1},percent,{}\n", 
            report.accuracy_metrics.overall_accuracy,
            if report.accuracy_metrics.overall_accuracy >= self.config.fail_on_accuracy_threshold { "pass" } else { "fail" }
        ));
        csv_content.push_str(&format!("performance_meets_targets,{},boolean,{}\n", 
            report.performance_metrics.meets_targets,
            if report.performance_metrics.meets_targets { "pass" } else { "fail" }
        ));
        csv_content.push_str(&format!("security_passed,{},boolean,{}\n", 
            self.all_security_tests_passed(report),
            if self.all_security_tests_passed(report) { "pass" } else { "fail" }
        ));
        csv_content.push_str(&format!("execution_time,{:.1},minutes,info\n", report.metadata.test_duration_minutes));
        csv_content.push_str(&format!("test_cases,{},count,info\n", report.metadata.total_test_cases));
        
        std::fs::write(path, csv_content)?;
        Ok(())
    }
    
    // Helper methods
    fn all_security_tests_passed(&self, report: &ValidationReport) -> bool {
        report.security_audit.sql_injection_tests.passed &&
        report.security_audit.input_validation_tests.passed &&
        report.security_audit.dos_prevention_tests.passed &&
        report.security_audit.malicious_query_tests.passed
    }
    
    fn count_failed_tests(&self, report: &ValidationReport) -> usize {
        let accuracy_failures = if report.accuracy_metrics.overall_accuracy < self.config.fail_on_accuracy_threshold { 1 } else { 0 };
        let performance_failures = if !report.performance_metrics.meets_targets { 1 } else { 0 };
        let security_failures = self.count_security_failures(report);
        
        accuracy_failures + performance_failures + security_failures
    }
    
    fn count_security_failures(&self, report: &ValidationReport) -> usize {
        let mut failures = 0;
        if !report.security_audit.sql_injection_tests.passed { failures += 1; }
        if !report.security_audit.input_validation_tests.passed { failures += 1; }
        if !report.security_audit.dos_prevention_tests.passed { failures += 1; }
        if !report.security_audit.malicious_query_tests.passed { failures += 1; }
        failures
    }
}

impl CiMetrics {
    fn default_failed() -> Self {
        Self {
            overall_score: 0.0,
            accuracy_percentage: 0.0,
            performance_score: 0.0,
            security_passed: false,
            total_test_cases: 0,
            failed_test_cases: 0,
            execution_time_seconds: 0.0,
        }
    }
}

// CLI entry point for CI integration
pub async fn run_ci_validation() -> ExitCode {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    match CiHooks::from_environment() {
        Ok(mut hooks) => {
            match hooks.run_validation().await {
                Ok(result) => {
                    // Print final summary
                    println!("{}", result.summary_message);
                    
                    // Exit with appropriate code
                    ExitCode::from(result.exit_code as u8)
                }
                Err(e) => {
                    eprintln!("CI validation failed: {}", e);
                    CiExitCode::SystemError.into()
                }
            }
        }
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            CiExitCode::ConfigurationError.into()
        }
    }
}
```

### `.github/workflows/validation.yml`
```yaml
name: LLMKG Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run validation daily at 2 AM UTC
    - cron: '0 2 * * *'

env:
  LLMKG_ACCURACY_THRESHOLD: 95.0
  LLMKG_PERFORMANCE_THRESHOLD: 80.0
  LLMKG_FAIL_ON_SECURITY: true
  LLMKG_PARALLEL_JOBS: 4
  LLMKG_TIMEOUT_MINUTES: 60

jobs:
  validation:
    name: Run LLMKG Validation
    runs-on: ubuntu-latest
    timeout-minutes: 90
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        components: clippy, rustfmt
    
    - name: Cache Rust dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        restore-keys: |
          ${{ runner.os }}-cargo-
    
    - name: Build LLMKG
      run: cargo build --release
    
    - name: Setup test environment
      run: |
        mkdir -p tests/data
        mkdir -p target/validation_reports
        
        # Set up environment variables
        echo "LLMKG_OUTPUT_DIR=${{ github.workspace }}/target/validation_reports" >> $GITHUB_ENV
        echo "LLMKG_GROUND_TRUTH_PATH=${{ github.workspace }}/tests/data/ground_truth.json" >> $GITHUB_ENV
        echo "LLMKG_TEST_DATA_PATH=${{ github.workspace }}/tests/data" >> $GITHUB_ENV
    
    - name: Generate test data
      run: |
        # Generate test data if it doesn't exist
        if [ ! -f "tests/data/ground_truth.json" ]; then
          cargo run --bin llmkg -- generate-test-data --output tests/data
        fi
    
    - name: Run LLMKG Validation
      id: validation
      run: |
        cargo run --bin llmkg -- validate \
          --config validation_config.toml \
          --output target/validation_reports
      continue-on-error: true
    
    - name: Upload validation artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: validation-results
        path: |
          target/validation_reports/
        retention-days: 30
    
    - name: Parse validation results
      if: always()
      run: |
        if [ -f "target/validation_reports/validation-metrics.json" ]; then
          SCORE=$(jq -r '.overall_score' target/validation_reports/validation-metrics.json)
          ACCURACY=$(jq -r '.accuracy.overall' target/validation_reports/validation-metrics.json)
          echo "VALIDATION_SCORE=$SCORE" >> $GITHUB_ENV
          echo "VALIDATION_ACCURACY=$ACCURACY" >> $GITHUB_ENV
          
          # Set GitHub step outputs
          echo "score=$SCORE" >> $GITHUB_OUTPUT
          echo "accuracy=$ACCURACY" >> $GITHUB_OUTPUT
        fi
    
    - name: Generate validation badge
      if: always()
      run: |
        if [ -f "target/validation_reports/validation-badge.json" ]; then
          cp target/validation_reports/validation-badge.json validation-badge.json
        fi
    
    - name: Publish test results
      uses: dorny/test-reporter@v1
      if: always()
      with:
        name: LLMKG Validation Results
        path: target/validation_reports/validation-results.xml
        reporter: java-junit
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          
          let comment = '## 🔍 LLMKG Validation Results\n\n';
          
          if (fs.existsSync('target/validation_reports/validation_report.md')) {
            const report = fs.readFileSync('target/validation_reports/validation_report.md', 'utf8');
            const summary = report.split('## Executive Summary')[1]?.split('##')[0] || 'Report summary not available';
            comment += summary;
          }
          
          comment += '\n\n📊 **Metrics:**\n';
          comment += `- Overall Score: ${process.env.VALIDATION_SCORE || 'N/A'}/100\n`;
          comment += `- Accuracy: ${process.env.VALIDATION_ACCURACY || 'N/A'}%\n`;
          
          comment += '\n\n📁 **Artifacts:** [Download validation results](https://github.com/' + 
                     context.repo.owner + '/' + context.repo.repo + '/actions/runs/' + 
                     context.runId + '/artifacts)\n';
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
    
    - name: Fail if validation failed
      if: steps.validation.outcome == 'failure'
      run: |
        echo "❌ LLMKG validation failed"
        exit 1

  performance-regression:
    name: Performance Regression Check
    runs-on: ubuntu-latest
    needs: validation
    if: github.event_name == 'pull_request'
    
    steps:
    - name: Download current results
      uses: actions/download-artifact@v3
      with:
        name: validation-results
        path: current-results/
    
    - name: Download baseline results
      uses: actions/download-artifact@v3
      with:
        name: validation-results
        path: baseline-results/
      continue-on-error: true
    
    - name: Compare performance
      run: |
        echo "🔄 Checking for performance regressions..."
        
        if [ -f "current-results/validation-metrics.json" ] && [ -f "baseline-results/validation-metrics.json" ]; then
          CURRENT_SCORE=$(jq -r '.overall_score' current-results/validation-metrics.json)
          BASELINE_SCORE=$(jq -r '.overall_score' baseline-results/validation-metrics.json)
          
          REGRESSION=$(echo "$BASELINE_SCORE - $CURRENT_SCORE > 5" | bc -l)
          
          if [ "$REGRESSION" = "1" ]; then
            echo "⚠️ Performance regression detected: $CURRENT_SCORE vs $BASELINE_SCORE"
            echo "::warning title=Performance Regression::Score dropped from $BASELINE_SCORE to $CURRENT_SCORE"
          else
            echo "✅ No significant performance regression detected"
          fi
        else
          echo "ℹ️ Baseline results not available for comparison"
        fi
```

### `scripts/validate.sh`
```bash
#!/bin/bash

# LLMKG Validation Script for CI/CD Integration
# Usage: ./scripts/validate.sh [options]

set -e

# Default configuration
ACCURACY_THRESHOLD=${LLMKG_ACCURACY_THRESHOLD:-95.0}
PERFORMANCE_THRESHOLD=${LLMKG_PERFORMANCE_THRESHOLD:-80.0}
PARALLEL_JOBS=${LLMKG_PARALLEL_JOBS:-$(nproc)}
TIMEOUT_MINUTES=${LLMKG_TIMEOUT_MINUTES:-60}
OUTPUT_DIR=${LLMKG_OUTPUT_DIR:-target/validation_reports}
QUIET_MODE=${LLMKG_QUIET:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --accuracy-threshold)
            ACCURACY_THRESHOLD="$2"
            shift 2
            ;;
        --performance-threshold)
            PERFORMANCE_THRESHOLD="$2"
            shift 2
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_MINUTES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        --help)
            echo "LLMKG Validation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --accuracy-threshold FLOAT     Minimum accuracy threshold (default: 95.0)"
            echo "  --performance-threshold FLOAT  Minimum performance threshold (default: 80.0)"
            echo "  --parallel-jobs INT             Number of parallel jobs (default: CPU count)"
            echo "  --timeout INT                   Timeout in minutes (default: 60)"
            echo "  --output-dir PATH               Output directory (default: target/validation_reports)"
            echo "  --quiet                         Quiet mode"
            echo "  --help                          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export LLMKG_ACCURACY_THRESHOLD="$ACCURACY_THRESHOLD"
export LLMKG_PERFORMANCE_THRESHOLD="$PERFORMANCE_THRESHOLD"
export LLMKG_PARALLEL_JOBS="$PARALLEL_JOBS"
export LLMKG_TIMEOUT_MINUTES="$TIMEOUT_MINUTES"
export LLMKG_OUTPUT_DIR="$OUTPUT_DIR"
export LLMKG_QUIET="$QUIET_MODE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration (unless quiet)
if [ "$QUIET_MODE" != "true" ]; then
    echo "🚀 LLMKG Validation Starting"
    echo "📊 Accuracy Threshold: $ACCURACY_THRESHOLD%"
    echo "⚡ Performance Threshold: $PERFORMANCE_THRESHOLD%"
    echo "🔄 Parallel Jobs: $PARALLEL_JOBS"
    echo "⏱️  Timeout: $TIMEOUT_MINUTES minutes"
    echo "📁 Output Directory: $OUTPUT_DIR"
    echo ""
fi

# Build the project
if [ "$QUIET_MODE" != "true" ]; then
    echo "🔨 Building LLMKG..."
fi

cargo build --release

# Run validation
if [ "$QUIET_MODE" != "true" ]; then
    echo "🧪 Running validation..."
fi

# Run with timeout
timeout "${TIMEOUT_MINUTES}m" cargo run --release --bin llmkg -- validate \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/validation.log"

VALIDATION_EXIT_CODE=$?

# Check results
if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    if [ "$QUIET_MODE" != "true" ]; then
        echo "✅ Validation completed successfully!"
    fi
    
    # Display summary if available
    if [ -f "$OUTPUT_DIR/validation-metrics.json" ]; then
        SCORE=$(jq -r '.overall_score // "N/A"' "$OUTPUT_DIR/validation-metrics.json" 2>/dev/null || echo "N/A")
        ACCURACY=$(jq -r '.accuracy.overall // "N/A"' "$OUTPUT_DIR/validation-metrics.json" 2>/dev/null || echo "N/A")
        
        if [ "$QUIET_MODE" != "true" ]; then
            echo "📊 Overall Score: $SCORE/100"
            echo "🎯 Accuracy: $ACCURACY%"
        fi
    fi
else
    if [ "$QUIET_MODE" != "true" ]; then
        echo "❌ Validation failed with exit code: $VALIDATION_EXIT_CODE"
    fi
    
    # Show error details if available
    if [ -f "$OUTPUT_DIR/validation.log" ]; then
        if [ "$QUIET_MODE" != "true" ]; then
            echo ""
            echo "📝 Last 20 lines of validation log:"
            tail -20 "$OUTPUT_DIR/validation.log"
        fi
    fi
fi

# Generate summary for CI systems
if [ -f "$OUTPUT_DIR/validation-metrics.json" ]; then
    # GitHub Actions
    if [ -n "$GITHUB_ACTIONS" ]; then
        SCORE=$(jq -r '.overall_score // 0' "$OUTPUT_DIR/validation-metrics.json")
        echo "validation-score=$SCORE" >> "$GITHUB_OUTPUT"
    fi
    
    # Azure Pipelines
    if [ -n "$AZURE_HTTP_USER_AGENT" ]; then
        SCORE=$(jq -r '.overall_score // 0' "$OUTPUT_DIR/validation-metrics.json")
        echo "##vso[task.setvariable variable=ValidationScore]$SCORE"
    fi
fi

exit $VALIDATION_EXIT_CODE
```

### `scripts/validate.ps1`
```powershell
# LLMKG Validation Script for Windows CI/CD Integration
# Usage: .\scripts\validate.ps1 [options]

param(
    [double]$AccuracyThreshold = $env:LLMKG_ACCURACY_THRESHOLD ?? 95.0,
    [double]$PerformanceThreshold = $env:LLMKG_PERFORMANCE_THRESHOLD ?? 80.0,
    [int]$ParallelJobs = $env:LLMKG_PARALLEL_JOBS ?? (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors,
    [int]$TimeoutMinutes = $env:LLMKG_TIMEOUT_MINUTES ?? 60,
    [string]$OutputDir = $env:LLMKG_OUTPUT_DIR ?? "target\validation_reports",
    [switch]$Quiet = $env:LLMKG_QUIET -eq "true",
    [switch]$Help
)

if ($Help) {
    Write-Host "LLMKG Validation Script for Windows"
    Write-Host ""
    Write-Host "Usage: .\scripts\validate.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -AccuracyThreshold FLOAT     Minimum accuracy threshold (default: 95.0)"
    Write-Host "  -PerformanceThreshold FLOAT  Minimum performance threshold (default: 80.0)"
    Write-Host "  -ParallelJobs INT            Number of parallel jobs (default: CPU count)"
    Write-Host "  -TimeoutMinutes INT          Timeout in minutes (default: 60)"
    Write-Host "  -OutputDir PATH              Output directory (default: target\validation_reports)"
    Write-Host "  -Quiet                       Quiet mode"
    Write-Host "  -Help                        Show this help message"
    exit 0
}

# Set environment variables
$env:LLMKG_ACCURACY_THRESHOLD = $AccuracyThreshold
$env:LLMKG_PERFORMANCE_THRESHOLD = $PerformanceThreshold
$env:LLMKG_PARALLEL_JOBS = $ParallelJobs
$env:LLMKG_TIMEOUT_MINUTES = $TimeoutMinutes
$env:LLMKG_OUTPUT_DIR = $OutputDir
$env:LLMKG_QUIET = $Quiet.ToString().ToLower()

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Print configuration (unless quiet)
if (-not $Quiet) {
    Write-Host "🚀 LLMKG Validation Starting" -ForegroundColor Green
    Write-Host "📊 Accuracy Threshold: $AccuracyThreshold%" -ForegroundColor Cyan
    Write-Host "⚡ Performance Threshold: $PerformanceThreshold%" -ForegroundColor Cyan
    Write-Host "🔄 Parallel Jobs: $ParallelJobs" -ForegroundColor Cyan
    Write-Host "⏱️  Timeout: $TimeoutMinutes minutes" -ForegroundColor Cyan
    Write-Host "📁 Output Directory: $OutputDir" -ForegroundColor Cyan
    Write-Host ""
}

try {
    # Build the project
    if (-not $Quiet) {
        Write-Host "🔨 Building LLMKG..." -ForegroundColor Yellow
    }
    
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    
    # Run validation with timeout
    if (-not $Quiet) {
        Write-Host "🧪 Running validation..." -ForegroundColor Yellow
    }
    
    $job = Start-Job -ScriptBlock {
        param($outputDir)
        cargo run --release --bin llmkg -- validate --output $outputDir
    } -ArgumentList $OutputDir
    
    $completed = Wait-Job -Job $job -Timeout ($TimeoutMinutes * 60)
    
    if ($completed) {
        $result = Receive-Job -Job $job
        $exitCode = $job.State -eq "Completed" ? 0 : 1
        
        if ($result) {
            $result | Out-File -FilePath "$OutputDir\validation.log" -Encoding UTF8
        }
    } else {
        Stop-Job -Job $job
        throw "Validation timed out after $TimeoutMinutes minutes"
    }
    
    Remove-Job -Job $job
    
    # Check results
    if ($exitCode -eq 0) {
        if (-not $Quiet) {
            Write-Host "✅ Validation completed successfully!" -ForegroundColor Green
        }
        
        # Display summary if available
        $metricsPath = Join-Path $OutputDir "validation-metrics.json"
        if (Test-Path $metricsPath) {
            $metrics = Get-Content $metricsPath | ConvertFrom-Json
            
            if (-not $Quiet) {
                Write-Host "📊 Overall Score: $($metrics.overall_score)/100" -ForegroundColor Green
                Write-Host "🎯 Accuracy: $($metrics.accuracy.overall)%" -ForegroundColor Green
            }
        }
    } else {
        throw "Validation failed with exit code: $exitCode"
    }
    
    # Generate summary for CI systems
    if (Test-Path (Join-Path $OutputDir "validation-metrics.json")) {
        $metrics = Get-Content (Join-Path $OutputDir "validation-metrics.json") | ConvertFrom-Json
        
        # Azure Pipelines
        if ($env:AZURE_HTTP_USER_AGENT) {
            Write-Host "##vso[task.setvariable variable=ValidationScore]$($metrics.overall_score)"
        }
        
        # GitHub Actions
        if ($env:GITHUB_ACTIONS) {
            Add-Content -Path $env:GITHUB_OUTPUT -Value "validation-score=$($metrics.overall_score)"
        }
    }
    
    exit 0
    
} catch {
    if (-not $Quiet) {
        Write-Host "❌ Validation failed: $($_.Exception.Message)" -ForegroundColor Red
        
        # Show error details if available
        $logPath = Join-Path $OutputDir "validation.log"
        if (Test-Path $logPath) {
            Write-Host ""
            Write-Host "📝 Last 20 lines of validation log:" -ForegroundColor Yellow
            Get-Content $logPath | Select-Object -Last 20
        }
    }
    
    exit 1
}
```

## Success Criteria
- CI hooks integrate seamlessly with GitHub Actions, Azure Pipelines, and Jenkins
- Proper exit codes are returned for different failure scenarios
- Artifacts are generated in formats suitable for CI dashboards
- Configuration can be controlled through environment variables
- Scripts work on both Unix and Windows environments
- JUnit XML reports are compatible with CI test result displays

## Time Limit
20 minutes maximum