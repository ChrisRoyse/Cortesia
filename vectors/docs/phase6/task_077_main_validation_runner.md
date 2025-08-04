# Task 077: Create Main Validation Runner

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The Main Validation Runner orchestrates all validation components and provides a single entry point for running the complete validation suite.

## Project Structure
```
src/
  validation/
    runner.rs          <- Create this file
  lib.rs
  main.rs              <- Update with validation CLI
```

## Task Description
Create the `ValidationRunner` struct that coordinates all validation phases (correctness, performance, stress testing, security) and produces comprehensive reports.

## Requirements
1. Create `src/validation/runner.rs`
2. Implement `ValidationRunner` with all validation phases
3. Add CLI interface for running validations
4. Implement parallel validation execution
5. Add progress tracking and logging

## Expected Code Structure
```rust
use anyhow::{Result, Context};
use std::path::PathBuf;
use tokio::time::Instant;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

use crate::validation::{
    ground_truth::GroundTruthDataset,
    correctness::CorrectnessValidator,
    performance::PerformanceBenchmark,
    stress::StressTester,
    security::SecurityAuditor,
    report::ValidationReport,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub ground_truth_path: PathBuf,
    pub test_data_path: PathBuf,
    pub text_index_path: PathBuf,
    pub vector_db_path: String,
    pub output_dir: PathBuf,
    pub phases: ValidationPhases,
    pub parallel_execution: bool,
    pub max_concurrent_tests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPhases {
    pub correctness: bool,
    pub performance: bool,
    pub stress_testing: bool,
    pub security_audit: bool,
    pub baseline_comparison: bool,
}

impl Default for ValidationPhases {
    fn default() -> Self {
        Self {
            correctness: true,
            performance: true,
            stress_testing: true,
            security_audit: true,
            baseline_comparison: true,
        }
    }
}

pub struct ValidationRunner {
    config: ValidationConfig,
    report: ValidationReport,
}

impl ValidationRunner {
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            report: ValidationReport::new(),
        }
    }
    
    pub async fn run_complete_validation(&mut self) -> Result<ValidationReport> {
        info!("Starting LLMKG comprehensive validation suite");
        let start_time = Instant::now();
        
        // Initialize output directory
        std::fs::create_dir_all(&self.config.output_dir)
            .context("Failed to create output directory")?;
        
        // Load ground truth dataset
        let dataset = GroundTruthDataset::load_from_file(&self.config.ground_truth_path)
            .context("Failed to load ground truth dataset")?;
        
        info!("Ground truth dataset loaded: {} test cases", dataset.test_cases.len());
        
        // Run validation phases
        if self.config.phases.correctness {
            self.run_correctness_validation(&dataset).await?;
        }
        
        if self.config.phases.performance {
            self.run_performance_benchmarks(&dataset).await?;
        }
        
        if self.config.phases.stress_testing {
            self.run_stress_tests().await?;
        }
        
        if self.config.phases.security_audit {
            self.run_security_audit().await?;
        }
        
        if self.config.phases.baseline_comparison {
            self.run_baseline_comparison(&dataset).await?;
        }
        
        // Calculate final metrics
        let total_duration = start_time.elapsed();
        self.report.metadata.test_duration_minutes = total_duration.as_secs_f64() / 60.0;
        self.report.metadata.total_test_cases = dataset.test_cases.len();
        self.report.calculate_overall_score();
        
        // Generate recommendations
        self.generate_recommendations();
        
        // Save reports
        self.save_reports().await?;
        
        info!("Validation complete. Overall score: {:.1}/100", self.report.overall_score);
        
        Ok(self.report.clone())
    }
    
    async fn run_correctness_validation(&mut self, dataset: &GroundTruthDataset) -> Result<()> {
        info!("Running correctness validation phase");
        
        let validator = CorrectnessValidator::new(&self.config.text_index_path, &self.config.vector_db_path).await?;
        
        let mut passed_count = 0;
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        
        // Group test cases by query type for detailed reporting
        let mut query_type_results = std::collections::HashMap::new();
        
        for (i, test_case) in dataset.test_cases.iter().enumerate() {
            info!("Validating test case {}/{}: {}", i + 1, dataset.test_cases.len(), test_case.query);
            
            match validator.validate(test_case).await {
                Ok(result) => {
                    if result.is_correct {
                        passed_count += 1;
                    }
                    
                    total_precision += result.precision;
                    total_recall += result.recall;
                    total_f1 += result.f1_score;
                    false_positives += result.false_positives;
                    false_negatives += result.false_negatives;
                    
                    // Update query type results
                    let query_type_str = format!("{:?}", test_case.query_type);
                    let entry = query_type_results.entry(query_type_str.clone()).or_insert_with(|| {
                        crate::validation::report::QueryTypeResult {
                            query_type: query_type_str,
                            test_cases_count: 0,
                            passed_count: 0,
                            accuracy_percentage: 0.0,
                            average_precision: 0.0,
                            average_recall: 0.0,
                            average_f1_score: 0.0,
                        }
                    });
                    
                    entry.test_cases_count += 1;
                    if result.is_correct {
                        entry.passed_count += 1;
                    }
                    entry.average_precision = (entry.average_precision * (entry.test_cases_count - 1) as f64 + result.precision) / entry.test_cases_count as f64;
                    entry.average_recall = (entry.average_recall * (entry.test_cases_count - 1) as f64 + result.recall) / entry.test_cases_count as f64;
                    entry.average_f1_score = (entry.average_f1_score * (entry.test_cases_count - 1) as f64 + result.f1_score) / entry.test_cases_count as f64;
                    entry.accuracy_percentage = (entry.passed_count as f64 / entry.test_cases_count as f64) * 100.0;
                }
                Err(e) => {
                    error!("Validation failed for test case {}: {}", i + 1, e);
                    false_negatives += 1; // Treat validation errors as false negatives
                }
            }
        }
        
        // Update accuracy report
        self.report.accuracy_metrics.overall_accuracy = (passed_count as f64 / dataset.test_cases.len() as f64) * 100.0;
        self.report.accuracy_metrics.query_type_results = query_type_results;
        self.report.accuracy_metrics.false_positives_total = false_positives;
        self.report.accuracy_metrics.false_negatives_total = false_negatives;
        self.report.accuracy_metrics.perfect_accuracy_achieved = passed_count == dataset.test_cases.len();
        
        info!("Correctness validation complete: {:.1}% accuracy", self.report.accuracy_metrics.overall_accuracy);
        
        Ok(())
    }
    
    async fn run_performance_benchmarks(&mut self, dataset: &GroundTruthDataset) -> Result<()> {
        info!("Running performance benchmark phase");
        
        let benchmark = PerformanceBenchmark::new(&self.config.text_index_path, &self.config.vector_db_path).await?;
        
        // Run latency benchmarks
        let latency_results = benchmark.run_latency_benchmark(&dataset.test_cases).await?;
        
        // Run throughput benchmarks
        let throughput_results = benchmark.run_throughput_benchmark(&dataset.test_cases).await?;
        
        // Monitor resource usage
        let resource_usage = benchmark.monitor_resource_usage().await?;
        
        // Update performance report
        self.report.performance_metrics.latency_metrics = latency_results;
        self.report.performance_metrics.throughput_metrics = throughput_results;
        self.report.performance_metrics.resource_usage = resource_usage;
        
        // Check if performance targets are met
        self.report.performance_metrics.meets_targets = 
            latency_results.p50_ms <= latency_results.target_p50_ms &&
            latency_results.p95_ms <= latency_results.target_p95_ms &&
            latency_results.p99_ms <= latency_results.target_p99_ms &&
            throughput_results.queries_per_second >= throughput_results.target_qps;
        
        info!("Performance benchmarks complete. Targets met: {}", self.report.performance_metrics.meets_targets);
        
        Ok(())
    }
    
    async fn run_stress_tests(&mut self) -> Result<()> {
        info!("Running stress testing phase");
        
        let stress_tester = StressTester::new(&self.config.text_index_path, &self.config.vector_db_path).await?;
        
        // Test large file handling
        let large_file_result = stress_tester.test_large_file_handling().await?;
        self.report.stress_test_results.large_file_handling = large_file_result;
        
        // Test concurrent users
        let concurrent_result = stress_tester.test_concurrent_users(100).await?;
        self.report.stress_test_results.concurrent_users = concurrent_result;
        
        // Test memory pressure
        let memory_result = stress_tester.test_memory_pressure().await?;
        self.report.stress_test_results.memory_pressure = memory_result;
        
        // Test sustained load
        let sustained_result = stress_tester.test_sustained_load(std::time::Duration::from_secs(300)).await?;
        self.report.stress_test_results.sustained_load = sustained_result;
        
        info!("Stress testing phase complete");
        
        Ok(())
    }
    
    async fn run_security_audit(&mut self) -> Result<()> {
        info!("Running security audit phase");
        
        let security_auditor = SecurityAuditor::new(&self.config.text_index_path, &self.config.vector_db_path).await?;
        
        // Test SQL injection resistance
        let sql_result = security_auditor.test_sql_injection_resistance().await?;
        self.report.security_audit.sql_injection_tests = sql_result;
        
        // Test input validation
        let input_result = security_auditor.test_input_validation().await?;
        self.report.security_audit.input_validation_tests = input_result;
        
        // Test DoS prevention
        let dos_result = security_auditor.test_dos_prevention().await?;
        self.report.security_audit.dos_prevention_tests = dos_result;
        
        // Test malicious query handling
        let malicious_result = security_auditor.test_malicious_queries().await?;
        self.report.security_audit.malicious_query_tests = malicious_result;
        
        info!("Security audit phase complete");
        
        Ok(())
    }
    
    async fn run_baseline_comparison(&mut self, dataset: &GroundTruthDataset) -> Result<()> {
        info!("Running baseline comparison phase");
        
        // This would compare against ripgrep, tantivy, etc.
        // Implementation depends on baseline comparison structs
        
        info!("Baseline comparison phase complete");
        
        Ok(())
    }
    
    fn generate_recommendations(&mut self) {
        let mut recommendations = Vec::new();
        
        // Accuracy recommendations
        if self.report.accuracy_metrics.overall_accuracy < 100.0 {
            recommendations.push(format!(
                "Accuracy is {:.1}% - investigate {} failed test cases",
                self.report.accuracy_metrics.overall_accuracy,
                self.report.accuracy_metrics.false_negatives_total + self.report.accuracy_metrics.false_positives_total
            ));
        }
        
        // Performance recommendations
        if !self.report.performance_metrics.meets_targets {
            recommendations.push("Performance targets not met - consider optimization".to_string());
        }
        
        // Memory recommendations
        if self.report.performance_metrics.resource_usage.peak_memory_mb > 1024.0 {
            recommendations.push("High memory usage detected - review memory efficiency".to_string());
        }
        
        // Security recommendations
        let security_tests = [
            &self.report.security_audit.sql_injection_tests,
            &self.report.security_audit.input_validation_tests,
            &self.report.security_audit.dos_prevention_tests,
            &self.report.security_audit.malicious_query_tests,
        ];
        
        for test in security_tests {
            if !test.passed {
                recommendations.push(format!("Security concern: {}", test.details));
            }
        }
        
        self.report.recommendations = recommendations;
    }
    
    async fn save_reports(&self) -> Result<()> {
        // Save markdown report
        let markdown_path = self.config.output_dir.join("validation_report.md");
        self.report.save_markdown(&markdown_path)
            .context("Failed to save markdown report")?;
        
        // Save JSON report
        let json_path = self.config.output_dir.join("validation_report.json");
        self.report.save_json(&json_path)
            .context("Failed to save JSON report")?;
        
        info!("Reports saved to {}", self.config.output_dir.display());
        
        Ok(())
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            ground_truth_path: PathBuf::from("tests/data/ground_truth.json"),
            test_data_path: PathBuf::from("tests/data"),
            text_index_path: PathBuf::from("target/debug/text_index"),
            vector_db_path: "target/debug/vector.lance".to_string(),
            output_dir: PathBuf::from("target/validation_reports"),
            phases: ValidationPhases::default(),
            parallel_execution: true,
            max_concurrent_tests: num_cpus::get(),
        }
    }
}
```

## CLI Integration
Add to `src/main.rs`:
```rust
use clap::{Parser, Subcommand};
use crate::validation::{runner::ValidationRunner, runner::ValidationConfig};

#[derive(Parser)]
#[command(name = "llmkg")]
#[command(about = "LLMKG Vector Indexing System")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive validation suite
    Validate {
        /// Path to validation configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
        
        /// Run only specific phases (comma-separated)
        #[arg(short, long)]
        phases: Option<String>,
        
        /// Output directory for reports
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

async fn handle_validate_command(
    config_path: Option<PathBuf>,
    phases: Option<String>,
    output_dir: Option<PathBuf>,
) -> Result<()> {
    let mut config = if let Some(config_path) = config_path {
        // Load from file
        let config_content = std::fs::read_to_string(&config_path)?;
        toml::from_str(&config_content)?
    } else {
        ValidationConfig::default()
    };
    
    // Override output directory if specified
    if let Some(output_dir) = output_dir {
        config.output_dir = output_dir;
    }
    
    // Override phases if specified
    if let Some(phases_str) = phases {
        let phase_list: Vec<&str> = phases_str.split(',').collect();
        config.phases = ValidationPhases {
            correctness: phase_list.contains(&"correctness"),
            performance: phase_list.contains(&"performance"),
            stress_testing: phase_list.contains(&"stress"),
            security_audit: phase_list.contains(&"security"),
            baseline_comparison: phase_list.contains(&"baseline"),
        };
    }
    
    let mut runner = ValidationRunner::new(config);
    let report = runner.run_complete_validation().await?;
    
    println!("Validation complete!");
    println!("Overall score: {:.1}/100", report.overall_score);
    println!("Reports saved to: {}", runner.config.output_dir.display());
    
    Ok(())
}
```

## Success Criteria
- ValidationRunner orchestrates all validation phases
- CLI interface works correctly
- Progress tracking and logging are comprehensive
- Parallel execution improves performance
- Reports are generated successfully
- Recommendations provide actionable insights

## Time Limit
15 minutes maximum