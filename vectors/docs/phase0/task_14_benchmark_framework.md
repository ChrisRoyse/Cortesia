# Task 14: Create Benchmark Framework Structure

## Context
You are beginning the baseline benchmarking phase (Phase 0, Tasks 14-17). Test data generation is complete. Now you need to create a comprehensive benchmarking framework that can measure performance across all components and establish baseline metrics.

## Objective
Create a robust benchmarking framework that measures performance of individual components (Tantivy, LanceDB, Rayon, tree-sitter) and the integrated system, providing detailed metrics and comparison capabilities.

## Requirements
1. Create benchmark framework structure with timing and metrics
2. Implement benchmark harness for consistent measurement
3. Create benchmark configuration system
4. Implement result collection and reporting
5. Add statistical analysis for reliable measurements
6. Create benchmark comparison utilities

## Implementation for benchmark.rs
```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub timeout_seconds: u64,
    pub collect_memory_stats: bool,
    pub collect_cpu_stats: bool,
    pub output_format: OutputFormat,
    pub comparison_baseline: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Console,
    Html,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub description: String,
    pub timestamp: String,
    pub config: BenchmarkConfig,
    pub metrics: BenchmarkMetrics,
    pub statistical_analysis: StatisticalAnalysis,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub execution_times: Vec<Duration>,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_deviation: Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_usage: MemoryUsage,
    pub cpu_usage: CpuUsage,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub confidence_interval_95: (Duration, Duration),
    pub coefficient_of_variation: f64,
    pub outliers_removed: usize,
    pub is_reliable: bool,
    pub reliability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_allocations: u64,
    pub memory_deallocations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsage {
    pub average_cpu_percent: f64,
    pub peak_cpu_percent: f64,
    pub context_switches: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_total_mb: f64,
    pub rust_version: String,
    pub build_profile: String,
}

pub struct BenchmarkFramework {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
    system_info: SystemInfo,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 100,
            timeout_seconds: 300,
            collect_memory_stats: true,
            collect_cpu_stats: true,
            output_format: OutputFormat::Json,
            comparison_baseline: None,
        }
    }
}

impl BenchmarkFramework {
    /// Create new benchmark framework with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            system_info: Self::collect_system_info(),
        }
    }
    
    /// Create framework with default configuration
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
    
    /// Run a benchmark with the given name and function
    pub fn benchmark<F, R>(&mut self, name: &str, description: &str, mut benchmark_fn: F) -> Result<&BenchmarkResult>
    where
        F: FnMut() -> Result<R>,
    {
        info!("Starting benchmark: {}", name);
        debug!("Description: {}", description);
        
        // Warmup phase
        debug!("Running {} warmup iterations", self.config.warmup_iterations);
        for i in 0..self.config.warmup_iterations {
            debug!("Warmup iteration {}/{}", i + 1, self.config.warmup_iterations);
            benchmark_fn()?;
        }
        
        // Measurement phase
        debug!("Running {} measurement iterations", self.config.measurement_iterations);
        let mut execution_times = Vec::new();
        let mut memory_samples = Vec::new();
        let mut cpu_samples = Vec::new();
        
        for i in 0..self.config.measurement_iterations {
            debug!("Measurement iteration {}/{}", i + 1, self.config.measurement_iterations);
            
            // Collect baseline metrics
            let memory_before = if self.config.collect_memory_stats {
                Some(Self::get_memory_usage())
            } else {
                None
            };
            
            let cpu_before = if self.config.collect_cpu_stats {
                Some(Self::get_cpu_usage())
            } else {
                None
            };
            
            // Execute benchmark
            let start_time = Instant::now();
            benchmark_fn()?;
            let execution_time = start_time.elapsed();
            
            // Collect post-execution metrics
            if let Some(memory_before_val) = memory_before {
                let memory_after = Self::get_memory_usage();
                memory_samples.push(memory_after - memory_before_val);
            }
            
            if let Some(cpu_before_val) = cpu_before {
                let cpu_after = Self::get_cpu_usage();
                cpu_samples.push(cpu_after - cpu_before_val);
            }
            
            execution_times.push(execution_time);
            
            // Check for timeout
            if execution_time.as_secs() > self.config.timeout_seconds {
                warn!("Benchmark iteration exceeded timeout: {} seconds", execution_time.as_secs());
                break;
            }
        }
        
        // Analyze results
        let metrics = Self::analyze_metrics(execution_times, memory_samples, cpu_samples)?;
        let statistical_analysis = Self::perform_statistical_analysis(&metrics);
        
        let result = BenchmarkResult {
            name: name.to_string(),
            description: description.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            metrics,
            statistical_analysis,
            system_info: self.system_info.clone(),
        };
        
        self.results.push(result);
        
        info!(
            "Benchmark '{}' completed: mean={:?}, throughput={:.2} ops/sec",
            name,
            self.results.last().unwrap().metrics.mean_time,
            self.results.last().unwrap().metrics.throughput_ops_per_sec
        );
        
        Ok(self.results.last().unwrap())
    }
    
    /// Benchmark with custom metrics collection
    pub fn benchmark_with_metrics<F, R, M>(
        &mut self,
        name: &str,
        description: &str,
        mut benchmark_fn: F,
        mut metrics_fn: M,
    ) -> Result<&BenchmarkResult>
    where
        F: FnMut() -> Result<R>,
        M: FnMut() -> HashMap<String, f64>,
    {
        info!("Starting benchmark with custom metrics: {}", name);
        
        // Similar to benchmark() but collect custom metrics
        let mut execution_times = Vec::new();
        let mut custom_metrics_samples = Vec::new();
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            benchmark_fn()?;
        }
        
        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let start_time = Instant::now();
            benchmark_fn()?;
            let execution_time = start_time.elapsed();
            
            let custom_metrics = metrics_fn();
            
            execution_times.push(execution_time);
            custom_metrics_samples.push(custom_metrics);
        }
        
        // Analyze results with custom metrics
        let mut metrics = Self::analyze_metrics(execution_times, Vec::new(), Vec::new())?;
        
        // Aggregate custom metrics
        let mut aggregated_custom = HashMap::new();
        for sample in &custom_metrics_samples {
            for (key, value) in sample {
                let entry = aggregated_custom.entry(key.clone()).or_insert(Vec::new());
                entry.push(*value);
            }
        }
        
        for (key, values) in aggregated_custom {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            metrics.custom_metrics.insert(key, mean);
        }
        
        let statistical_analysis = Self::perform_statistical_analysis(&metrics);
        
        let result = BenchmarkResult {
            name: name.to_string(),
            description: description.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            metrics,
            statistical_analysis,
            system_info: self.system_info.clone(),
        };
        
        self.results.push(result);
        Ok(self.results.last().unwrap())
    }
    
    /// Compare benchmark results
    pub fn compare_results(&self, name1: &str, name2: &str) -> Option<BenchmarkComparison> {
        let result1 = self.results.iter().find(|r| r.name == name1)?;
        let result2 = self.results.iter().find(|r| r.name == name2)?;
        
        Some(BenchmarkComparison {
            benchmark1: name1.to_string(),
            benchmark2: name2.to_string(),
            time_improvement: Self::calculate_improvement(
                result1.metrics.mean_time,
                result2.metrics.mean_time,
            ),
            throughput_improvement: Self::calculate_improvement(
                result1.metrics.throughput_ops_per_sec,
                result2.metrics.throughput_ops_per_sec,
            ),
            memory_improvement: Self::calculate_improvement(
                result1.metrics.memory_usage.average_memory_mb,
                result2.metrics.memory_usage.average_memory_mb,
            ),
            is_significant: Self::is_statistically_significant(result1, result2),
        })
    }
    
    /// Generate benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            system_info: self.system_info.clone(),
            config: self.config.clone(),
            results: self.results.clone(),
            summary: Self::generate_summary(&self.results),
        }
    }
    
    /// Save results to file
    pub fn save_results(&self, path: &str) -> Result<()> {
        let report = self.generate_report();
        
        match self.config.output_format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(&report)?;
                std::fs::write(path, json)?;
            }
            OutputFormat::Csv => {
                let csv = Self::results_to_csv(&self.results)?;
                std::fs::write(path, csv)?;
            }
            OutputFormat::Console => {
                println!("{}", Self::results_to_console(&self.results));
            }
            OutputFormat::Html => {
                let html = Self::results_to_html(&self.results);
                std::fs::write(path, html)?;
            }
        }
        
        info!("Benchmark results saved to: {}", path);
        Ok(())
    }
    
    // Helper methods
    fn collect_system_info() -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu_model: "Unknown".to_string(), // Would use system crate
            cpu_cores: num_cpus::get(),
            memory_total_mb: 0.0, // Would use system crate
            rust_version: env!("RUSTC_VERSION").to_string(),
            build_profile: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
        }
    }
    
    fn get_memory_usage() -> f64 {
        // Simplified - would use system metrics
        0.0
    }
    
    fn get_cpu_usage() -> f64 {
        // Simplified - would use system metrics  
        0.0
    }
    
    fn analyze_metrics(
        execution_times: Vec<Duration>,
        memory_samples: Vec<f64>,
        cpu_samples: Vec<f64>,
    ) -> Result<BenchmarkMetrics> {
        if execution_times.is_empty() {
            return Err(anyhow::anyhow!("No execution times recorded"));
        }
        
        let mut times = execution_times.clone();
        times.sort();
        
        let mean_time = Duration::from_nanos(
            times.iter().map(|d| d.as_nanos()).sum::<u128>() / times.len() as u128
        );
        
        let median_time = times[times.len() / 2];
        let min_time = times[0];
        let max_time = times[times.len() - 1];
        
        // Calculate standard deviation
        let variance = times.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_time.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / times.len() as f64;
        
        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);
        
        let throughput_ops_per_sec = 1.0 / mean_time.as_secs_f64();
        
        Ok(BenchmarkMetrics {
            execution_times: times,
            mean_time,
            median_time,
            min_time,
            max_time,
            std_deviation,
            throughput_ops_per_sec,
            memory_usage: MemoryUsage {
                peak_memory_mb: memory_samples.iter().fold(0.0, |a, &b| a.max(b)),
                average_memory_mb: if !memory_samples.is_empty() {
                    memory_samples.iter().sum::<f64>() / memory_samples.len() as f64
                } else {
                    0.0
                },
                memory_allocations: 0,
                memory_deallocations: 0,
            },
            cpu_usage: CpuUsage {
                average_cpu_percent: if !cpu_samples.is_empty() {
                    cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64
                } else {
                    0.0
                },
                peak_cpu_percent: cpu_samples.iter().fold(0.0, |a, &b| a.max(b)),
                context_switches: 0,
            },
            custom_metrics: HashMap::new(),
        })
    }
    
    fn perform_statistical_analysis(metrics: &BenchmarkMetrics) -> StatisticalAnalysis {
        let mean = metrics.mean_time.as_nanos() as f64;
        let std_dev = metrics.std_deviation.as_nanos() as f64;
        
        // 95% confidence interval (assuming normal distribution)
        let margin_of_error = 1.96 * std_dev / (metrics.execution_times.len() as f64).sqrt();
        let ci_lower = Duration::from_nanos((mean - margin_of_error) as u64);
        let ci_upper = Duration::from_nanos((mean + margin_of_error) as u64);
        
        let coefficient_of_variation = std_dev / mean;
        
        StatisticalAnalysis {
            confidence_interval_95: (ci_lower, ci_upper),
            coefficient_of_variation,
            outliers_removed: 0,
            is_reliable: coefficient_of_variation < 0.1, // Less than 10% variation
            reliability_score: 1.0 - coefficient_of_variation.min(1.0),
        }
    }
    
    fn calculate_improvement(old_value: Duration, new_value: Duration) -> f64 {
        let old_ns = old_value.as_nanos() as f64;
        let new_ns = new_value.as_nanos() as f64;
        (old_ns - new_ns) / old_ns * 100.0
    }
    
    fn calculate_improvement(old_value: f64, new_value: f64) -> f64 {
        (new_value - old_value) / old_value * 100.0
    }
    
    fn is_statistically_significant(result1: &BenchmarkResult, result2: &BenchmarkResult) -> bool {
        // Simplified significance test
        let diff = (result1.metrics.mean_time.as_nanos() as f64 - 
                   result2.metrics.mean_time.as_nanos() as f64).abs();
        let combined_std = (result1.metrics.std_deviation.as_nanos() as f64 + 
                           result2.metrics.std_deviation.as_nanos() as f64) / 2.0;
        
        diff > 2.0 * combined_std // Roughly 95% confidence
    }
    
    fn generate_summary(results: &[BenchmarkResult]) -> BenchmarkSummary {
        BenchmarkSummary {
            total_benchmarks: results.len(),
            fastest_benchmark: results.iter()
                .min_by_key(|r| r.metrics.mean_time)
                .map(|r| r.name.clone()),
            slowest_benchmark: results.iter()
                .max_by_key(|r| r.metrics.mean_time)
                .map(|r| r.name.clone()),
            average_throughput: results.iter()
                .map(|r| r.metrics.throughput_ops_per_sec)
                .sum::<f64>() / results.len() as f64,
            total_test_time: results.iter()
                .map(|r| r.metrics.mean_time)
                .sum(),
        }
    }
    
    fn results_to_csv(results: &[BenchmarkResult]) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("name,mean_time_ns,throughput_ops_per_sec,memory_mb,cpu_percent\n");
        
        for result in results {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                result.name,
                result.metrics.mean_time.as_nanos(),
                result.metrics.throughput_ops_per_sec,
                result.metrics.memory_usage.average_memory_mb,
                result.metrics.cpu_usage.average_cpu_percent
            ));
        }
        
        Ok(csv)
    }
    
    fn results_to_console(results: &[BenchmarkResult]) -> String {
        let mut output = String::new();
        output.push_str("Benchmark Results\n");
        output.push_str("=================\n");
        
        for result in results {
            output.push_str(&format!(
                "{}: {:?} ({:.2} ops/sec)\n",
                result.name,
                result.metrics.mean_time,
                result.metrics.throughput_ops_per_sec
            ));
        }
        
        output
    }
    
    fn results_to_html(results: &[BenchmarkResult]) -> String {
        let mut html = String::new();
        html.push_str("<html><head><title>Benchmark Results</title></head><body>");
        html.push_str("<h1>Benchmark Results</h1>");
        html.push_str("<table border='1'>");
        html.push_str("<tr><th>Name</th><th>Mean Time</th><th>Throughput</th></tr>");
        
        for result in results {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:?}</td><td>{:.2} ops/sec</td></tr>",
                result.name,
                result.metrics.mean_time,
                result.metrics.throughput_ops_per_sec
            ));
        }
        
        html.push_str("</table></body></html>");
        html
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub benchmark1: String,
    pub benchmark2: String,
    pub time_improvement: f64,
    pub throughput_improvement: f64,
    pub memory_improvement: f64,
    pub is_significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub system_info: SystemInfo,
    pub config: BenchmarkConfig,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_benchmarks: usize,
    pub fastest_benchmark: Option<String>,
    pub slowest_benchmark: Option<String>,
    pub average_throughput: f64,
    pub total_test_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_framework_creation() {
        let framework = BenchmarkFramework::default();
        assert_eq!(framework.results.len(), 0);
        assert_eq!(framework.config.warmup_iterations, 5);
    }
    
    #[test]
    fn test_simple_benchmark() {
        let mut framework = BenchmarkFramework::default();
        
        let result = framework.benchmark(
            "simple_test",
            "A simple test benchmark",
            || {
                std::thread::sleep(Duration::from_millis(1));
                Ok(42)
            }
        ).unwrap();
        
        assert_eq!(result.name, "simple_test");
        assert!(result.metrics.mean_time > Duration::from_nanos(1000)); // At least 1μs
    }
    
    #[test]
    fn test_custom_metrics_benchmark() {
        let mut framework = BenchmarkFramework::default();
        
        let result = framework.benchmark_with_metrics(
            "custom_metrics_test",
            "Test with custom metrics",
            || {
                std::thread::sleep(Duration::from_millis(1));
                Ok(())
            },
            || {
                let mut metrics = HashMap::new();
                metrics.insert("custom_value".to_string(), 42.0);
                metrics
            }
        ).unwrap();
        
        assert!(result.metrics.custom_metrics.contains_key("custom_value"));
        assert_eq!(result.metrics.custom_metrics["custom_value"], 42.0);
    }
}
```

## Implementation Steps
1. Add comprehensive benchmark data structures to benchmark.rs
2. Implement BenchmarkFramework with configuration system
3. Add timing and metrics collection functionality
4. Implement statistical analysis for reliable measurements
5. Add comparison utilities for benchmark results
6. Implement multiple output formats (JSON, CSV, HTML, Console)
7. Add system information collection
8. Create report generation functionality
9. Add test suite for benchmark framework

## Success Criteria
- [ ] BenchmarkFramework struct implemented and compiling
- [ ] Configuration system for benchmark parameters
- [ ] Timing measurement with statistical analysis
- [ ] Memory and CPU usage collection
- [ ] Custom metrics support for specialized measurements
- [ ] Result comparison and significance testing
- [ ] Multiple output formats supported
- [ ] System information collection for context
- [ ] All benchmark framework tests pass

## Test Command
```bash
cargo test test_benchmark_framework_creation
cargo test test_simple_benchmark
cargo test test_custom_metrics_benchmark
```

## Framework Capabilities
After completion, the framework provides:
- **Precise Timing**: Nanosecond-precision timing with statistical analysis
- **System Metrics**: Memory usage, CPU usage, and system information
- **Custom Metrics**: Support for domain-specific measurements
- **Statistical Analysis**: Confidence intervals, outlier detection, significance testing
- **Comparison Tools**: Compare different implementations or configurations
- **Multiple Formats**: JSON, CSV, HTML, and console output
- **Reliability Scoring**: Assess measurement quality and consistency

## Time Estimate
10 minutes

## Next Task
Task 15: Implement and run Tantivy performance benchmarks with Windows optimization.