# Micro-Task 140: Setup Statistical Analysis Tools

## Objective
Implement statistical analysis tools for benchmark data to ensure measurement validity and detect performance regressions with confidence intervals.

## Context
Statistical analysis is critical for benchmark reliability. This task provides tools to analyze performance data, detect outliers, and validate that measurements meet the <5ms allocation target with statistical significance.

## Prerequisites
- Task 139 completed (Windows performance monitoring configured)
- Criterion benchmark framework available
- Baseline measurement framework implemented

## Time Estimate
10 minutes

## Instructions
1. Create statistical analysis module `stats_analysis.rs`:
   ```rust
   use std::collections::VecDeque;
   
   #[derive(Debug, Clone)]
   pub struct StatisticalAnalysis {
       samples: VecDeque<f64>,
       max_samples: usize,
   }
   
   impl StatisticalAnalysis {
       pub fn new(max_samples: usize) -> Self {
           Self {
               samples: VecDeque::with_capacity(max_samples),
               max_samples,
           }
       }
       
       pub fn add_sample(&mut self, value: f64) {
           if self.samples.len() >= self.max_samples {
               self.samples.pop_front();
           }
           self.samples.push_back(value);
       }
       
       pub fn mean(&self) -> Option<f64> {
           if self.samples.is_empty() {
               return None;
           }
           Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
       }
       
       pub fn standard_deviation(&self) -> Option<f64> {
           let mean = self.mean()?;
           if self.samples.len() < 2 {
               return None;
           }
           
           let variance = self.samples.iter()
               .map(|x| (x - mean).powi(2))
               .sum::<f64>() / (self.samples.len() - 1) as f64;
           
           Some(variance.sqrt())
       }
       
       pub fn confidence_interval_95(&self) -> Option<(f64, f64)> {
           let mean = self.mean()?;
           let std_dev = self.standard_deviation()?;
           let n = self.samples.len() as f64;
           
           // t-distribution critical value for 95% confidence (approximate)
           let t_critical = if n > 30.0 { 1.96 } else { 2.042 }; // Conservative estimate
           
           let margin_of_error = t_critical * (std_dev / n.sqrt());
           Some((mean - margin_of_error, mean + margin_of_error))
       }
       
       pub fn meets_allocation_target(&self) -> Option<bool> {
           let (lower, upper) = self.confidence_interval_95()?;
           // 5ms = 5000 microseconds target
           Some(upper <= 5000.0)
       }
       
       pub fn detect_outliers(&self) -> Vec<(usize, f64)> {
           let mean = match self.mean() {
               Some(m) => m,
               None => return Vec::new(),
           };
           
           let std_dev = match self.standard_deviation() {
               Some(s) => s,
               None => return Vec::new(),
           };
           
           let threshold = 2.0 * std_dev;
           
           self.samples.iter()
               .enumerate()
               .filter(|(_, &value)| (value - mean).abs() > threshold)
               .map(|(i, &value)| (i, value))
               .collect()
       }
       
       pub fn performance_trend(&self) -> Option<f64> {
           if self.samples.len() < 3 {
               return None;
           }
           
           let n = self.samples.len() as f64;
           let x_mean = (n - 1.0) / 2.0;
           let y_mean = self.mean()?;
           
           let numerator: f64 = self.samples.iter()
               .enumerate()
               .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
               .sum();
           
           let denominator: f64 = (0..self.samples.len())
               .map(|i| (i as f64 - x_mean).powi(2))
               .sum();
           
           if denominator.abs() < f64::EPSILON {
               None
           } else {
               Some(numerator / denominator)
           }
       }
   }
   
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_allocation_target_validation() {
           let mut stats = StatisticalAnalysis::new(100);
           
           // Add samples under 5ms (5000 microseconds)
           for _ in 0..30 {
               stats.add_sample(4500.0); // 4.5ms
           }
           
           assert!(stats.meets_allocation_target().unwrap_or(false));
       }
       
       #[test]
       fn test_outlier_detection() {
           let mut stats = StatisticalAnalysis::new(100);
           
           // Normal samples around 3ms
           for _ in 0..20 {
               stats.add_sample(3000.0);
           }
           
           // Add outlier
           stats.add_sample(10000.0); // 10ms outlier
           
           let outliers = stats.detect_outliers();
           assert!(!outliers.is_empty());
       }
   }
   ```
2. Create performance regression detector `regression_detector.rs`:
   ```rust
   use super::stats_analysis::StatisticalAnalysis;
   use std::collections::HashMap;
   use serde::{Serialize, Deserialize};
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct PerformanceBaseline {
       pub benchmark_name: String,
       pub baseline_mean: f64,
       pub baseline_std_dev: f64,
       pub sample_count: usize,
       pub timestamp: u64,
   }
   
   pub struct RegressionDetector {
       baselines: HashMap<String, PerformanceBaseline>,
       current_stats: HashMap<String, StatisticalAnalysis>,
   }
   
   impl RegressionDetector {
       pub fn new() -> Self {
           Self {
               baselines: HashMap::new(),
               current_stats: HashMap::new(),
           }
       }
       
       pub fn record_baseline(&mut self, name: &str, stats: &StatisticalAnalysis) {
           if let (Some(mean), Some(std_dev)) = (stats.mean(), stats.standard_deviation()) {
               let baseline = PerformanceBaseline {
                   benchmark_name: name.to_string(),
                   baseline_mean: mean,
                   baseline_std_dev: std_dev,
                   sample_count: stats.samples.len(),
                   timestamp: std::time::SystemTime::now()
                       .duration_since(std::time::UNIX_EPOCH)
                       .unwrap_or_default()
                       .as_secs(),
               };
               
               self.baselines.insert(name.to_string(), baseline);
           }
       }
       
       pub fn add_measurement(&mut self, name: &str, value: f64) {
           self.current_stats.entry(name.to_string())
               .or_insert_with(|| StatisticalAnalysis::new(1000))
               .add_sample(value);
       }
       
       pub fn detect_regression(&self, name: &str, significance_level: f64) -> Option<bool> {
           let baseline = self.baselines.get(name)?;
           let current_stats = self.current_stats.get(name)?;
           let current_mean = current_stats.mean()?;
           
           // Detect if current performance is significantly worse than baseline
           let threshold = baseline.baseline_mean + (significance_level * baseline.baseline_std_dev);
           Some(current_mean > threshold)
       }
       
       pub fn performance_report(&self, name: &str) -> Option<String> {
           let baseline = self.baselines.get(name)?;
           let current_stats = self.current_stats.get(name)?;
           let current_mean = current_stats.mean()?;
           let current_std_dev = current_stats.standard_deviation()?;
           
           let change_percent = ((current_mean - baseline.baseline_mean) / baseline.baseline_mean) * 100.0;
           
           Some(format!(
               "Performance Report for {}\n\
               Baseline: {:.2}μs (±{:.2}μs)\n\
               Current:  {:.2}μs (±{:.2}μs)\n\
               Change:   {:.2}% {}\n\
               5ms Target: {}",
               name,
               baseline.baseline_mean,
               baseline.baseline_std_dev,
               current_mean,
               current_std_dev,
               change_percent,
               if change_percent > 0.0 { "slower" } else { "faster" },
               if current_stats.meets_allocation_target().unwrap_or(false) { "✓ PASS" } else { "✗ FAIL" }
           ))
       }
   }
   ```
3. Create benchmark reporting utility `benchmark_reporter.rs`:
   ```rust
   use std::fs;
   use std::path::Path;
   use serde_json;
   use super::regression_detector::{RegressionDetector, PerformanceBaseline};
   
   pub fn generate_html_report(detector: &RegressionDetector, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
       let html_content = format!(r#"
   <!DOCTYPE html>
   <html>
   <head>
       <title>Vector Search Performance Report</title>
       <style>
           body {{ font-family: Arial, sans-serif; margin: 20px; }}
           .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
           .pass {{ background-color: #d4edda; }}
           .fail {{ background-color: #f8d7da; }}
           .target {{ font-weight: bold; color: #007bff; }}
       </style>
   </head>
   <body>
       <h1>Vector Search Performance Benchmarks</h1>
       <div class="target">Target: Neuromorphic Concept Allocation &lt; 5ms</div>
       
       <h2>Current Performance Status</h2>
       <div id="performance-data">
           <!-- Performance data will be inserted here -->
       </div>
       
       <h2>Statistical Analysis</h2>
       <p>All measurements include 95% confidence intervals and outlier detection.</p>
       
       <footer>
           <p>Generated: {}</p>
       </footer>
   </body>
   </html>
       "#, chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
       
       fs::write(output_path, html_content)?;
       Ok(())
   }
   ```
4. Update `Cargo.toml` dependencies:
   ```toml
   [dev-dependencies]
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   chrono = { version = "0.4", features = ["serde"] }
   ```
5. Test statistical analysis: `cargo test stats_analysis`
6. Commit analysis tools: `git add . && git commit -m "Add statistical analysis tools for 5ms allocation validation"`

## Expected Output
- Statistical analysis framework with confidence intervals
- Performance regression detection system
- HTML reporting utility
- Unit tests validating statistical functions

## Success Criteria
- [ ] Statistical analysis module compiles and passes tests
- [ ] Confidence interval calculation functional
- [ ] <5ms allocation target validation implemented
- [ ] Outlier detection working
- [ ] Regression detector operational
- [ ] HTML report generation functional
- [ ] All tests pass
- [ ] Tools committed to git

## Validation Commands
```batch
# Run statistical analysis tests
cargo test stats_analysis

# Test regression detection
cargo test regression_detector

# Check HTML generation
cargo test benchmark_reporter

# Verify compilation
cargo check --all-targets
```

## Next Task
task_141_validate_benchmark_environment_integration.md

## Notes
- Statistical significance ensures benchmark reliability
- 95% confidence intervals provide measurement certainty
- Outlier detection prevents measurement corruption
- Regression detection validates performance stability over time