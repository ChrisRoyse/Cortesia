# Task 034: Create BaselineResults Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The BaselineResults struct provides comprehensive metrics and comparative analysis capabilities for baseline benchmark results, including statistical significance testing and performance ratio calculations.

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `BaselineResults` struct that aggregates and analyzes baseline benchmark results. This struct performs comparative analysis, calculates statistical significance, and provides reporting capabilities for baseline performance data.

## Requirements
1. Extend `src/validation/baseline.rs` with BaselineResults struct
2. Implement comprehensive metrics aggregation
3. Add statistical significance testing
4. Create performance ratio calculations
5. Implement export capabilities for reporting
6. Add confidence interval calculations
7. Support for multiple baseline tool comparisons

## Expected Code Structure
```rust
// Add to baseline.rs file

use std::collections::BTreeMap;
use statrs::statistics::{Statistics, OrderStatistics};
use statrs::distribution::{StudentsT, ContinuousCDF};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResults {
    pub test_name: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_info: SystemInfo,
    pub tool_results: BTreeMap<BaselineTool, ToolResults>,
    pub comparative_analysis: ComparativeAnalysis,
    pub statistical_summary: StatisticalSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
    pub test_data_size_mb: f64,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResults {
    pub tool: BaselineTool,
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub average_latency_ms: f64,
    pub median_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub std_dev_latency_ms: f64,
    pub average_memory_mb: f64,
    pub success_rate: f64,
    pub throughput_qps: f64,
    pub total_execution_time: Duration,
    pub individual_results: Vec<BaselineResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub fastest_tool: BaselineTool,
    pub most_reliable_tool: BaselineTool,
    pub memory_efficient_tool: BaselineTool,
    pub performance_ratios: BTreeMap<BaselineTool, PerformanceRatio>,
    pub statistical_significance: BTreeMap<(BaselineTool, BaselineTool), SignificanceTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRatio {
    pub tool: BaselineTool,
    pub speed_ratio: f64,          // Compared to fastest
    pub memory_ratio: f64,         // Compared to most memory efficient
    pub reliability_ratio: f64,    // Compared to most reliable
    pub overall_score: f64,        // Weighted composite score
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTest {
    pub tool_a: BaselineTool,
    pub tool_b: BaselineTool,
    pub p_value: f64,
    pub significant: bool,
    pub confidence_level: f64,
    pub effect_size: f64,
    pub conclusion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub total_measurements: usize,
    pub confidence_interval_95: (f64, f64),
    pub coefficient_of_variation: f64,
    pub measurement_stability: String,
    pub recommended_sample_size: usize,
}

impl BaselineResults {
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            timestamp: chrono::Utc::now(),
            system_info: SystemInfo::collect(),
            tool_results: BTreeMap::new(),
            comparative_analysis: ComparativeAnalysis::default(),
            statistical_summary: StatisticalSummary::default(),
        }
    }
    
    pub fn add_tool_results(&mut self, results: Vec<BaselineResult>) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }
        
        let tool = results[0].tool;
        let tool_result = self.aggregate_tool_results(results)?;
        self.tool_results.insert(tool, tool_result);
        
        // Recalculate comparative analysis
        self.calculate_comparative_analysis()?;
        self.calculate_statistical_summary()?;
        
        Ok(())
    }
    
    fn aggregate_tool_results(&self, results: Vec<BaselineResult>) -> Result<ToolResults> {
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if successful_results.is_empty() {
            return Ok(ToolResults {
                tool: results[0].tool,
                total_queries: results.len(),
                successful_queries: 0,
                failed_queries: results.len(),
                average_latency_ms: 0.0,
                median_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                min_latency_ms: 0.0,
                max_latency_ms: 0.0,
                std_dev_latency_ms: 0.0,
                average_memory_mb: 0.0,
                success_rate: 0.0,
                throughput_qps: 0.0,
                total_execution_time: Duration::from_secs(0),
                individual_results: results,
            });
        }
        
        let latencies: Vec<f64> = successful_results.iter()
            .map(|r| r.execution_time.as_secs_f64() * 1000.0)
            .collect();
        
        let memory_usage: Vec<f64> = successful_results.iter()
            .map(|r| r.memory_usage_mb)
            .collect();
        
        let total_time: Duration = results.iter()
            .map(|r| r.execution_time)
            .sum();
        
        Ok(ToolResults {
            tool: results[0].tool,
            total_queries: results.len(),
            successful_queries: successful_results.len(),
            failed_queries: results.len() - successful_results.len(),
            average_latency_ms: latencies.mean(),
            median_latency_ms: latencies.median(),
            p95_latency_ms: latencies.quantile(0.95),
            p99_latency_ms: latencies.quantile(0.99),
            min_latency_ms: latencies.min(),
            max_latency_ms: latencies.max(),
            std_dev_latency_ms: latencies.std_dev(),
            average_memory_mb: memory_usage.mean(),
            success_rate: (successful_results.len() as f64 / results.len() as f64) * 100.0,
            throughput_qps: if total_time.as_secs_f64() > 0.0 {
                successful_results.len() as f64 / total_time.as_secs_f64()
            } else {
                0.0
            },
            total_execution_time: total_time,
            individual_results: results,
        })
    }
    
    fn calculate_comparative_analysis(&mut self) -> Result<()> {
        if self.tool_results.is_empty() {
            return Ok(());
        }
        
        // Find best performers
        let fastest_tool = self.tool_results.iter()
            .min_by(|a, b| a.1.average_latency_ms.partial_cmp(&b.1.average_latency_ms).unwrap())
            .map(|(tool, _)| *tool)
            .unwrap();
        
        let most_reliable_tool = self.tool_results.iter()
            .max_by(|a, b| a.1.success_rate.partial_cmp(&b.1.success_rate).unwrap())
            .map(|(tool, _)| *tool)
            .unwrap();
        
        let memory_efficient_tool = self.tool_results.iter()
            .min_by(|a, b| a.1.average_memory_mb.partial_cmp(&b.1.average_memory_mb).unwrap())
            .map(|(tool, _)| *tool)
            .unwrap();
        
        // Calculate performance ratios
        let mut performance_ratios = BTreeMap::new();
        let fastest_time = self.tool_results[&fastest_tool].average_latency_ms;
        let most_reliable_rate = self.tool_results[&most_reliable_tool].success_rate;
        let lowest_memory = self.tool_results[&memory_efficient_tool].average_memory_mb;
        
        for (tool, results) in &self.tool_results {
            let speed_ratio = results.average_latency_ms / fastest_time;
            let memory_ratio = results.average_memory_mb / lowest_memory;
            let reliability_ratio = results.success_rate / most_reliable_rate;
            
            // Composite score (lower is better for speed/memory, higher for reliability)
            let overall_score = (0.4 / speed_ratio) + (0.3 * reliability_ratio) + (0.3 / memory_ratio);
            
            performance_ratios.insert(*tool, PerformanceRatio {
                tool: *tool,
                speed_ratio,
                memory_ratio,
                reliability_ratio,
                overall_score,
            });
        }
        
        // Calculate statistical significance
        let mut significance_tests = BTreeMap::new();
        let tools: Vec<_> = self.tool_results.keys().collect();
        
        for i in 0..tools.len() {
            for j in (i + 1)..tools.len() {
                let tool_a = *tools[i];
                let tool_b = *tools[j];
                
                if let Some(test) = self.perform_t_test(tool_a, tool_b) {
                    significance_tests.insert((tool_a, tool_b), test);
                }
            }
        }
        
        self.comparative_analysis = ComparativeAnalysis {
            fastest_tool,
            most_reliable_tool,
            memory_efficient_tool,
            performance_ratios,
            statistical_significance: significance_tests,
        };
        
        Ok(())
    }
    
    fn perform_t_test(&self, tool_a: BaselineTool, tool_b: BaselineTool) -> Option<SignificanceTest> {
        let results_a = &self.tool_results[&tool_a];
        let results_b = &self.tool_results[&tool_b];
        
        let latencies_a: Vec<f64> = results_a.individual_results.iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time.as_secs_f64() * 1000.0)
            .collect();
        
        let latencies_b: Vec<f64> = results_b.individual_results.iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time.as_secs_f64() * 1000.0)
            .collect();
        
        if latencies_a.len() < 2 || latencies_b.len() < 2 {
            return None;
        }
        
        // Two-sample t-test
        let mean_a = latencies_a.mean();
        let mean_b = latencies_b.mean();
        let var_a = latencies_a.variance();
        let var_b = latencies_b.variance();
        let n_a = latencies_a.len() as f64;
        let n_b = latencies_b.len() as f64;
        
        // Pooled standard error
        let pooled_se = ((var_a / n_a) + (var_b / n_b)).sqrt();
        let t_stat = (mean_a - mean_b) / pooled_se;
        
        // Degrees of freedom (Welch's approximation)
        let df = ((var_a / n_a + var_b / n_b).powi(2)) / 
                 ((var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0));
        
        // Calculate p-value
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
        
        let significant = p_value < 0.05;
        let effect_size = (mean_a - mean_b).abs() / ((var_a + var_b) / 2.0).sqrt();
        
        let conclusion = if significant {
            if mean_a < mean_b {
                format!("{:?} is significantly faster than {:?}", tool_a, tool_b)
            } else {
                format!("{:?} is significantly faster than {:?}", tool_b, tool_a)
            }
        } else {
            format!("No significant difference between {:?} and {:?}", tool_a, tool_b)
        };
        
        Some(SignificanceTest {
            tool_a,
            tool_b,
            p_value,
            significant,
            confidence_level: 0.95,
            effect_size,
            conclusion,
        })
    }
    
    fn calculate_statistical_summary(&mut self) -> Result<()> {
        let all_latencies: Vec<f64> = self.tool_results.values()
            .flat_map(|tr| tr.individual_results.iter())
            .filter(|r| r.success)
            .map(|r| r.execution_time.as_secs_f64() * 1000.0)
            .collect();
        
        if all_latencies.is_empty() {
            self.statistical_summary = StatisticalSummary::default();
            return Ok(());
        }
        
        let mean = all_latencies.mean();
        let std_dev = all_latencies.std_dev();
        let n = all_latencies.len() as f64;
        
        // 95% confidence interval for mean
        let t_critical = 1.96; // Approximate for large n
        let margin_error = t_critical * (std_dev / n.sqrt());
        let confidence_interval = (mean - margin_error, mean + margin_error);
        
        let coefficient_of_variation = std_dev / mean;
        
        let measurement_stability = match coefficient_of_variation {
            cv if cv < 0.1 => "Excellent".to_string(),
            cv if cv < 0.2 => "Good".to_string(),
            cv if cv < 0.3 => "Moderate".to_string(),
            _ => "Poor".to_string(),
        };
        
        // Recommended sample size for 5% margin of error
        let recommended_sample_size = ((1.96 * std_dev / (0.05 * mean)).powi(2) as usize).max(30);
        
        self.statistical_summary = StatisticalSummary {
            total_measurements: all_latencies.len(),
            confidence_interval_95: confidence_interval,
            coefficient_of_variation,
            measurement_stability,
            recommended_sample_size,
        };
        
        Ok(())
    }
    
    pub fn export_to_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize baseline results to JSON")?;
        
        std::fs::write(path, json)
            .context("Failed to write baseline results JSON file")?;
        
        Ok(())
    }
    
    pub fn export_to_csv(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut csv_content = String::new();
        csv_content.push_str("Tool,Query,LatencyMs,ResultsCount,Success,MemoryMB\n");
        
        for tool_result in self.tool_results.values() {
            for result in &tool_result.individual_results {
                csv_content.push_str(&format!(
                    "{:?},{},{},{},{},{}\n",
                    result.tool,
                    result.query.replace(',', ';'), // Escape commas
                    result.execution_time.as_secs_f64() * 1000.0,
                    result.results_count,
                    result.success,
                    result.memory_usage_mb
                ));
            }
        }
        
        std::fs::write(path, csv_content)
            .context("Failed to write baseline results CSV file")?;
        
        Ok(())
    }
    
    pub fn print_summary(&self) {
        println!("\n=== Baseline Benchmark Results Summary ===");
        println!("Test: {}", self.test_name);
        println!("Timestamp: {}", self.timestamp);
        println!("System: {} ({} cores, {:.1}GB RAM)", 
                 self.system_info.os, self.system_info.cpu_cores, self.system_info.memory_gb);
        
        println!("\n--- Tool Performance ---");
        for (tool, results) in &self.tool_results {
            println!("{:?}:", tool);
            println!("  Success Rate: {:.1}%", results.success_rate);
            println!("  Avg Latency: {:.2}ms", results.average_latency_ms);
            println!("  P95 Latency: {:.2}ms", results.p95_latency_ms);
            println!("  Throughput: {:.2} QPS", results.throughput_qps);
            println!("  Memory: {:.2}MB", results.average_memory_mb);
        }
        
        println!("\n--- Comparative Analysis ---");
        println!("Fastest: {:?}", self.comparative_analysis.fastest_tool);
        println!("Most Reliable: {:?}", self.comparative_analysis.most_reliable_tool);
        println!("Memory Efficient: {:?}", self.comparative_analysis.memory_efficient_tool);
        
        println!("\n--- Statistical Summary ---");
        println!("Total Measurements: {}", self.statistical_summary.total_measurements);
        println!("Measurement Stability: {}", self.statistical_summary.measurement_stability);
        println!("Coefficient of Variation: {:.3}", self.statistical_summary.coefficient_of_variation);
    }
}

impl Default for ComparativeAnalysis {
    fn default() -> Self {
        Self {
            fastest_tool: BaselineTool::Ripgrep,
            most_reliable_tool: BaselineTool::Ripgrep,
            memory_efficient_tool: BaselineTool::Ripgrep,
            performance_ratios: BTreeMap::new(),
            statistical_significance: BTreeMap::new(),
        }
    }
}

impl Default for StatisticalSummary {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            confidence_interval_95: (0.0, 0.0),
            coefficient_of_variation: 0.0,
            measurement_stability: "Unknown".to_string(),
            recommended_sample_size: 30,
        }
    }
}

impl SystemInfo {
    fn collect() -> Self {
        use sysinfo::{System, SystemExt};
        
        let mut sys = System::new_all();
        sys.refresh_all();
        
        Self {
            os: sys.long_os_version().unwrap_or_else(|| "Unknown".to_string()),
            cpu_cores: sys.cpus().len(),
            memory_gb: sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            rust_version: env!("RUSTC_VERSION").to_string(),
            test_data_size_mb: 0.0, // Will be filled by caller
            file_count: 0, // Will be filled by caller
        }
    }
}
```

## Dependencies to Add
```toml
[dependencies]
statrs = "0.16"
chrono = { version = "0.4", features = ["serde"] }
```

## Success Criteria
- BaselineResults struct compiles without errors
- Statistical analysis functions work correctly
- Comparative analysis identifies best performers
- Export functionality generates valid JSON and CSV
- T-test implementation provides meaningful significance testing
- Summary reporting is comprehensive and clear
- Memory usage calculations are accurate

## Time Limit
10 minutes maximum