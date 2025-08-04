# Task 033: Create BaselineBenchmark Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The baseline benchmarking system measures performance against established tools to validate the new system meets or exceeds existing capabilities. This provides comparative analysis against industry-standard tools like ripgrep, find+grep, and standalone Tantivy.

## Project Structure
```
src/
  validation/
    baseline.rs  <- Create this file
  lib.rs
```

## Task Description
Create the `BaselineBenchmark` struct that orchestrates performance testing against established baseline tools. This struct manages multiple baseline tools, configures test scenarios, and integrates with the existing PerformanceBenchmark framework for comprehensive comparative analysis.

## Requirements
1. Create `src/validation/baseline.rs` with BaselineBenchmark struct
2. Support multiple baseline tools (ripgrep, find+grep, Tantivy, system search)
3. Implement configuration for different test scenarios
4. Integrate with existing PerformanceBenchmark infrastructure
5. Ensure Windows and cross-platform compatibility
6. Add async execution support for parallel baseline testing
7. Create baseline tool detection and validation

## Expected Code Structure
```rust
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::process::Command;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, bail};
use tokio::process::Command as AsyncCommand;

// Import existing performance infrastructure
use crate::validation::performance::{PerformanceBenchmark, PerformanceMetrics};

#[derive(Debug, Clone)]
pub struct BaselineBenchmark {
    test_data_dir: PathBuf,
    baseline_tools: Vec<BaselineTool>,
    config: BaselineConfig,
    available_tools: HashMap<BaselineTool, bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaselineTool {
    Ripgrep,
    FindGrep,
    TantivyStandalone,
    SystemSearch,
    WindowsPowerShell,
    WindowsGrep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub timeout_seconds: u64,
    pub warmup_runs: usize,
    pub measurement_runs: usize,
    pub parallel_execution: bool,
    pub include_index_time: bool,
    pub temp_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    pub tool: BaselineTool,
    pub query: String,
    pub execution_time: Duration,
    pub results_count: usize,
    pub success: bool,
    pub error_message: Option<String>,
    pub memory_usage_mb: f64,
    pub index_time: Option<Duration>,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 30,
            warmup_runs: 3,
            measurement_runs: 10,
            parallel_execution: true,
            include_index_time: false,
            temp_dir: std::env::temp_dir().join("baseline_benchmarks"),
        }
    }
}

impl BaselineBenchmark {
    pub async fn new(test_data_dir: PathBuf) -> Result<Self> {
        let config = BaselineConfig::default();
        
        // Create temp directory
        std::fs::create_dir_all(&config.temp_dir)
            .context("Failed to create baseline temp directory")?;
        
        let mut benchmark = Self {
            test_data_dir,
            baseline_tools: Vec::new(),
            config,
            available_tools: HashMap::new(),
        };
        
        // Detect available tools
        benchmark.detect_available_tools().await?;
        
        Ok(benchmark)
    }
    
    pub async fn with_config(test_data_dir: PathBuf, config: BaselineConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.temp_dir)
            .context("Failed to create baseline temp directory")?;
        
        let mut benchmark = Self {
            test_data_dir,
            baseline_tools: Vec::new(),
            config,
            available_tools: HashMap::new(),
        };
        
        benchmark.detect_available_tools().await?;
        Ok(benchmark)
    }
    
    async fn detect_available_tools(&mut self) -> Result<()> {
        let tools_to_check = vec![
            (BaselineTool::Ripgrep, vec!["rg", "--version"]),
            (BaselineTool::FindGrep, vec!["grep", "--version"]),
            (BaselineTool::SystemSearch, vec!["find", "--version"]),
        ];
        
        for (tool, cmd_args) in tools_to_check {
            let available = self.check_tool_availability(&cmd_args).await;
            self.available_tools.insert(tool, available);
        }
        
        // Windows-specific tools
        if cfg!(windows) {
            let powershell_available = self.check_tool_availability(&["powershell", "-Command", "Get-Command", "Select-String"]).await;
            self.available_tools.insert(BaselineTool::WindowsPowerShell, powershell_available);
        }
        
        // Always mark Tantivy as available (we control this)
        self.available_tools.insert(BaselineTool::TantivyStandalone, true);
        
        println!("Available baseline tools:");
        for (tool, available) in &self.available_tools {
            println!("  {:?}: {}", tool, if *available { "✓" } else { "✗" });
        }
        
        Ok(())
    }
    
    async fn check_tool_availability(&self, cmd_args: &[&str]) -> bool {
        if cmd_args.is_empty() {
            return false;
        }
        
        let result = AsyncCommand::new(cmd_args[0])
            .args(&cmd_args[1..])
            .output()
            .await;
        
        match result {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
    
    pub fn set_tools(&mut self, tools: Vec<BaselineTool>) -> Result<()> {
        // Validate that requested tools are available
        for tool in &tools {
            if !self.available_tools.get(tool).unwrap_or(&false) {
                bail!("Baseline tool {:?} is not available on this system", tool);
            }
        }
        
        self.baseline_tools = tools;
        Ok(())
    }
    
    pub fn get_available_tools(&self) -> Vec<&BaselineTool> {
        self.available_tools
            .iter()
            .filter_map(|(tool, available)| if *available { Some(tool) } else { None })
            .collect()
    }
    
    pub async fn run_all_baselines(&self, queries: &[String]) -> Result<Vec<BaselineResult>> {
        if self.baseline_tools.is_empty() {
            bail!("No baseline tools configured. Use set_tools() first.");
        }
        
        let mut all_results = Vec::new();
        
        println!("Running baseline benchmarks with {} tools and {} queries...", 
                 self.baseline_tools.len(), queries.len());
        
        for tool in &self.baseline_tools {
            println!("Running baseline for {:?}...", tool);
            
            for query in queries {
                // Warmup runs
                for _ in 0..self.config.warmup_runs {
                    let _ = self.run_single_baseline(*tool, query).await;
                }
                
                // Measurement runs
                let mut tool_results = Vec::new();
                for _ in 0..self.config.measurement_runs {
                    match self.run_single_baseline(*tool, query).await {
                        Ok(result) => tool_results.push(result),
                        Err(e) => {
                            tool_results.push(BaselineResult {
                                tool: *tool,
                                query: query.clone(),
                                execution_time: Duration::from_secs(0),
                                results_count: 0,
                                success: false,
                                error_message: Some(e.to_string()),
                                memory_usage_mb: 0.0,
                                index_time: None,
                            });
                        }
                    }
                }
                
                // Calculate average result
                if !tool_results.is_empty() {
                    let avg_result = self.calculate_average_result(tool_results);
                    all_results.push(avg_result);
                }
            }
        }
        
        Ok(all_results)
    }
    
    pub async fn run_specific_baseline(&self, tool: BaselineTool, queries: &[String]) -> Result<Vec<BaselineResult>> {
        if !self.available_tools.get(&tool).unwrap_or(&false) {
            bail!("Baseline tool {:?} is not available", tool);
        }
        
        let mut results = Vec::new();
        
        for query in queries {
            // Warmup
            for _ in 0..self.config.warmup_runs {
                let _ = self.run_single_baseline(tool, query).await;
            }
            
            // Measurements
            let mut measurements = Vec::new();
            for _ in 0..self.config.measurement_runs {
                match self.run_single_baseline(tool, query).await {
                    Ok(result) => measurements.push(result),
                    Err(e) => {
                        measurements.push(BaselineResult {
                            tool,
                            query: query.clone(),
                            execution_time: Duration::from_secs(0),
                            results_count: 0,
                            success: false,
                            error_message: Some(e.to_string()),
                            memory_usage_mb: 0.0,
                            index_time: None,
                        });
                    }
                }
            }
            
            if !measurements.is_empty() {
                results.push(self.calculate_average_result(measurements));
            }
        }
        
        Ok(results)
    }
    
    async fn run_single_baseline(&self, tool: BaselineTool, query: &str) -> Result<BaselineResult> {
        let start_time = Instant::now();
        
        match tool {
            BaselineTool::Ripgrep => self.run_ripgrep(query, start_time).await,
            BaselineTool::FindGrep => self.run_find_grep(query, start_time).await,
            BaselineTool::TantivyStandalone => self.run_tantivy_standalone(query, start_time).await,
            BaselineTool::SystemSearch => self.run_system_search(query, start_time).await,
            BaselineTool::WindowsPowerShell => self.run_windows_powershell(query, start_time).await,
            BaselineTool::WindowsGrep => self.run_windows_grep(query, start_time).await,
        }
    }
    
    // Placeholder methods - will be implemented in subsequent tasks
    async fn run_ripgrep(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_035
        bail!("Ripgrep baseline not yet implemented")
    }
    
    async fn run_find_grep(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_037
        bail!("Find+grep baseline not yet implemented")
    }
    
    async fn run_tantivy_standalone(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_036
        bail!("Tantivy standalone baseline not yet implemented")
    }
    
    async fn run_system_search(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_037
        bail!("System search baseline not yet implemented")
    }
    
    async fn run_windows_powershell(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_037
        bail!("Windows PowerShell baseline not yet implemented")
    }
    
    async fn run_windows_grep(&self, _query: &str, _start_time: Instant) -> Result<BaselineResult> {
        // Implementation in task_037
        bail!("Windows grep baseline not yet implemented")
    }
    
    fn calculate_average_result(&self, results: Vec<BaselineResult>) -> BaselineResult {
        if results.is_empty() {
            return BaselineResult {
                tool: BaselineTool::Ripgrep, // Default
                query: String::new(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("No results to average".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            };
        }
        
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if successful_results.is_empty() {
            return results[0].clone(); // Return first error
        }
        
        let avg_execution_time = Duration::from_nanos(
            successful_results.iter()
                .map(|r| r.execution_time.as_nanos() as u64)
                .sum::<u64>() / successful_results.len() as u64
        );
        
        let avg_memory = successful_results.iter()
            .map(|r| r.memory_usage_mb)
            .sum::<f64>() / successful_results.len() as f64;
        
        BaselineResult {
            tool: results[0].tool,
            query: results[0].query.clone(),
            execution_time: avg_execution_time,
            results_count: successful_results[0].results_count, // Assume consistent
            success: true,
            error_message: None,
            memory_usage_mb: avg_memory,
            index_time: successful_results[0].index_time,
        }
    }
    
    pub fn cleanup(&self) -> Result<()> {
        if self.config.temp_dir.exists() {
            std::fs::remove_dir_all(&self.config.temp_dir)
                .context("Failed to cleanup baseline temp directory")?;
        }
        Ok(())
    }
}

impl Drop for BaselineBenchmark {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}
```

## Dependencies to Add
```toml
[dependencies]
tokio = { version = "1.0", features = ["process", "fs", "time"] }
sysinfo = "0.29"
```

## Success Criteria
- BaselineBenchmark struct compiles without errors
- Tool detection works across Windows and Unix platforms
- Configuration system supports different test scenarios
- Integration points prepared for specific tool implementations
- Async execution framework established
- Error handling covers tool availability and execution failures
- Cleanup mechanisms prevent temp file accumulation

## Time Limit
10 minutes maximum