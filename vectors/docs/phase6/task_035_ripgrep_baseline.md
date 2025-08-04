# Task 035: Implement Ripgrep Baseline

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The ripgrep baseline provides performance comparison against the industry-standard command-line search tool. Ripgrep is known for its exceptional speed and serves as an excellent baseline for text search performance.

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the ripgrep performance baseline by executing ripgrep processes, parsing results, and measuring performance metrics. This implementation must handle Windows command execution, error cases, and provide accurate performance measurements.

## Requirements
1. Extend `src/validation/baseline.rs` with ripgrep implementation
2. Execute ripgrep processes and parse output
3. Measure execution time and memory usage accurately
4. Handle missing ripgrep installation gracefully
5. Support Windows command execution compatibility
6. Parse ripgrep output to count results
7. Implement timeout and error handling

## Expected Code Structure
```rust
// Add to baseline.rs file (extend existing implementations)

use std::process::Stdio;
use sysinfo::{Pid, PidExt, Process, ProcessExt, System, SystemExt};

impl BaselineBenchmark {
    async fn run_ripgrep(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        // Check if ripgrep is available
        if !self.available_tools.get(&BaselineTool::Ripgrep).unwrap_or(&false) {
            return Ok(BaselineResult {
                tool: BaselineTool::Ripgrep,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("Ripgrep not available on this system".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        let query_start = Instant::now();
        
        // Prepare ripgrep command
        let mut cmd = AsyncCommand::new("rg");
        
        // Configure ripgrep options for consistent behavior
        cmd.args(&[
            "--count",           // Count matches instead of showing content
            "--no-heading",      // Don't show file headers
            "--no-line-number",  // Don't show line numbers  
            "--no-filename",     // Don't show filenames in output
            "--text",            // Treat all files as text
            "--no-ignore",       // Don't use .gitignore files
            "--hidden",          // Search hidden files
            "--follow",          // Follow symlinks
            "--threads", "1",    // Single thread for consistent timing
        ]);
        
        // Add the search pattern
        cmd.arg(query);
        
        // Add the search directory
        cmd.arg(&self.test_data_dir);
        
        // Configure process
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .kill_on_drop(true);
        
        // Start system monitoring
        let mut system = System::new();
        let initial_memory = self.get_system_memory_usage(&mut system);
        
        // Execute ripgrep with timeout
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);
        
        let result = match tokio::time::timeout(timeout_duration, cmd.output()).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                
                // Parse output to count results
                let results_count = self.parse_ripgrep_output(&output.stdout)?;
                
                // Measure memory usage (approximate)
                let final_memory = self.get_system_memory_usage(&mut system);
                let memory_used = (final_memory - initial_memory).max(0.0);
                
                let success = output.status.success();
                let error_message = if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                };
                
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message,
                    memory_usage_mb: memory_used,
                    index_time: None, // Ripgrep doesn't have index time
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Failed to execute ripgrep: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: timeout_duration,
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Ripgrep timed out after {} seconds", self.config.timeout_seconds)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    fn parse_ripgrep_output(&self, output: &[u8]) -> Result<usize> {
        let output_str = String::from_utf8_lossy(output);
        
        // With --count flag, ripgrep outputs one number per file that has matches
        // We need to sum all the counts
        let mut total_matches = 0;
        
        for line in output_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            // Try to parse the line as a number
            match line.parse::<usize>() {
                Ok(count) => total_matches += count,
                Err(_) => {
                    // If we can't parse as number, this might be an error or unexpected format
                    // Check if it looks like a filename:count format
                    if let Some(colon_pos) = line.rfind(':') {
                        if let Ok(count) = line[colon_pos + 1..].parse::<usize>() {
                            total_matches += count;
                        }
                    }
                }
            }
        }
        
        Ok(total_matches)
    }
    
    fn get_system_memory_usage(&self, system: &mut System) -> f64 {
        system.refresh_memory();
        let used_memory = system.used_memory();
        used_memory as f64 / (1024.0 * 1024.0) // Convert to MB
    }
    
    // Alternative implementation for more accurate per-process memory tracking
    async fn run_ripgrep_with_process_monitoring(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        if !self.available_tools.get(&BaselineTool::Ripgrep).unwrap_or(&false) {
            return Ok(BaselineResult {
                tool: BaselineTool::Ripgrep,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("Ripgrep not available on this system".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        let query_start = Instant::now();
        
        // Create the command
        let mut cmd = AsyncCommand::new("rg");
        cmd.args(&[
            "--count",
            "--no-heading",
            "--no-line-number",
            "--no-filename", 
            "--text",
            "--no-ignore",
            "--hidden",
            "--follow",
            "--threads", "1",
            query,
        ]);
        cmd.arg(&self.test_data_dir);
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .kill_on_drop(true);
        
        // Start the process
        let mut child = cmd.spawn()
            .context("Failed to spawn ripgrep process")?;
        
        // Monitor memory usage in a separate task
        let memory_monitor = {
            let pid = child.id();
            tokio::spawn(async move {
                let mut max_memory = 0.0f64;
                let mut system = System::new();
                
                // Monitor for up to the timeout duration
                let monitor_duration = Duration::from_secs(30); // Reasonable monitor time
                let start_monitor = Instant::now();
                
                while start_monitor.elapsed() < monitor_duration {
                    system.refresh_processes();
                    
                    if let Some(process_id) = pid {
                        if let Some(process) = system.process(Pid::from_u32(process_id)) {
                            let memory_mb = process.memory() as f64 / (1024.0 * 1024.0);
                            max_memory = max_memory.max(memory_mb);
                        }
                    }
                    
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                
                max_memory
            })
        };
        
        // Wait for the process with timeout
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);
        
        let result = match tokio::time::timeout(timeout_duration, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let max_memory = memory_monitor.await.unwrap_or(0.0);
                
                let results_count = self.parse_ripgrep_output(&output.stdout)?;
                let success = output.status.success();
                let error_message = if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                };
                
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message,
                    memory_usage_mb: max_memory,
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                memory_monitor.abort();
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Ripgrep process error: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                // Kill the child process on timeout
                let _ = child.kill().await;
                memory_monitor.abort();
                
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: timeout_duration,
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Ripgrep timed out after {} seconds", self.config.timeout_seconds)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    // Windows-specific ripgrep execution
    #[cfg(windows)]
    async fn run_ripgrep_windows(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        use std::os::windows::process::CommandExt;
        
        if !self.available_tools.get(&BaselineTool::Ripgrep).unwrap_or(&false) {
            return Ok(BaselineResult {
                tool: BaselineTool::Ripgrep,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("Ripgrep not available on this system".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        let query_start = Instant::now();
        
        // Try to find ripgrep.exe explicitly on Windows
        let rg_cmd = which::which("rg")
            .or_else(|_| which::which("rg.exe"))
            .unwrap_or_else(|_| "rg".into());
        
        let mut cmd = AsyncCommand::new(rg_cmd);
        
        // Use Windows-specific process creation flags if needed
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        
        cmd.args(&[
            "--count",
            "--no-heading", 
            "--no-line-number",
            "--no-filename",
            "--text",
            "--no-ignore",
            "--hidden", 
            "--follow",
            "--threads", "1",
        ]);
        
        cmd.arg(query);
        cmd.arg(&self.test_data_dir);
        
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .kill_on_drop(true);
        
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);
        
        let result = match tokio::time::timeout(timeout_duration, cmd.output()).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let results_count = self.parse_ripgrep_output(&output.stdout)?;
                
                let success = output.status.success();
                let error_message = if !success {
                    Some(String::from_utf8_lossy(&output.stderr).to_string())
                } else {
                    None
                };
                
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message,
                    memory_usage_mb: 0.0, // Windows memory monitoring is more complex
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Failed to execute ripgrep on Windows: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::Ripgrep,
                    query: query.to_string(),
                    execution_time: timeout_duration,
                    results_count: 0,
                    success: false,
                    error_message: Some("Ripgrep timed out on Windows".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    // Public method to test ripgrep functionality
    pub async fn test_ripgrep_installation(&self) -> Result<bool> {
        let test_query = "test";
        
        // Create a small test file
        let test_file = self.config.temp_dir.join("ripgrep_test.txt");
        std::fs::write(&test_file, "This is a test file for ripgrep validation\ntest line\nanother line")?;
        
        let mut cmd = AsyncCommand::new("rg");
        cmd.args(&["--count", "test"]);
        cmd.arg(&test_file);
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());
        
        let result = match tokio::time::timeout(Duration::from_secs(5), cmd.output()).await {
            Ok(Ok(output)) => {
                let success = output.status.success();
                if success {
                    let count_str = String::from_utf8_lossy(&output.stdout);
                    // Should find at least 2 matches for "test"
                    if let Ok(count) = count_str.trim().parse::<usize>() {
                        success && count >= 2
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            _ => false,
        };
        
        // Cleanup
        let _ = std::fs::remove_file(&test_file);
        
        Ok(result)
    }
}

// Update the main run_ripgrep method to use the best implementation for the platform
impl BaselineBenchmark {
    async fn run_ripgrep(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        #[cfg(windows)]
        {
            self.run_ripgrep_windows(query, start_time).await
        }
        #[cfg(not(windows))]
        {
            self.run_ripgrep_with_process_monitoring(query, start_time).await
        }
    }
}
```

## Dependencies to Add
```toml
[dependencies]
which = "4.4"  # For finding executables on Windows
```

## Success Criteria
- Ripgrep execution works on both Windows and Unix platforms
- Output parsing correctly counts search results
- Memory usage monitoring provides reasonable estimates
- Timeout handling prevents hanging processes
- Error handling covers all failure modes (missing binary, process errors, timeouts)
- Performance measurements are accurate and consistent
- Process cleanup works correctly

## Time Limit
10 minutes maximum