# Task 037: Implement System Tools Baseline

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The system tools baseline provides performance comparison against standard command-line tools available on different platforms (find+grep, PowerShell Select-String, etc.).

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement baseline performance testing for standard system tools including find+grep combinations, Windows PowerShell Select-String, and other platform-specific search utilities. This provides comparison against the most basic search tools available on every system.

## Requirements
1. Extend `src/validation/baseline.rs` with system tools implementations
2. Support find+grep combinations on Unix-like systems
3. Implement Windows PowerShell Select-String baseline
4. Add Windows cmd findstr baseline
5. Handle cross-platform command execution differences
6. Implement proper error handling for missing tools
7. Support timeout and resource monitoring

## Expected Code Structure
```rust
// Add to baseline.rs file

use std::ffi::OsStr;

impl BaselineBenchmark {
    async fn run_find_grep(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        #[cfg(unix)]
        {
            self.run_unix_find_grep(query, start_time).await
        }
        #[cfg(windows)]
        {
            // On Windows, fallback to PowerShell or findstr
            self.run_windows_findstr(query, start_time).await
        }
        #[cfg(not(any(unix, windows)))]
        {
            Ok(BaselineResult {
                tool: BaselineTool::FindGrep,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("Platform not supported for find+grep".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            })
        }
    }
    
    #[cfg(unix)]
    async fn run_unix_find_grep(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        if !self.available_tools.get(&BaselineTool::FindGrep).unwrap_or(&false) {
            return Ok(BaselineResult {
                tool: BaselineTool::FindGrep,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("find or grep not available".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        let query_start = Instant::now();
        
        // Create the find+grep pipeline
        // find . -type f -name "*.txt" -o -name "*.md" -o -name "*.rs" | xargs grep -c "pattern"
        let mut find_cmd = AsyncCommand::new("find");
        find_cmd.arg(&self.test_data_dir)
             .args(&["-type", "f"])
             .args(&["-name", "*.txt"])
             .args(&["-o", "-name", "*.md"])
             .args(&["-o", "-name", "*.rs"])
             .args(&["-o", "-name", "*.py"])
             .args(&["-o", "-name", "*.js"])
             .args(&["-o", "-name", "*.json"])
             .args(&["-o", "-name", "*.log"])
             .args(&["-o", "-name", "*.csv"]);
        
        find_cmd.stdout(Stdio::piped())
                .stderr(Stdio::piped());
        
        let find_output = match tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds / 2), 
            find_cmd.output()
        ).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Ok(BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Find command failed: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                });
            },
            Err(_) => {
                return Ok(BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: Duration::from_secs(self.config.timeout_seconds / 2),
                    results_count: 0,
                    success: false,
                    error_message: Some("Find command timed out".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                });
            }
        };
        
        if !find_output.status.success() {
            return Ok(BaselineResult {
                tool: BaselineTool::FindGrep,
                query: query.to_string(),
                execution_time: query_start.elapsed(),
                results_count: 0,
                success: false,
                error_message: Some(String::from_utf8_lossy(&find_output.stderr).to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        // Parse file list and run grep on each file
        let file_list = String::from_utf8_lossy(&find_output.stdout);
        let files: Vec<&str> = file_list.lines().filter(|line| !line.trim().is_empty()).collect();
        
        if files.is_empty() {
            return Ok(BaselineResult {
                tool: BaselineTool::FindGrep,
                query: query.to_string(),
                execution_time: query_start.elapsed(),
                results_count: 0,
                success: true,
                error_message: None,
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        // Run grep on the files
        let mut grep_cmd = AsyncCommand::new("grep");
        grep_cmd.args(&["-c", query]); // Count matches
        grep_cmd.args(&files);
        grep_cmd.stdout(Stdio::piped())
                .stderr(Stdio::piped());
        
        let remaining_timeout = self.config.timeout_seconds - 
                               std::cmp::min(query_start.elapsed().as_secs(), self.config.timeout_seconds / 2);
        
        let grep_result = match tokio::time::timeout(
            Duration::from_secs(remaining_timeout), 
            grep_cmd.output()
        ).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let results_count = self.parse_grep_count_output(&output.stdout)?;
                
                // grep returns exit code 1 if no matches found, which is not an error for us
                let success = output.status.success() || output.status.code() == Some(1);
                
                BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message: if success { None } else { 
                        Some(String::from_utf8_lossy(&output.stderr).to_string())
                    },
                    memory_usage_mb: 0.0, // Memory monitoring for shell pipelines is complex
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Grep command failed: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: Duration::from_secs(self.config.timeout_seconds),
                    results_count: 0,
                    success: false,
                    error_message: Some("Grep command timed out".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(grep_result)
    }
    
    fn parse_grep_count_output(&self, output: &[u8]) -> Result<usize> {
        let output_str = String::from_utf8_lossy(output);
        let mut total_matches = 0;
        
        for line in output_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            // Parse lines like "filename:count" or just "count"
            if let Some(colon_pos) = line.rfind(':') {
                // Format: filename:count
                if let Ok(count) = line[colon_pos + 1..].parse::<usize>() {
                    total_matches += count;
                }
            } else {
                // Format: just count
                if let Ok(count) = line.parse::<usize>() {
                    total_matches += count;
                }
            }
        }
        
        Ok(total_matches)
    }
    
    async fn run_windows_powershell(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        if !self.available_tools.get(&BaselineTool::WindowsPowerShell).unwrap_or(&false) {
            return Ok(BaselineResult {
                tool: BaselineTool::WindowsPowerShell,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("PowerShell not available".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        }
        
        let query_start = Instant::now();
        
        // Create PowerShell command to search files
        let search_path = self.test_data_dir.to_string_lossy();
        let powershell_script = format!(
            r#"
            $totalMatches = 0
            Get-ChildItem -Path '{}' -Recurse -File -Include *.txt,*.md,*.rs,*.py,*.js,*.json,*.log,*.csv | 
            ForEach-Object {{
                $matches = Select-String -Path $_.FullName -Pattern '{}' -AllMatches
                if ($matches) {{
                    $totalMatches += $matches.Count
                }}
            }}
            Write-Output $totalMatches
            "#, 
            search_path, query.replace("'", "''") // Escape single quotes
        );
        
        let mut ps_cmd = AsyncCommand::new("powershell");
        ps_cmd.args(&["-NoProfile", "-Command", &powershell_script]);
        ps_cmd.stdout(Stdio::piped())
              .stderr(Stdio::piped());
        
        let result = match tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds), 
            ps_cmd.output()
        ).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let output_str = String::from_utf8_lossy(&output.stdout);
                
                let results_count = output_str.trim()
                    .parse::<usize>()
                    .unwrap_or(0);
                
                let success = output.status.success();
                
                BaselineResult {
                    tool: BaselineTool::WindowsPowerShell,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message: if success { None } else {
                        Some(String::from_utf8_lossy(&output.stderr).to_string())
                    },
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::WindowsPowerShell,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("PowerShell execution failed: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::WindowsPowerShell,
                    query: query.to_string(),
                    execution_time: Duration::from_secs(self.config.timeout_seconds),
                    results_count: 0,
                    success: false,
                    error_message: Some("PowerShell command timed out".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    #[cfg(windows)]
    async fn run_windows_findstr(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        let query_start = Instant::now();
        
        // Use Windows findstr command
        // findstr /S /I /C:"pattern" *.txt *.md *.rs *.py *.js *.json *.log *.csv
        let mut findstr_cmd = AsyncCommand::new("findstr");
        findstr_cmd.args(&[
            "/S",    // Search subdirectories
            "/I",    // Case insensitive
            "/N",    // Show line numbers (to count matches)
            "/C:" // Specify search string
        ]);
        findstr_cmd.arg(format!("/C:{}", query));
        
        // Add file patterns
        findstr_cmd.args(&["*.txt", "*.md", "*.rs", "*.py", "*.js", "*.json", "*.log", "*.csv"]);
        
        // Set working directory
        findstr_cmd.current_dir(&self.test_data_dir);
        findstr_cmd.stdout(Stdio::piped())
                   .stderr(Stdio::piped());
        
        let result = match tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds), 
            findstr_cmd.output()
        ).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let output_str = String::from_utf8_lossy(&output.stdout);
                
                // Count lines in output (each line represents a match)
                let results_count = output_str.lines()
                    .filter(|line| !line.trim().is_empty())
                    .count();
                
                // findstr returns 1 if no matches found, which is not an error
                let success = output.status.success() || output.status.code() == Some(1);
                
                BaselineResult {
                    tool: BaselineTool::FindGrep, // Using FindGrep as the tool type
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message: if success { None } else {
                        Some(String::from_utf8_lossy(&output.stderr).to_string())
                    },
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Findstr execution failed: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::FindGrep,
                    query: query.to_string(),
                    execution_time: Duration::from_secs(self.config.timeout_seconds),
                    results_count: 0,
                    success: false,
                    error_message: Some("Findstr command timed out".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    async fn run_system_search(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        // Try multiple system search approaches and return the best result
        let mut results = Vec::new();
        
        // Try find+grep first
        if self.available_tools.get(&BaselineTool::FindGrep).unwrap_or(&false) {
            if let Ok(result) = self.run_find_grep(query, start_time).await {
                results.push(result);
            }
        }
        
        // Try PowerShell on Windows
        #[cfg(windows)]
        if self.available_tools.get(&BaselineTool::WindowsPowerShell).unwrap_or(&false) {
            if let Ok(result) = self.run_windows_powershell(query, start_time).await {
                results.push(result);
            }
        }
        
        // Return the first successful result, or the first result if none succeeded
        let best_result = results.into_iter()
            .find(|r| r.success)
            .or_else(|| results.into_iter().next())
            .unwrap_or_else(|| BaselineResult {
                tool: BaselineTool::SystemSearch,
                query: query.to_string(),
                execution_time: Duration::from_secs(0),
                results_count: 0,
                success: false,
                error_message: Some("No system search tools available".to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            });
        
        Ok(BaselineResult {
            tool: BaselineTool::SystemSearch,
            ..best_result
        })
    }
    
    async fn run_windows_grep(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        // Windows doesn't have native grep, try alternatives
        #[cfg(windows)]
        {
            // Try git bash grep if available
            if let Ok(grep_path) = which::which("grep") {
                return self.run_git_bash_grep(&grep_path, query, start_time).await;
            }
            
            // Fallback to findstr
            self.run_windows_findstr(query, start_time).await
        }
        #[cfg(not(windows))]
        {
            // On non-Windows, delegate to regular grep
            self.run_unix_find_grep(query, start_time).await
        }
    }
    
    #[cfg(windows)]
    async fn run_git_bash_grep(&self, grep_path: &Path, query: &str, start_time: Instant) -> Result<BaselineResult> {
        let query_start = Instant::now();
        
        // Use grep from Git Bash installation
        let mut cmd = AsyncCommand::new(grep_path);
        cmd.args(&["-r", "-c", query]);
        cmd.arg(&self.test_data_dir);
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());
        
        let result = match tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds), 
            cmd.output()
        ).await {
            Ok(Ok(output)) => {
                let execution_time = query_start.elapsed();
                let results_count = self.parse_grep_count_output(&output.stdout)?;
                
                // grep returns 1 if no matches, which is not an error
                let success = output.status.success() || output.status.code() == Some(1);
                
                BaselineResult {
                    tool: BaselineTool::WindowsGrep,
                    query: query.to_string(),
                    execution_time,
                    results_count,
                    success,
                    error_message: if success { None } else {
                        Some(String::from_utf8_lossy(&output.stderr).to_string())
                    },
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Ok(Err(e)) => {
                BaselineResult {
                    tool: BaselineTool::WindowsGrep,
                    query: query.to_string(),
                    execution_time: query_start.elapsed(),
                    results_count: 0,
                    success: false,
                    error_message: Some(format!("Git Bash grep failed: {}", e)),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            },
            Err(_) => {
                BaselineResult {
                    tool: BaselineTool::WindowsGrep,
                    query: query.to_string(),
                    execution_time: Duration::from_secs(self.config.timeout_seconds),
                    results_count: 0,
                    success: false,
                    error_message: Some("Git Bash grep timed out".to_string()),
                    memory_usage_mb: 0.0,
                    index_time: None,
                }
            }
        };
        
        Ok(result)
    }
    
    // Test system tools functionality
    pub async fn test_system_tools(&self) -> Result<SystemToolsAvailability> {
        let mut availability = SystemToolsAvailability::default();
        
        // Test find+grep on Unix
        #[cfg(unix)]
        {
            availability.find_grep = self.test_find_grep_functionality().await;
        }
        
        // Test Windows tools
        #[cfg(windows)]
        {
            availability.powershell = self.test_powershell_functionality().await;
            availability.findstr = self.test_findstr_functionality().await;
            availability.git_bash_grep = self.test_git_bash_grep_functionality().await;
        }
        
        Ok(availability)
    }
    
    #[cfg(unix)]
    async fn test_find_grep_functionality(&self) -> bool {
        // Create a test file
        let test_file = self.config.temp_dir.join("system_tools_test.txt");
        if std::fs::write(&test_file, "test content for system tools\nmore test data").is_err() {
            return false;
        }
        
        // Test find command
        let find_result = AsyncCommand::new("find")
            .arg(&self.config.temp_dir)
            .args(&["-name", "system_tools_test.txt"])
            .output()
            .await;
        
        let find_works = find_result.map(|output| output.status.success()).unwrap_or(false);
        
        if !find_works {
            let _ = std::fs::remove_file(&test_file);
            return false;
        }
        
        // Test grep command
        let grep_result = AsyncCommand::new("grep")
            .args(&["-c", "test"])
            .arg(&test_file)
            .output()
            .await;
        
        let grep_works = grep_result.map(|output| output.status.success()).unwrap_or(false);
        
        let _ = std::fs::remove_file(&test_file);
        find_works && grep_works
    }
    
    #[cfg(windows)]
    async fn test_powershell_functionality(&self) -> bool {
        let test_result = AsyncCommand::new("powershell")
            .args(&["-NoProfile", "-Command", "Write-Output 'test'"])
            .output()
            .await;
        
        test_result.map(|output| output.status.success()).unwrap_or(false)
    }
    
    #[cfg(windows)]
    async fn test_findstr_functionality(&self) -> bool {
        let test_file = self.config.temp_dir.join("findstr_test.txt");
        if std::fs::write(&test_file, "test content").is_err() {
            return false;
        }
        
        let result = AsyncCommand::new("findstr")
            .args(&["/C:test"])
            .arg(&test_file)
            .output()
            .await;
        
        let works = result.map(|output| output.status.success()).unwrap_or(false);
        let _ = std::fs::remove_file(&test_file);
        works
    }
    
    #[cfg(windows)]
    async fn test_git_bash_grep_functionality(&self) -> bool {
        if let Ok(grep_path) = which::which("grep") {
            let test_file = self.config.temp_dir.join("grep_test.txt");
            if std::fs::write(&test_file, "test content").is_err() {
                return false;
            }
            
            let result = AsyncCommand::new(grep_path)
                .args(&["-c", "test"])
                .arg(&test_file)
                .output()
                .await;
            
            let works = result.map(|output| output.status.success()).unwrap_or(false);
            let _ = std::fs::remove_file(&test_file);
            works
        } else {
            false
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemToolsAvailability {
    pub find_grep: bool,
    pub powershell: bool,
    pub findstr: bool,
    pub git_bash_grep: bool,
}
```

## Success Criteria
- Find+grep pipeline works correctly on Unix systems
- Windows PowerShell Select-String baseline functions properly
- Windows findstr command executes and parses results correctly
- Cross-platform compatibility handles platform differences
- Output parsing accurately counts search results
- Error handling covers missing tools and command failures
- Timeout mechanisms prevent hanging processes

## Time Limit
10 minutes maximum