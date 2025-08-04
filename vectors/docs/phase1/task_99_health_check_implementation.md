# Task 99: Health Check and Readiness Probe Implementation

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 98  
**Dependencies:** Monitoring system in place

## Objective
Implement comprehensive health checks and readiness probes for production deployment.

## Context
You're adding health check endpoints for the vector search system. These checks will be used by orchestration systems (Kubernetes, Docker) to determine if the service is healthy and ready to accept traffic. The checks must verify all critical components: Tantivy index, file system access, memory availability, and search functionality.

## Task Details

### What You Need to Do

1. **Create health check module** (`src/health.rs`):
```rust
use std::path::Path;
use std::time::{Duration, Instant};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub checks: Vec<ComponentCheck>,
    pub timestamp: String,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub response_time_ms: f64,
}

pub struct HealthChecker {
    start_time: Instant,
    index_path: PathBuf,
}

impl HealthChecker {
    pub fn new(index_path: PathBuf) -> Self {
        Self {
            start_time: Instant::now(),
            index_path,
        }
    }
    
    pub async fn check_health(&self) -> Result<HealthCheck> {
        let mut checks = Vec::new();
        let mut overall_status = HealthStatus::Healthy;
        
        // Check Tantivy index
        checks.push(self.check_index().await);
        
        // Check file system
        checks.push(self.check_filesystem().await);
        
        // Check memory
        checks.push(self.check_memory().await);
        
        // Check search functionality
        checks.push(self.check_search().await);
        
        // Determine overall status
        for check in &checks {
            match check.status {
                HealthStatus::Unhealthy => overall_status = HealthStatus::Unhealthy,
                HealthStatus::Degraded if overall_status != HealthStatus::Unhealthy => {
                    overall_status = HealthStatus::Degraded;
                }
                _ => {}
            }
        }
        
        Ok(HealthCheck {
            status: overall_status,
            checks,
            timestamp: chrono::Utc::now().to_rfc3339(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        })
    }
    
    async fn check_index(&self) -> ComponentCheck {
        let start = Instant::now();
        let mut check = ComponentCheck {
            name: "tantivy_index".to_string(),
            status: HealthStatus::Healthy,
            message: String::new(),
            response_time_ms: 0.0,
        };
        
        // Try to open the index
        match Index::open_in_dir(&self.index_path) {
            Ok(index) => {
                // Verify we can create a reader
                match index.reader() {
                    Ok(reader) => {
                        let searcher = reader.searcher();
                        let num_docs = searcher.num_docs();
                        check.message = format!("Index healthy with {} documents", num_docs);
                        
                        // Check if index is too small (might indicate issues)
                        if num_docs == 0 {
                            check.status = HealthStatus::Degraded;
                            check.message = "Index contains no documents".to_string();
                        }
                    }
                    Err(e) => {
                        check.status = HealthStatus::Unhealthy;
                        check.message = format!("Cannot read index: {}", e);
                    }
                }
            }
            Err(e) => {
                check.status = HealthStatus::Unhealthy;
                check.message = format!("Cannot open index: {}", e);
            }
        }
        
        check.response_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        check
    }
    
    async fn check_filesystem(&self) -> ComponentCheck {
        let start = Instant::now();
        let mut check = ComponentCheck {
            name: "filesystem".to_string(),
            status: HealthStatus::Healthy,
            message: String::new(),
            response_time_ms: 0.0,
        };
        
        // Check if we can write to temp directory
        let test_file = self.index_path.parent()
            .unwrap_or(Path::new("."))
            .join(".health_check_test");
        
        match std::fs::write(&test_file, b"health_check") {
            Ok(_) => {
                // Clean up test file
                let _ = std::fs::remove_file(&test_file);
                
                // Check available disk space
                #[cfg(windows)]
                {
                    use winapi::um::fileapi::GetDiskFreeSpaceExW;
                    use std::ptr;
                    use std::ffi::OsStr;
                    use std::os::windows::ffi::OsStrExt;
                    
                    unsafe {
                        let mut free_bytes: u64 = 0;
                        let path: Vec<u16> = OsStr::new(self.index_path.to_str().unwrap())
                            .encode_wide()
                            .chain(Some(0))
                            .collect();
                        
                        if GetDiskFreeSpaceExW(
                            path.as_ptr(),
                            &mut free_bytes as *mut u64,
                            ptr::null_mut(),
                            ptr::null_mut()
                        ) != 0 {
                            let free_mb = free_bytes / (1024 * 1024);
                            check.message = format!("Filesystem healthy, {}MB free", free_mb);
                            
                            if free_mb < 100 {
                                check.status = HealthStatus::Degraded;
                                check.message = format!("Low disk space: {}MB", free_mb);
                            }
                        }
                    }
                }
                
                #[cfg(not(windows))]
                {
                    check.message = "Filesystem healthy".to_string();
                }
            }
            Err(e) => {
                check.status = HealthStatus::Unhealthy;
                check.message = format!("Cannot write to filesystem: {}", e);
            }
        }
        
        check.response_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        check
    }
    
    async fn check_memory(&self) -> ComponentCheck {
        let start = Instant::now();
        let mut check = ComponentCheck {
            name: "memory".to_string(),
            status: HealthStatus::Healthy,
            message: String::new(),
            response_time_ms: 0.0,
        };
        
        // Get current memory usage
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let available_memory = total_memory - used_memory;
        
        let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;
        let available_mb = available_memory / 1024;
        
        check.message = format!(
            "Memory usage: {:.1}%, {}MB available",
            usage_percent, available_mb
        );
        
        if usage_percent > 90.0 {
            check.status = HealthStatus::Unhealthy;
            check.message = format!("Critical memory usage: {:.1}%", usage_percent);
        } else if usage_percent > 75.0 {
            check.status = HealthStatus::Degraded;
            check.message = format!("High memory usage: {:.1}%", usage_percent);
        }
        
        check.response_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        check
    }
    
    async fn check_search(&self) -> ComponentCheck {
        let start = Instant::now();
        let mut check = ComponentCheck {
            name: "search_engine".to_string(),
            status: HealthStatus::Healthy,
            message: String::new(),
            response_time_ms: 0.0,
        };
        
        // Try to perform a simple search
        match SearchEngine::new(&self.index_path) {
            Ok(engine) => {
                // Run a test query
                match engine.search("test") {
                    Ok(_) => {
                        check.message = "Search engine operational".to_string();
                    }
                    Err(e) => {
                        check.status = HealthStatus::Unhealthy;
                        check.message = format!("Search failed: {}", e);
                    }
                }
            }
            Err(e) => {
                check.status = HealthStatus::Unhealthy;
                check.message = format!("Cannot initialize search engine: {}", e);
            }
        }
        
        let response_time = start.elapsed().as_secs_f64() * 1000.0;
        check.response_time_ms = response_time;
        
        // Check if search is too slow
        if response_time > 100.0 {
            check.status = HealthStatus::Degraded;
            check.message = format!("Search slow: {:.1}ms", response_time);
        }
        
        check
    }
}

// HTTP endpoint handler
pub async fn health_endpoint(checker: &HealthChecker) -> impl warp::Reply {
    match checker.check_health().await {
        Ok(health) => {
            let status_code = match health.status {
                HealthStatus::Healthy => warp::http::StatusCode::OK,
                HealthStatus::Degraded => warp::http::StatusCode::OK, // Still return 200 for degraded
                HealthStatus::Unhealthy => warp::http::StatusCode::SERVICE_UNAVAILABLE,
            };
            
            warp::reply::with_status(
                warp::reply::json(&health),
                status_code
            )
        }
        Err(e) => {
            let error_response = serde_json::json!({
                "status": "Unhealthy",
                "error": e.to_string()
            });
            
            warp::reply::with_status(
                warp::reply::json(&error_response),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR
            )
        }
    }
}

// Readiness probe - lighter weight check
pub async fn readiness_endpoint(checker: &HealthChecker) -> impl warp::Reply {
    // Just check if index is accessible
    match Index::open_in_dir(&checker.index_path) {
        Ok(_) => warp::reply::with_status(
            "ready",
            warp::http::StatusCode::OK
        ),
        Err(_) => warp::reply::with_status(
            "not ready",
            warp::http::StatusCode::SERVICE_UNAVAILABLE
        )
    }
}
```

2. **Add test for health checks**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_health_check() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let index_path = temp_dir.path().join("test_index");
        
        // Create a test index
        let index = create_tantivy_index(&index_path)?;
        
        let checker = HealthChecker::new(index_path);
        let health = checker.check_health().await?;
        
        // Verify structure
        assert!(health.uptime_seconds >= 0);
        assert!(!health.checks.is_empty());
        
        // All checks should complete
        for check in &health.checks {
            assert!(check.response_time_ms >= 0.0);
            assert!(!check.message.is_empty());
        }
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Health endpoint returns 200 when healthy
- [ ] Health endpoint returns 503 when unhealthy
- [ ] All component checks complete within 1 second
- [ ] Readiness probe responds within 100ms
- [ ] Memory usage tracking is accurate
- [ ] Disk space warnings trigger appropriately

## Common Pitfalls to Avoid
- Don't make health checks too expensive
- Avoid caching health status for too long
- Ensure checks don't block each other
- Handle permission errors gracefully on Windows
- Don't expose sensitive information in health responses

## Context for Next Task
Task 100 will implement graceful shutdown and resource cleanup.