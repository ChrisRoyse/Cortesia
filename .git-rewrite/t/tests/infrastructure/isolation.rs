//! Environment Isolation System
//! 
//! Provides resource isolation and controlled execution environments for reliable testing.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::fs;

/// Test environment with resource isolation
#[derive(Debug)]
pub struct TestEnvironment {
    /// Unique environment ID
    pub id: String,
    /// Temporary directory for this environment
    pub temp_dir: TempDir,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// File system isolation
    pub fs_isolation: FileSystemIsolation,
    /// Process monitoring
    pub process_monitor: Arc<Mutex<ProcessMonitor>>,
    /// Cleanup configuration
    pub cleanup_config: CleanupConfig,
}

/// Resource limits for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Maximum CPU percentage (0.0 to 100.0)
    pub max_cpu_percent: f64,
    /// Maximum file handles
    pub max_file_handles: u32,
    /// Maximum network connections
    pub max_network_connections: u32,
    /// Maximum disk usage in bytes
    pub max_disk_usage_bytes: u64,
    /// Execution timeout
    pub execution_timeout: Duration,
    /// Maximum number of threads
    pub max_threads: u32,
    /// Maximum open files
    pub max_open_files: u32,
}

/// Network configuration for isolated environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Whether network access is allowed
    pub enabled: bool,
    /// Allowed hosts/domains
    pub allowed_hosts: Vec<String>,
    /// Blocked hosts/domains
    pub blocked_hosts: Vec<String>,
    /// Maximum bandwidth in bytes per second
    pub max_bandwidth_bps: u64,
    /// Network timeout
    pub timeout: Duration,
    /// Allowed ports
    pub allowed_ports: Vec<u16>,
    /// Proxy configuration
    pub proxy_config: Option<ProxyConfig>,
}

/// Proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// Proxy host
    pub host: String,
    /// Proxy port
    pub port: u16,
    /// Authentication if required
    pub auth: Option<ProxyAuth>,
}

/// Proxy authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyAuth {
    /// Username
    pub username: String,
    /// Password
    pub password: String,
}

/// File system isolation configuration
#[derive(Debug)]
pub struct FileSystemIsolation {
    /// Root directory for isolation
    pub root_dir: PathBuf,
    /// Read-only paths
    pub readonly_paths: Vec<PathBuf>,
    /// Writable paths
    pub writable_paths: Vec<PathBuf>,
    /// Blocked paths
    pub blocked_paths: Vec<PathBuf>,
    /// Maximum file size
    pub max_file_size: u64,
    /// Enable file access monitoring
    pub monitor_access: bool,
    /// File access log
    pub access_log: Arc<Mutex<Vec<FileAccess>>>,
}

/// File access record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAccess {
    /// Path accessed
    pub path: PathBuf,
    /// Type of access
    pub access_type: FileAccessType,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Success or failure
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Types of file access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileAccessType {
    Read,
    Write,
    Create,
    Delete,
    Execute,
    Metadata,
}

/// Process monitoring for resource usage
#[derive(Debug)]
pub struct ProcessMonitor {
    /// Process ID
    pub pid: Option<u32>,
    /// Resource usage samples
    pub usage_samples: Vec<ResourceUsageSample>,
    /// Monitoring interval
    pub monitor_interval: Duration,
    /// Whether monitoring is active
    pub active: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Resource usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSample {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU usage percentage
    pub cpu_percent: f64,
    /// Open file descriptors
    pub open_fds: u32,
    /// Thread count
    pub thread_count: u32,
    /// Disk I/O bytes read
    pub disk_read_bytes: u64,
    /// Disk I/O bytes written
    pub disk_write_bytes: u64,
    /// Network bytes sent
    pub network_sent_bytes: u64,
    /// Network bytes received
    pub network_received_bytes: u64,
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold (percentage of limit)
    pub memory_threshold_percent: f64,
    /// CPU usage threshold
    pub cpu_threshold_percent: f64,
    /// File descriptor threshold
    pub fd_threshold_percent: f64,
    /// Disk usage threshold
    pub disk_threshold_percent: f64,
}

/// Cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupConfig {
    /// When to cleanup
    pub policy: CleanupPolicy,
    /// Paths to preserve during cleanup
    pub preserve_paths: Vec<PathBuf>,
    /// Maximum age of files to keep
    pub max_file_age: Option<Duration>,
    /// Force cleanup even if test failed
    pub force_cleanup: bool,
}

/// Cleanup policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupPolicy {
    /// Always cleanup after test
    Always,
    /// Cleanup only on test success
    OnSuccess,
    /// Cleanup only on test failure
    OnFailure,
    /// Never cleanup (for debugging)
    Never,
    /// Cleanup based on configuration
    Conditional,
}

/// Temporary directory manager
#[derive(Debug)]
pub struct TempDir {
    /// Directory path
    pub path: PathBuf,
    /// Whether to cleanup on drop
    pub cleanup_on_drop: bool,
    /// File tracking
    pub files_created: Arc<Mutex<Vec<PathBuf>>>,
}

impl TestEnvironment {
    /// Create a new isolated test environment
    pub fn new(resource_requirements: &crate::infrastructure::ResourceRequirements) -> Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let temp_dir = TempDir::new(&format!("llmkg_test_{}", &id[..8]))?;
        
        let resource_limits = ResourceLimits {
            max_memory_bytes: resource_requirements.min_memory_mb * 1024 * 1024 * 2, // 2x requirement
            max_cpu_percent: 80.0,
            max_file_handles: 1024,
            max_network_connections: 100,
            max_disk_usage_bytes: 1024 * 1024 * 1024, // 1GB
            execution_timeout: Duration::from_secs(300),
            max_threads: 64,
            max_open_files: 256,
        };

        let network_config = NetworkConfig {
            enabled: resource_requirements.network_required,
            allowed_hosts: vec!["localhost".to_string(), "127.0.0.1".to_string()],
            blocked_hosts: Vec::new(),
            max_bandwidth_bps: 10 * 1024 * 1024, // 10 MB/s
            timeout: Duration::from_secs(30),
            allowed_ports: vec![80, 443, 8080],
            proxy_config: None,
        };

        let fs_isolation = FileSystemIsolation {
            root_dir: temp_dir.path.clone(),
            readonly_paths: Vec::new(),
            writable_paths: vec![temp_dir.path.clone()],
            blocked_paths: Vec::new(),
            max_file_size: 100 * 1024 * 1024, // 100MB
            monitor_access: true,
            access_log: Arc::new(Mutex::new(Vec::new())),
        };

        let process_monitor = Arc::new(Mutex::new(ProcessMonitor {
            pid: None,
            usage_samples: Vec::new(),
            monitor_interval: Duration::from_millis(100),
            active: false,
            alert_thresholds: AlertThresholds {
                memory_threshold_percent: 90.0,
                cpu_threshold_percent: 95.0,
                fd_threshold_percent: 90.0,
                disk_threshold_percent: 90.0,
            },
        }));

        let cleanup_config = CleanupConfig {
            policy: CleanupPolicy::OnSuccess,
            preserve_paths: Vec::new(),
            max_file_age: Some(Duration::from_secs(3600)), // 1 hour
            force_cleanup: false,
        };

        Ok(Self {
            id,
            temp_dir,
            env_vars: HashMap::new(),
            resource_limits,
            network_config,
            fs_isolation,
            process_monitor,
            cleanup_config,
        })
    }

    /// Set an environment variable for this test environment
    pub fn set_env_var(&mut self, key: String, value: String) {
        self.env_vars.insert(key, value);
    }

    /// Get an environment variable
    pub fn get_env_var(&self, key: &str) -> Option<&String> {
        self.env_vars.get(key)
    }

    /// Create a file in the isolated environment
    pub fn create_file(&self, relative_path: &str, content: &[u8]) -> Result<PathBuf> {
        let file_path = self.temp_dir.path.join(relative_path);
        
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create directory {}: {}", parent.display(), e))?;
        }

        // Check file size limits
        if content.len() as u64 > self.fs_isolation.max_file_size {
            return Err(anyhow!("File size {} exceeds limit {}", 
                              content.len(), self.fs_isolation.max_file_size));
        }

        // Write file
        fs::write(&file_path, content)
            .map_err(|e| anyhow!("Failed to write file {}: {}", file_path.display(), e))?;

        // Log access
        self.log_file_access(&file_path, FileAccessType::Create, true, None);

        // Track created file
        if let Ok(mut files) = self.temp_dir.files_created.lock() {
            files.push(file_path.clone());
        }

        Ok(file_path)
    }

    /// Read a file from the isolated environment
    pub fn read_file(&self, relative_path: &str) -> Result<Vec<u8>> {
        let file_path = self.temp_dir.path.join(relative_path);
        
        // Check if path is allowed
        if !self.is_path_readable(&file_path) {
            self.log_file_access(&file_path, FileAccessType::Read, false, 
                                Some("Path not readable".to_string()));
            return Err(anyhow!("Access denied to path: {}", file_path.display()));
        }

        match fs::read(&file_path) {
            Ok(content) => {
                self.log_file_access(&file_path, FileAccessType::Read, true, None);
                Ok(content)
            }
            Err(e) => {
                self.log_file_access(&file_path, FileAccessType::Read, false, 
                                   Some(e.to_string()));
                Err(anyhow!("Failed to read file {}: {}", file_path.display(), e))
            }
        }
    }

    /// Check if a path is readable in this environment
    pub fn is_path_readable(&self, path: &Path) -> bool {
        // Check if path is in writable areas or explicitly allowed
        for writable in &self.fs_isolation.writable_paths {
            if path.starts_with(writable) {
                return true;
            }
        }

        for readonly in &self.fs_isolation.readonly_paths {
            if path.starts_with(readonly) {
                return true;
            }
        }

        // Check if path is blocked
        for blocked in &self.fs_isolation.blocked_paths {
            if path.starts_with(blocked) {
                return false;
            }
        }

        false
    }

    /// Start monitoring process resources
    pub fn start_monitoring(&self, pid: u32) -> Result<()> {
        let mut monitor = self.process_monitor.lock()
            .map_err(|_| anyhow!("Failed to acquire monitor lock"))?;
        
        monitor.pid = Some(pid);
        monitor.active = true;
        monitor.usage_samples.clear();
        
        Ok(())
    }

    /// Stop monitoring and return collected samples
    pub fn stop_monitoring(&self) -> Result<Vec<ResourceUsageSample>> {
        let mut monitor = self.process_monitor.lock()
            .map_err(|_| anyhow!("Failed to acquire monitor lock"))?;
        
        monitor.active = false;
        monitor.pid = None;
        
        Ok(monitor.usage_samples.clone())
    }

    /// Sample current resource usage
    pub fn sample_resource_usage(&self) -> Result<ResourceUsageSample> {
        let monitor = self.process_monitor.lock()
            .map_err(|_| anyhow!("Failed to acquire monitor lock"))?;
        
        let pid = monitor.pid.ok_or_else(|| anyhow!("No process being monitored"))?;
        
        // In a real implementation, this would use system APIs to get actual usage
        // For now, return mock data
        let sample = ResourceUsageSample {
            timestamp: std::time::SystemTime::now(),
            memory_bytes: 50 * 1024 * 1024, // 50MB mock
            cpu_percent: 25.0,
            open_fds: 10,
            thread_count: 4,
            disk_read_bytes: 1024 * 1024,
            disk_write_bytes: 512 * 1024,
            network_sent_bytes: 256 * 1024,
            network_received_bytes: 128 * 1024,
        };

        Ok(sample)
    }

    /// Check if resource usage exceeds limits
    pub fn check_resource_limits(&self, sample: &ResourceUsageSample) -> Vec<String> {
        let mut violations = Vec::new();

        if sample.memory_bytes > self.resource_limits.max_memory_bytes {
            violations.push(format!("Memory usage {} exceeds limit {}", 
                                   sample.memory_bytes, self.resource_limits.max_memory_bytes));
        }

        if sample.cpu_percent > self.resource_limits.max_cpu_percent {
            violations.push(format!("CPU usage {}% exceeds limit {}%", 
                                   sample.cpu_percent, self.resource_limits.max_cpu_percent));
        }

        if sample.open_fds > self.resource_limits.max_file_handles {
            violations.push(format!("Open file descriptors {} exceeds limit {}", 
                                   sample.open_fds, self.resource_limits.max_file_handles));
        }

        if sample.thread_count > self.resource_limits.max_threads {
            violations.push(format!("Thread count {} exceeds limit {}", 
                                   sample.thread_count, self.resource_limits.max_threads));
        }

        violations
    }

    /// Get file access log
    pub fn get_file_access_log(&self) -> Result<Vec<FileAccess>> {
        let log = self.fs_isolation.access_log.lock()
            .map_err(|_| anyhow!("Failed to acquire access log lock"))?;
        Ok(log.clone())
    }

    /// Log file access
    fn log_file_access(&self, path: &Path, access_type: FileAccessType, success: bool, error: Option<String>) {
        if !self.fs_isolation.monitor_access {
            return;
        }

        let access = FileAccess {
            path: path.to_path_buf(),
            access_type,
            timestamp: std::time::SystemTime::now(),
            success,
            error,
        };

        if let Ok(mut log) = self.fs_isolation.access_log.lock() {
            log.push(access);
        }
    }

    /// Clean up the test environment
    pub fn cleanup(&self) -> Result<()> {
        match self.cleanup_config.policy {
            CleanupPolicy::Never => return Ok(()),
            CleanupPolicy::Always => self.perform_cleanup()?,
            CleanupPolicy::OnSuccess => {
                // Would check test result here
                self.perform_cleanup()?;
            }
            CleanupPolicy::OnFailure => {
                // Would check test result here
                return Ok(());
            }
            CleanupPolicy::Conditional => {
                if self.cleanup_config.force_cleanup {
                    self.perform_cleanup()?;
                }
            }
        }

        Ok(())
    }

    /// Perform actual cleanup
    fn perform_cleanup(&self) -> Result<()> {
        // Remove temporary files, but preserve specified paths
        if let Ok(files) = self.temp_dir.files_created.lock() {
            for file_path in files.iter() {
                let should_preserve = self.cleanup_config.preserve_paths.iter()
                    .any(|preserve| file_path.starts_with(preserve));

                if !should_preserve {
                    if let Err(e) = fs::remove_file(file_path) {
                        log::warn!("Failed to remove file {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get environment statistics
    pub fn get_statistics(&self) -> EnvironmentStatistics {
        let monitor = self.process_monitor.lock().unwrap();
        let access_log = self.fs_isolation.access_log.lock().unwrap();

        EnvironmentStatistics {
            env_id: self.id.clone(),
            temp_dir_path: self.temp_dir.path.clone(),
            files_created: self.temp_dir.files_created.lock().unwrap().len(),
            total_resource_samples: monitor.usage_samples.len(),
            file_accesses: access_log.len(),
            env_vars_count: self.env_vars.len(),
            cleanup_policy: self.cleanup_config.policy.clone(),
        }
    }
}

impl TempDir {
    /// Create a new temporary directory
    pub fn new(prefix: &str) -> Result<Self> {
        let temp_base = std::env::temp_dir();
        let dir_name = format!("{}_{}", prefix, uuid::Uuid::new_v4().to_string()[..8].to_string());
        let temp_path = temp_base.join(dir_name);

        fs::create_dir_all(&temp_path)
            .map_err(|e| anyhow!("Failed to create temp directory {}: {}", temp_path.display(), e))?;

        Ok(Self {
            path: temp_path,
            cleanup_on_drop: true,
            files_created: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Get the temporary directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Set cleanup on drop behavior
    pub fn set_cleanup_on_drop(&mut self, cleanup: bool) {
        self.cleanup_on_drop = cleanup;
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if self.cleanup_on_drop && self.path.exists() {
            if let Err(e) = fs::remove_dir_all(&self.path) {
                log::warn!("Failed to cleanup temp directory {}: {}", self.path.display(), e);
            }
        }
    }
}

/// Environment statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentStatistics {
    pub env_id: String,
    pub temp_dir_path: PathBuf,
    pub files_created: usize,
    pub total_resource_samples: usize,
    pub file_accesses: usize,
    pub env_vars_count: usize,
    pub cleanup_policy: CleanupPolicy,
}

/// Utility functions for environment management
pub mod utils {
    use super::*;

    /// Create a minimal test environment
    pub fn minimal_environment() -> Result<TestEnvironment> {
        let requirements = crate::infrastructure::ResourceRequirements {
            min_cpu_cores: 1,
            min_memory_mb: 256,
            gpu_memory_mb: None,
            network_required: false,
        };
        TestEnvironment::new(&requirements)
    }

    /// Create a network-enabled test environment
    pub fn network_environment() -> Result<TestEnvironment> {
        let requirements = crate::infrastructure::ResourceRequirements {
            min_cpu_cores: 1,
            min_memory_mb: 512,
            gpu_memory_mb: None,
            network_required: true,
        };
        TestEnvironment::new(&requirements)
    }

    /// Create a high-performance test environment
    pub fn performance_environment() -> Result<TestEnvironment> {
        let requirements = crate::infrastructure::ResourceRequirements {
            min_cpu_cores: 4,
            min_memory_mb: 2048,
            gpu_memory_mb: Some(1024),
            network_required: true,
        };
        TestEnvironment::new(&requirements)
    }

    /// Run a closure in an isolated environment
    pub fn with_isolated_env<T, F>(f: F) -> Result<T>
    where
        F: FnOnce(&TestEnvironment) -> Result<T>,
    {
        let env = minimal_environment()?;
        let result = f(&env)?;
        env.cleanup()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_creation() {
        let requirements = crate::infrastructure::ResourceRequirements {
            min_cpu_cores: 1,
            min_memory_mb: 256,
            gpu_memory_mb: None,
            network_required: false,
        };

        let env = TestEnvironment::new(&requirements).unwrap();
        assert!(env.temp_dir.path.exists());
        assert!(!env.id.is_empty());
    }

    #[test]
    fn test_file_operations() {
        let env = utils::minimal_environment().unwrap();
        
        // Create a file
        let content = b"test content";
        let file_path = env.create_file("test.txt", content).unwrap();
        assert!(file_path.exists());

        // Read the file back
        let read_content = env.read_file("test.txt").unwrap();
        assert_eq!(content, read_content.as_slice());

        // Check access log
        let log = env.get_file_access_log().unwrap();
        assert_eq!(log.len(), 2); // Create + Read
    }

    #[test]
    fn test_environment_variables() {
        let mut env = utils::minimal_environment().unwrap();
        
        env.set_env_var("TEST_VAR".to_string(), "test_value".to_string());
        assert_eq!(env.get_env_var("TEST_VAR"), Some(&"test_value".to_string()));
        assert_eq!(env.get_env_var("NONEXISTENT"), None);
    }

    #[test]
    fn test_resource_monitoring() {
        let env = utils::minimal_environment().unwrap();
        
        // Start monitoring (mock PID)
        env.start_monitoring(12345).unwrap();
        
        // Sample resource usage
        let sample = env.sample_resource_usage().unwrap();
        assert!(sample.memory_bytes > 0);
        
        // Check limits
        let violations = env.check_resource_limits(&sample);
        assert!(violations.is_empty()); // Mock data should be within limits
        
        // Stop monitoring
        let samples = env.stop_monitoring().unwrap();
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_path_access_control() {
        let env = utils::minimal_environment().unwrap();
        
        // Temp directory should be readable
        assert!(env.is_path_readable(&env.temp_dir.path));
        
        // Random system path should not be readable
        let system_path = PathBuf::from("/etc/passwd");
        assert!(!env.is_path_readable(&system_path));
    }

    #[test]
    fn test_temp_dir() {
        let temp_dir = TempDir::new("test").unwrap();
        let path = temp_dir.path().to_path_buf();
        
        assert!(path.exists());
        
        // Drop should cleanup
        drop(temp_dir);
        assert!(!path.exists());
    }

    #[test]
    fn test_temp_dir_no_cleanup() {
        let mut temp_dir = TempDir::new("test_no_cleanup").unwrap();
        let path = temp_dir.path().to_path_buf();
        
        temp_dir.set_cleanup_on_drop(false);
        drop(temp_dir);
        
        // Should still exist
        assert!(path.exists());
        
        // Manual cleanup
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn test_with_isolated_env() {
        let result = utils::with_isolated_env(|env| {
            env.create_file("test.txt", b"content")?;
            Ok(42)
        }).unwrap();
        
        assert_eq!(result, 42);
    }

    #[test]
    fn test_environment_statistics() {
        let env = utils::minimal_environment().unwrap();
        env.create_file("test1.txt", b"content1").unwrap();
        env.create_file("test2.txt", b"content2").unwrap();
        env.set_env_var("VAR1".to_string(), "value1".to_string());
        
        let stats = env.get_statistics();
        assert_eq!(stats.files_created, 2);
        assert_eq!(stats.env_vars_count, 1);
        assert_eq!(stats.file_accesses, 2); // Two creates
    }
}