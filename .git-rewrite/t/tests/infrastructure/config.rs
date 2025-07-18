//! Test Configuration Management
//! 
//! Provides comprehensive configuration management for the simulation infrastructure.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Main test configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Deterministic seed for reproducible random generation
    pub deterministic_seed: u64,
    /// Performance targets for validation
    pub performance_targets: PerformanceTargets,
    /// Data generation parameters
    pub data_generation_params: DataGenParams,
    /// Environment settings
    pub environment_settings: EnvironmentSettings,
    /// Validation thresholds
    pub validation_thresholds: ValidationThresholds,
    /// Execution settings
    pub execution_settings: ExecutionSettings,
    /// Reporting configuration
    pub reporting_config: ReportingConfig,
    /// Dashboard configuration
    pub dashboard_config: DashboardConfig,
    /// Dashboard enabled flag
    pub dashboard_enabled: bool,
    /// Performance database path
    pub performance_db_path: PathBuf,
    /// Data directory for test data storage
    pub data_directory: Option<PathBuf>,
    /// Maximum parallel tests
    pub max_parallel_tests: usize,
}

/// Performance targets for the LLMKG system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Maximum query latency in milliseconds (target: <1.0ms)
    pub query_latency_ms: f64,
    /// Maximum memory per entity in bytes (target: <70 bytes)
    pub memory_per_entity_bytes: u64,
    /// Maximum similarity search time in milliseconds (target: <5.0ms)
    pub similarity_search_ms: f64,
    /// Minimum compression ratio (target: 50-1000x)
    pub min_compression_ratio: f64,
    /// Maximum compression ratio
    pub max_compression_ratio: f64,
    /// Maximum indexing time per entity in microseconds
    pub indexing_time_us_per_entity: f64,
    /// Minimum throughput in operations per second
    pub min_throughput_ops_per_sec: f64,
    /// Maximum CPU usage percentage during operations
    pub max_cpu_usage_percent: f64,
    /// Maximum memory overhead percentage
    pub max_memory_overhead_percent: f64,
}

/// Data generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGenParams {
    /// Cache directory for generated data
    pub cache_dir: PathBuf,
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
    /// Cache eviction policy
    pub cache_eviction_policy: CacheEvictionPolicy,
    /// Data validation level
    pub validation_level: DataValidationLevel,
    /// Parallel generation threads
    pub generation_threads: usize,
    /// Compression for cached data
    pub compress_cached_data: bool,
    /// Custom data generators
    pub custom_generators: HashMap<String, String>,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    LRU,     // Least Recently Used
    LFU,     // Least Frequently Used
    FIFO,    // First In, First Out
    Random,  // Random eviction
    Size,    // Evict largest items first
}

/// Data validation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataValidationLevel {
    None,        // No validation
    Basic,       // Basic checksum validation
    Structural,  // Validate data structure
    Complete,    // Full semantic validation
}

/// Environment settings for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    /// Temporary directory for test artifacts
    pub temp_dir: PathBuf,
    /// Environment variables to set
    pub env_vars: HashMap<String, String>,
    /// Resource limits for test execution
    pub resource_limits: ResourceLimits,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Cleanup policy
    pub cleanup_policy: CleanupPolicy,
    /// Isolation level
    pub isolation_level: IsolationLevel,
}

/// Resource limits for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU percentage
    pub max_cpu_percent: f64,
    /// Maximum file handles
    pub max_file_handles: u32,
    /// Maximum network connections
    pub max_network_connections: u32,
    /// Maximum disk usage in MB
    pub max_disk_usage_mb: u64,
    /// Execution timeout
    pub execution_timeout: Duration,
}

/// Network configuration for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable network access
    pub enabled: bool,
    /// Allowed hosts for network access
    pub allowed_hosts: Vec<String>,
    /// Blocked hosts
    pub blocked_hosts: Vec<String>,
    /// Network timeout
    pub timeout: Duration,
    /// Maximum bandwidth in MB/s
    pub max_bandwidth_mb_per_s: f64,
}

/// Cleanup policies after test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupPolicy {
    Always,      // Always cleanup
    OnSuccess,   // Cleanup only on test success
    OnFailure,   // Cleanup only on test failure
    Never,       // Never cleanup (for debugging)
}

/// Test isolation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    None,        // No isolation
    Process,     // Process-level isolation
    Container,   // Container-level isolation
    VM,          // Virtual machine isolation
}

/// Validation thresholds for test outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationThresholds {
    /// Floating point comparison tolerance
    pub float_tolerance: f64,
    /// Checksum validation enabled
    pub checksum_validation: bool,
    /// Performance deviation tolerance (percentage)
    pub performance_tolerance_percent: f64,
    /// Memory usage tolerance (percentage)
    pub memory_tolerance_percent: f64,
    /// Timing tolerance in milliseconds
    pub timing_tolerance_ms: f64,
    /// Require deterministic results
    pub require_deterministic: bool,
}

/// Test execution settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSettings {
    /// Maximum parallel test execution
    pub max_parallel_tests: usize,
    /// Test retry policy
    pub retry_policy: RetryPolicy,
    /// Timeout for individual tests
    pub default_test_timeout: Duration,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Profiling configuration
    pub profiling_config: ProfilingConfig,
    /// Debug mode settings
    pub debug_mode: DebugMode,
}

/// Test retry policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay between attempts
    pub retry_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// Retry on specific errors only
    pub retry_on_errors: Vec<String>,
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable CPU profiling
    pub cpu_profiling: bool,
    /// Enable memory profiling
    pub memory_profiling: bool,
    /// Enable I/O profiling
    pub io_profiling: bool,
    /// Profiling sample rate
    pub sample_rate_hz: u32,
    /// Profile output directory
    pub output_dir: PathBuf,
}

/// Debug mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugMode {
    /// Enable debug mode
    pub enabled: bool,
    /// Verbose output
    pub verbose: bool,
    /// Debug output directory
    pub debug_dir: PathBuf,
    /// Save intermediate results
    pub save_intermediates: bool,
    /// Enable trace logging
    pub trace_logging: bool,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Output directory for reports
    pub output_dir: PathBuf,
    /// Report formats to generate
    pub formats: Vec<ReportFormat>,
    /// Include performance graphs
    pub include_graphs: bool,
    /// Include detailed logs
    pub include_detailed_logs: bool,
    /// Report generation timeout
    pub generation_timeout: Duration,
    /// Custom report templates
    pub custom_templates: HashMap<String, PathBuf>,
}

/// Supported report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    HTML,
    JSON,
    XML,
    PDF,
    CSV,
    Markdown,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard port
    pub port: u16,
    /// Dashboard host
    pub host: String,
    /// WebSocket configuration
    pub websocket_config: WebSocketConfig,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Authentication settings
    pub auth_config: Option<AuthConfig>,
}

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// Maximum connections
    pub max_connections: usize,
    /// Message buffer size
    pub message_buffer_size: usize,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication
    pub enabled: bool,
    /// Authentication method
    pub method: AuthMethod,
    /// Session timeout
    pub session_timeout: Duration,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    None,
    Token,
    Basic,
    OAuth,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            deterministic_seed: 42,
            performance_targets: PerformanceTargets::default(),
            data_generation_params: DataGenParams::default(),
            environment_settings: EnvironmentSettings::default(),
            validation_thresholds: ValidationThresholds::default(),
            execution_settings: ExecutionSettings::default(),
            reporting_config: ReportingConfig::default(),
            dashboard_config: DashboardConfig::default(),
            dashboard_enabled: false,
            performance_db_path: PathBuf::from("./test_performance.db"),
            data_directory: None,
            max_parallel_tests: num_cpus::get(),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            query_latency_ms: 1.0,
            memory_per_entity_bytes: 70,
            similarity_search_ms: 5.0,
            min_compression_ratio: 50.0,
            max_compression_ratio: 1000.0,
            indexing_time_us_per_entity: 10.0,
            min_throughput_ops_per_sec: 1000.0,
            max_cpu_usage_percent: 80.0,
            max_memory_overhead_percent: 20.0,
        }
    }
}

impl Default for DataGenParams {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("./test_data_cache"),
            max_cache_size_mb: 1024, // 1GB
            cache_eviction_policy: CacheEvictionPolicy::LRU,
            validation_level: DataValidationLevel::Basic,
            generation_threads: num_cpus::get(),
            compress_cached_data: true,
            custom_generators: HashMap::new(),
        }
    }
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            temp_dir: std::env::temp_dir().join("llmkg_tests"),
            env_vars: HashMap::new(),
            resource_limits: ResourceLimits::default(),
            network_config: NetworkConfig::default(),
            cleanup_policy: CleanupPolicy::OnSuccess,
            isolation_level: IsolationLevel::Process,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048, // 2GB
            max_cpu_percent: 80.0,
            max_file_handles: 1024,
            max_network_connections: 100,
            max_disk_usage_mb: 1024, // 1GB
            execution_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_hosts: vec!["localhost".to_string(), "127.0.0.1".to_string()],
            blocked_hosts: Vec::new(),
            timeout: Duration::from_secs(30),
            max_bandwidth_mb_per_s: 100.0,
        }
    }
}

impl Default for ValidationThresholds {
    fn default() -> Self {
        Self {
            float_tolerance: 1e-10,
            checksum_validation: true,
            performance_tolerance_percent: 5.0,
            memory_tolerance_percent: 10.0,
            timing_tolerance_ms: 1.0,
            require_deterministic: true,
        }
    }
}

impl Default for ExecutionSettings {
    fn default() -> Self {
        Self {
            max_parallel_tests: num_cpus::get(),
            retry_policy: RetryPolicy::default(),
            default_test_timeout: Duration::from_secs(300),
            enable_monitoring: true,
            profiling_config: ProfilingConfig::default(),
            debug_mode: DebugMode::default(),
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_retry_delay: Duration::from_secs(60),
            retry_on_errors: vec![
                "timeout".to_string(),
                "resource_unavailable".to_string(),
                "network_error".to_string(),
            ],
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            cpu_profiling: false,
            memory_profiling: false,
            io_profiling: false,
            sample_rate_hz: 100,
            output_dir: PathBuf::from("./profiles"),
        }
    }
}

impl Default for DebugMode {
    fn default() -> Self {
        Self {
            enabled: false,
            verbose: false,
            debug_dir: PathBuf::from("./debug"),
            save_intermediates: false,
            trace_logging: false,
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./test_reports"),
            formats: vec![ReportFormat::HTML, ReportFormat::JSON],
            include_graphs: true,
            include_detailed_logs: false,
            generation_timeout: Duration::from_secs(60),
            custom_templates: HashMap::new(),
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "localhost".to_string(),
            websocket_config: WebSocketConfig::default(),
            update_interval_ms: 1000,
            auth_config: None,
        }
    }
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            message_buffer_size: 1024,
            heartbeat_interval: Duration::from_secs(30),
        }
    }
}

impl TestConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {}: {}", path.display(), e))?;
        
        let config: Self = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse config file {}: {}", path.display(), e))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn to_file(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize config: {}", e))?;
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create config directory {}: {}", parent.display(), e))?;
        }
        
        std::fs::write(path, content)
            .map_err(|e| anyhow!("Failed to write config file {}: {}", path.display(), e))?;
        
        Ok(())
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Override with environment variables
        if let Ok(seed) = std::env::var("LLMKG_TEST_SEED") {
            config.deterministic_seed = seed.parse()
                .map_err(|e| anyhow!("Invalid LLMKG_TEST_SEED: {}", e))?;
        }
        
        if let Ok(parallel) = std::env::var("LLMKG_MAX_PARALLEL_TESTS") {
            config.max_parallel_tests = parallel.parse()
                .map_err(|e| anyhow!("Invalid LLMKG_MAX_PARALLEL_TESTS: {}", e))?;
        }
        
        if let Ok(cache_dir) = std::env::var("LLMKG_CACHE_DIR") {
            config.data_generation_params.cache_dir = PathBuf::from(cache_dir);
        }
        
        if let Ok(temp_dir) = std::env::var("LLMKG_TEMP_DIR") {
            config.environment_settings.temp_dir = PathBuf::from(temp_dir);
        }
        
        if let Ok(report_dir) = std::env::var("LLMKG_REPORT_DIR") {
            config.reporting_config.output_dir = PathBuf::from(report_dir);
        }
        
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate performance targets
        if self.performance_targets.query_latency_ms <= 0.0 {
            return Err(anyhow!("Query latency target must be positive"));
        }
        
        if self.performance_targets.memory_per_entity_bytes == 0 {
            return Err(anyhow!("Memory per entity target must be positive"));
        }
        
        if self.performance_targets.min_compression_ratio >= self.performance_targets.max_compression_ratio {
            return Err(anyhow!("Min compression ratio must be less than max compression ratio"));
        }
        
        // Validate execution settings
        if self.execution_settings.max_parallel_tests == 0 {
            return Err(anyhow!("Max parallel tests must be positive"));
        }
        
        if self.execution_settings.default_test_timeout.as_secs() == 0 {
            return Err(anyhow!("Default test timeout must be positive"));
        }
        
        // Validate resource limits
        let limits = &self.environment_settings.resource_limits;
        if limits.max_memory_mb == 0 {
            return Err(anyhow!("Max memory limit must be positive"));
        }
        
        if limits.max_cpu_percent <= 0.0 || limits.max_cpu_percent > 100.0 {
            return Err(anyhow!("CPU percentage must be between 0 and 100"));
        }
        
        // Validate validation thresholds
        if self.validation_thresholds.float_tolerance < 0.0 {
            return Err(anyhow!("Float tolerance must be non-negative"));
        }
        
        if self.validation_thresholds.performance_tolerance_percent < 0.0 {
            return Err(anyhow!("Performance tolerance must be non-negative"));
        }
        
        // Validate dashboard config
        if self.dashboard_config.port == 0 {
            return Err(anyhow!("Dashboard port must be positive"));
        }
        
        Ok(())
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge_with(mut self, other: TestConfig) -> Self {
        // Simple merge - in a real implementation you'd merge each field carefully
        self.deterministic_seed = other.deterministic_seed;
        self.max_parallel_tests = other.max_parallel_tests;
        self.dashboard_enabled = other.dashboard_enabled;
        // ... merge other fields as needed
        self
    }

    /// Create a configuration for quick testing
    pub fn quick_test_config() -> Self {
        let mut config = Self::default();
        config.execution_settings.max_parallel_tests = 1;
        config.execution_settings.default_test_timeout = Duration::from_secs(30);
        config.data_generation_params.max_cache_size_mb = 100;
        config.environment_settings.resource_limits.max_memory_mb = 512;
        config.validation_thresholds.performance_tolerance_percent = 20.0;
        config
    }

    /// Create a configuration for comprehensive testing
    pub fn comprehensive_test_config() -> Self {
        let mut config = Self::default();
        config.execution_settings.enable_monitoring = true;
        config.execution_settings.profiling_config.cpu_profiling = true;
        config.execution_settings.profiling_config.memory_profiling = true;
        config.reporting_config.include_detailed_logs = true;
        config.reporting_config.include_graphs = true;
        config.dashboard_enabled = true;
        config
    }
}

// Add num_cpus as a mock function since we don't have the dependency yet
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = TestConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quick_test_config() {
        let config = TestConfig::quick_test_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.execution_settings.max_parallel_tests, 1);
    }

    #[test]
    fn test_comprehensive_test_config() {
        let config = TestConfig::comprehensive_test_config();
        assert!(config.validate().is_ok());
        assert!(config.dashboard_enabled);
        assert!(config.execution_settings.profiling_config.cpu_profiling);
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut config = TestConfig::default();
        config.performance_targets.query_latency_ms = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_merge() {
        let config1 = TestConfig::default();
        let mut config2 = TestConfig::default();
        config2.deterministic_seed = 999;
        
        let merged = config1.merge_with(config2);
        assert_eq!(merged.deterministic_seed, 999);
    }
}