# Task 18: Implement Cache Configuration

## Context
You are implementing Phase 4 of a vector indexing system. Cache eviction policies were implemented in the previous task. Now you need to implement comprehensive cache configuration with validation, serialization, and runtime reconfiguration capabilities.

## Current State
- `src/cache.rs` exists with `MemoryEfficientCache` struct
- Multiple eviction policies (LRU, LFU, size-based, hybrid) are implemented
- Basic eviction configuration exists but needs enhancement
- Cache statistics and memory management are working

## Task Objective
Implement comprehensive cache configuration with validation, TOML serialization, runtime reconfiguration, and performance tuning presets.

## Implementation Requirements

### 1. Add comprehensive cache configuration struct
Add this enhanced configuration struct before the existing `EvictionConfig`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfiguration {
    pub max_entries: usize,
    pub max_memory_mb: usize,
    pub eviction: EvictionSettings,
    pub performance: PerformanceSettings,
    pub maintenance: MaintenanceSettings,
    pub monitoring: MonitoringSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionSettings {
    pub policy: EvictionPolicyConfig,
    pub memory_pressure_threshold: f64,
    pub aggressive_eviction_threshold: f64,
    pub min_entries_to_keep: usize,
    pub max_eviction_batch_size: usize,
    pub size_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicyConfig {
    LRU,
    LFU,
    SizeBased,
    Hybrid { size_weight: f64 },
    Adaptive { initial_policy: Box<EvictionPolicyConfig> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    pub enable_prefetch: bool,
    pub prefetch_batch_size: usize,
    pub enable_compression: bool,
    pub compression_threshold_bytes: usize,
    pub enable_background_cleanup: bool,
    pub cleanup_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceSettings {
    pub auto_defragment: bool,
    pub defragment_threshold: f64,
    pub max_fragmentation_ratio: f64,
    pub validate_consistency: bool,
    pub consistency_check_interval_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSettings {
    pub enable_detailed_stats: bool,
    pub track_query_patterns: bool,
    pub max_tracked_queries: usize,
    pub enable_performance_logging: bool,
    pub log_slow_operations_ms: u64,
}
```

### 2. Implement configuration validation
Add these validation methods:
```rust
impl CacheConfiguration {
    pub fn validate(&self) -> Result<(), ConfigurationError> {
        // Basic limits validation
        if self.max_entries == 0 {
            return Err(ConfigurationError::InvalidValue("max_entries must be greater than 0".to_string()));
        }
        
        if self.max_memory_mb == 0 {
            return Err(ConfigurationError::InvalidValue("max_memory_mb must be greater than 0".to_string()));
        }
        
        // Eviction settings validation
        self.eviction.validate()?;
        
        // Performance settings validation
        self.performance.validate()?;
        
        // Maintenance settings validation
        self.maintenance.validate()?;
        
        // Monitoring settings validation
        self.monitoring.validate()?;
        
        Ok(())
    }
    
    pub fn apply_memory_constraints(&mut self, available_memory_mb: usize) {
        // Don't exceed 50% of available memory
        let max_recommended = available_memory_mb / 2;
        if self.max_memory_mb > max_recommended {
            self.max_memory_mb = max_recommended;
        }
        
        // Adjust entry count based on memory
        let estimated_avg_entry_size_kb = 8; // Conservative estimate
        let max_entries_for_memory = (self.max_memory_mb * 1024) / estimated_avg_entry_size_kb;
        if self.max_entries > max_entries_for_memory {
            self.max_entries = max_entries_for_memory;
        }
    }
}

impl EvictionSettings {
    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if !(0.0..=1.0).contains(&self.memory_pressure_threshold) {
            return Err(ConfigurationError::InvalidValue("memory_pressure_threshold must be between 0.0 and 1.0".to_string()));
        }
        
        if !(0.0..=1.0).contains(&self.aggressive_eviction_threshold) {
            return Err(ConfigurationError::InvalidValue("aggressive_eviction_threshold must be between 0.0 and 1.0".to_string()));
        }
        
        if self.aggressive_eviction_threshold <= self.memory_pressure_threshold {
            return Err(ConfigurationError::InvalidValue("aggressive_eviction_threshold must be greater than memory_pressure_threshold".to_string()));
        }
        
        if !(0.0..=1.0).contains(&self.size_weight) {
            return Err(ConfigurationError::InvalidValue("size_weight must be between 0.0 and 1.0".to_string()));
        }
        
        if self.max_eviction_batch_size == 0 {
            return Err(ConfigurationError::InvalidValue("max_eviction_batch_size must be greater than 0".to_string()));
        }
        
        Ok(())
    }
}

impl PerformanceSettings {
    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if self.prefetch_batch_size == 0 {
            return Err(ConfigurationError::InvalidValue("prefetch_batch_size must be greater than 0".to_string()));
        }
        
        if self.cleanup_interval_seconds == 0 {
            return Err(ConfigurationError::InvalidValue("cleanup_interval_seconds must be greater than 0".to_string()));
        }
        
        Ok(())
    }
}

impl MaintenanceSettings {
    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if !(0.0..=1.0).contains(&self.defragment_threshold) {
            return Err(ConfigurationError::InvalidValue("defragment_threshold must be between 0.0 and 1.0".to_string()));
        }
        
        if !(0.0..=10.0).contains(&self.max_fragmentation_ratio) {
            return Err(ConfigurationError::InvalidValue("max_fragmentation_ratio must be between 0.0 and 10.0".to_string()));
        }
        
        if self.consistency_check_interval_minutes == 0 {
            return Err(ConfigurationError::InvalidValue("consistency_check_interval_minutes must be greater than 0".to_string()));
        }
        
        Ok(())
    }
}

impl MonitoringSettings {
    pub fn validate(&self) -> Result<(), ConfigurationError> {
        if self.max_tracked_queries == 0 {
            return Err(ConfigurationError::InvalidValue("max_tracked_queries must be greater than 0".to_string()));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ConfigurationError {
    InvalidValue(String),
    SerializationError(String),
    ValidationError(String),
}

impl std::fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigurationError::InvalidValue(msg) => write!(f, "Invalid configuration value: {}", msg),
            ConfigurationError::SerializationError(msg) => write!(f, "Configuration serialization error: {}", msg),
            ConfigurationError::ValidationError(msg) => write!(f, "Configuration validation error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigurationError {}
```

### 3. Add configuration presets
Add these preset configurations:
```rust
impl CacheConfiguration {
    pub fn default() -> Self {
        Self::balanced()
    }
    
    pub fn minimal() -> Self {
        Self {
            max_entries: 100,
            max_memory_mb: 10,
            eviction: EvictionSettings {
                policy: EvictionPolicyConfig::LRU,
                memory_pressure_threshold: 0.8,
                aggressive_eviction_threshold: 0.95,
                min_entries_to_keep: 5,
                max_eviction_batch_size: 10,
                size_weight: 0.0,
            },
            performance: PerformanceSettings {
                enable_prefetch: false,
                prefetch_batch_size: 5,
                enable_compression: false,
                compression_threshold_bytes: 1024,
                enable_background_cleanup: false,
                cleanup_interval_seconds: 300,
            },
            maintenance: MaintenanceSettings {
                auto_defragment: false,
                defragment_threshold: 0.5,
                max_fragmentation_ratio: 2.0,
                validate_consistency: false,
                consistency_check_interval_minutes: 60,
            },
            monitoring: MonitoringSettings {
                enable_detailed_stats: false,
                track_query_patterns: false,
                max_tracked_queries: 100,
                enable_performance_logging: false,
                log_slow_operations_ms: 1000,
            },
        }
    }
    
    pub fn balanced() -> Self {
        Self {
            max_entries: 1000,
            max_memory_mb: 100,
            eviction: EvictionSettings {
                policy: EvictionPolicyConfig::Hybrid { size_weight: 0.3 },
                memory_pressure_threshold: 0.75,
                aggressive_eviction_threshold: 0.9,
                min_entries_to_keep: 20,
                max_eviction_batch_size: 50,
                size_weight: 0.3,
            },
            performance: PerformanceSettings {
                enable_prefetch: true,
                prefetch_batch_size: 10,
                enable_compression: true,
                compression_threshold_bytes: 4096,
                enable_background_cleanup: true,
                cleanup_interval_seconds: 180,
            },
            maintenance: MaintenanceSettings {
                auto_defragment: true,
                defragment_threshold: 0.3,
                max_fragmentation_ratio: 1.5,
                validate_consistency: true,
                consistency_check_interval_minutes: 30,
            },
            monitoring: MonitoringSettings {
                enable_detailed_stats: true,
                track_query_patterns: true,
                max_tracked_queries: 500,
                enable_performance_logging: true,
                log_slow_operations_ms: 500,
            },
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            max_entries: 5000,
            max_memory_mb: 500,
            eviction: EvictionSettings {
                policy: EvictionPolicyConfig::Adaptive { 
                    initial_policy: Box::new(EvictionPolicyConfig::LRU) 
                },
                memory_pressure_threshold: 0.7,
                aggressive_eviction_threshold: 0.85,
                min_entries_to_keep: 100,
                max_eviction_batch_size: 100,
                size_weight: 0.2,
            },
            performance: PerformanceSettings {
                enable_prefetch: true,
                prefetch_batch_size: 20,
                enable_compression: true,
                compression_threshold_bytes: 2048,
                enable_background_cleanup: true,
                cleanup_interval_seconds: 120,
            },
            maintenance: MaintenanceSettings {
                auto_defragment: true,
                defragment_threshold: 0.2,
                max_fragmentation_ratio: 1.3,
                validate_consistency: true,
                consistency_check_interval_minutes: 15,
            },
            monitoring: MonitoringSettings {
                enable_detailed_stats: true,
                track_query_patterns: true,
                max_tracked_queries: 1000,
                enable_performance_logging: true,
                log_slow_operations_ms: 200,
            },
        }
    }
    
    pub fn memory_constrained() -> Self {
        Self {
            max_entries: 200,
            max_memory_mb: 20,
            eviction: EvictionSettings {
                policy: EvictionPolicyConfig::SizeBased,
                memory_pressure_threshold: 0.6,
                aggressive_eviction_threshold: 0.8,
                min_entries_to_keep: 10,
                max_eviction_batch_size: 20,
                size_weight: 1.0,
            },
            performance: PerformanceSettings {
                enable_prefetch: false,
                prefetch_batch_size: 3,
                enable_compression: true,
                compression_threshold_bytes: 512,
                enable_background_cleanup: true,
                cleanup_interval_seconds: 60,
            },
            maintenance: MaintenanceSettings {
                auto_defragment: true,
                defragment_threshold: 0.4,
                max_fragmentation_ratio: 1.8,
                validate_consistency: false,
                consistency_check_interval_minutes: 120,
            },
            monitoring: MonitoringSettings {
                enable_detailed_stats: false,
                track_query_patterns: false,
                max_tracked_queries: 50,
                enable_performance_logging: false,
                log_slow_operations_ms: 2000,
            },
        }
    }
}
```

### 4. Add TOML serialization support
Add these methods for configuration file handling:
```rust
impl CacheConfiguration {
    pub fn from_toml_file(path: &str) -> Result<Self, ConfigurationError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to read config file: {}", e)))?;
        
        let config: Self = toml::from_str(&content)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to parse TOML: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    pub fn to_toml_file(&self, path: &str) -> Result<(), ConfigurationError> {
        self.validate()?;
        
        let toml_content = toml::to_string_pretty(self)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to serialize to TOML: {}", e)))?;
        
        std::fs::write(path, toml_content)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    pub fn from_toml_string(toml_content: &str) -> Result<Self, ConfigurationError> {
        let config: Self = toml::from_str(toml_content)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to parse TOML: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    pub fn to_toml_string(&self) -> Result<String, ConfigurationError> {
        self.validate()?;
        
        toml::to_string_pretty(self)
            .map_err(|e| ConfigurationError::SerializationError(format!("Failed to serialize to TOML: {}", e)))
    }
    
    pub fn generate_example_config_file(path: &str) -> Result<(), ConfigurationError> {
        let example_config = Self::balanced();
        example_config.to_toml_file(path)
    }
}
```

### 5. Update MemoryEfficientCache to use new configuration
Update the cache constructor and add reconfiguration support:
```rust
impl MemoryEfficientCache {
    pub fn from_config(config: CacheConfiguration) -> Result<Self, ConfigurationError> {
        config.validate()?;
        
        let eviction_config = EvictionConfig {
            primary_policy: match config.eviction.policy {
                EvictionPolicyConfig::LRU => EvictionPolicy::LRU,
                EvictionPolicyConfig::LFU => EvictionPolicy::LFU,
                EvictionPolicyConfig::SizeBased => EvictionPolicy::SizeBased,
                EvictionPolicyConfig::Hybrid { size_weight } => EvictionPolicy::Hybrid,
                EvictionPolicyConfig::Adaptive { initial_policy: _ } => EvictionPolicy::LRU, // For now
            },
            memory_pressure_threshold: config.eviction.memory_pressure_threshold,
            aggressive_eviction_threshold: config.eviction.aggressive_eviction_threshold,
            min_entries_to_keep: config.eviction.min_entries_to_keep,
            max_eviction_batch_size: config.eviction.max_eviction_batch_size,
            size_weight: config.eviction.size_weight,
        };
        
        Ok(Self::with_eviction_policy(config.max_entries, config.max_memory_mb, eviction_config))
    }
    
    pub fn reconfigure(&mut self, new_config: CacheConfiguration) -> Result<(), ConfigurationError> {
        new_config.validate()?;
        
        // Update basic limits
        self.max_entries = new_config.max_entries;
        self.max_memory_mb = new_config.max_memory_mb;
        
        // Update eviction configuration
        self.eviction_config.memory_pressure_threshold = new_config.eviction.memory_pressure_threshold;
        self.eviction_config.aggressive_eviction_threshold = new_config.eviction.aggressive_eviction_threshold;
        self.eviction_config.min_entries_to_keep = new_config.eviction.min_entries_to_keep;
        self.eviction_config.max_eviction_batch_size = new_config.eviction.max_eviction_batch_size;
        self.eviction_config.size_weight = new_config.eviction.size_weight;
        
        // Apply new limits if necessary
        self.enforce_new_limits()?;
        
        Ok(())
    }
    
    fn enforce_new_limits(&self) -> Result<(), ConfigurationError> {
        let mut cache = self.query_cache.write().unwrap();
        let mut memory_usage = self.current_memory_usage.write().unwrap();
        
        // Enforce entry limit
        while cache.len() > self.max_entries {
            let candidates = self.select_eviction_candidates(&cache, 1);
            if candidates.is_empty() {
                break;
            }
            self.evict_entries(&mut cache, &mut memory_usage, candidates);
        }
        
        // Enforce memory limit
        let max_memory_bytes = self.max_memory_mb * 1024 * 1024;
        while *memory_usage > max_memory_bytes {
            let bytes_to_free = *memory_usage - max_memory_bytes;
            let entries_to_remove = self.estimate_entries_for_bytes(&cache, bytes_to_free);
            let candidates = self.select_eviction_candidates(&cache, entries_to_remove);
            
            if candidates.is_empty() {
                break;
            }
            
            self.evict_entries(&mut cache, &mut memory_usage, candidates);
        }
        
        Ok(())
    }
    
    pub fn get_current_config(&self) -> CacheConfiguration {
        CacheConfiguration {
            max_entries: self.max_entries,
            max_memory_mb: self.max_memory_mb,
            eviction: EvictionSettings {
                policy: match self.eviction_config.primary_policy {
                    EvictionPolicy::LRU => EvictionPolicyConfig::LRU,
                    EvictionPolicy::LFU => EvictionPolicyConfig::LFU,
                    EvictionPolicy::SizeBased => EvictionPolicyConfig::SizeBased,
                    EvictionPolicy::Hybrid => EvictionPolicyConfig::Hybrid { 
                        size_weight: self.eviction_config.size_weight 
                    },
                },
                memory_pressure_threshold: self.eviction_config.memory_pressure_threshold,
                aggressive_eviction_threshold: self.eviction_config.aggressive_eviction_threshold,
                min_entries_to_keep: self.eviction_config.min_entries_to_keep,
                max_eviction_batch_size: self.eviction_config.max_eviction_batch_size,
                size_weight: self.eviction_config.size_weight,
            },
            // Set defaults for other settings (not implemented in basic cache yet)
            performance: PerformanceSettings {
                enable_prefetch: false,
                prefetch_batch_size: 10,
                enable_compression: false,
                compression_threshold_bytes: 4096,
                enable_background_cleanup: false,
                cleanup_interval_seconds: 300,
            },
            maintenance: MaintenanceSettings {
                auto_defragment: false,
                defragment_threshold: 0.3,
                max_fragmentation_ratio: 1.5,
                validate_consistency: false,
                consistency_check_interval_minutes: 30,
            },
            monitoring: MonitoringSettings {
                enable_detailed_stats: false,
                track_query_patterns: false,
                max_tracked_queries: 100,
                enable_performance_logging: false,
                log_slow_operations_ms: 1000,
            },
        }
    }
}
```

### 6. Add configuration management tests
Add these tests to the test module:
```rust
#[test]
fn test_configuration_validation() {
    let mut config = CacheConfiguration::minimal();
    assert!(config.validate().is_ok());
    
    // Invalid max_entries
    config.max_entries = 0;
    assert!(config.validate().is_err());
    config.max_entries = 100;
    
    // Invalid memory pressure threshold
    config.eviction.memory_pressure_threshold = 1.5;
    assert!(config.validate().is_err());
    config.eviction.memory_pressure_threshold = 0.8;
    
    // Invalid threshold ordering
    config.eviction.aggressive_eviction_threshold = 0.7;
    assert!(config.validate().is_err());
}

#[test]
fn test_configuration_presets() {
    let minimal = CacheConfiguration::minimal();
    let balanced = CacheConfiguration::balanced();
    let performance = CacheConfiguration::performance_optimized();
    let memory_constrained = CacheConfiguration::memory_constrained();
    
    assert!(minimal.validate().is_ok());
    assert!(balanced.validate().is_ok());
    assert!(performance.validate().is_ok());
    assert!(memory_constrained.validate().is_ok());
    
    // Performance config should have higher limits than minimal
    assert!(performance.max_entries > minimal.max_entries);
    assert!(performance.max_memory_mb > minimal.max_memory_mb);
    
    // Memory constrained should be smaller than balanced
    assert!(memory_constrained.max_memory_mb < balanced.max_memory_mb);
}

#[test]
fn test_toml_serialization() {
    let config = CacheConfiguration::balanced();
    
    // Test to_toml_string
    let toml_string = config.to_toml_string().unwrap();
    assert!(toml_string.contains("max_entries"));
    assert!(toml_string.contains("max_memory_mb"));
    assert!(toml_string.contains("[eviction]"));
    
    // Test round-trip serialization
    let deserialized = CacheConfiguration::from_toml_string(&toml_string).unwrap();
    assert_eq!(config.max_entries, deserialized.max_entries);
    assert_eq!(config.max_memory_mb, deserialized.max_memory_mb);
}

#[test]
fn test_cache_from_config() {
    let config = CacheConfiguration::minimal();
    let cache = MemoryEfficientCache::from_config(config.clone()).unwrap();
    
    assert_eq!(cache.max_entries(), config.max_entries);
    assert_eq!(cache.max_memory_mb(), config.max_memory_mb);
}

#[test]
fn test_cache_reconfiguration() {
    let initial_config = CacheConfiguration::minimal();
    let mut cache = MemoryEfficientCache::from_config(initial_config).unwrap();
    
    // Add some test data
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    cache.put("test".to_string(), test_results);
    assert_eq!(cache.current_entries(), 1);
    
    // Reconfigure to smaller limits
    let mut new_config = CacheConfiguration::minimal();
    new_config.max_entries = 50;
    new_config.max_memory_mb = 5;
    
    assert!(cache.reconfigure(new_config).is_ok());
    assert_eq!(cache.max_entries(), 50);
    assert_eq!(cache.max_memory_mb(), 5);
}

#[test]
fn test_memory_constraint_application() {
    let mut config = CacheConfiguration::performance_optimized();
    
    // Should reduce limits when available memory is low
    config.apply_memory_constraints(100); // 100MB available
    assert!(config.max_memory_mb <= 50); // Should be <= 50% of available
}

#[test]
fn test_configuration_error_handling() {
    // Test invalid TOML
    let invalid_toml = "invalid toml content [[[";
    assert!(CacheConfiguration::from_toml_string(invalid_toml).is_err());
    
    // Test validation error propagation
    let invalid_config = r#"
        max_entries = 0
        max_memory_mb = 100
        [eviction]
        policy = "LRU"
        memory_pressure_threshold = 0.8
        aggressive_eviction_threshold = 0.9
        min_entries_to_keep = 10
        max_eviction_batch_size = 50
        size_weight = 0.0
    "#;
    
    assert!(CacheConfiguration::from_toml_string(invalid_config).is_err());
}
```

## Success Criteria
- [ ] Comprehensive cache configuration with validation
- [ ] Configuration presets for different use cases
- [ ] TOML serialization and deserialization
- [ ] Runtime reconfiguration without data loss
- [ ] Memory constraint application
- [ ] Error handling for invalid configurations
- [ ] All tests pass for configuration management
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Configuration validation prevents invalid cache states
- Presets provide optimized configurations for different scenarios
- TOML support enables configuration files
- Runtime reconfiguration allows cache tuning without restart
- Memory constraints prevent system resource exhaustion
- Serde integration enables easy serialization