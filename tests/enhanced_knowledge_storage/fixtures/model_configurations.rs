//! Model Configuration Fixtures for Testing
//! 
//! Provides standard model configurations for consistent testing
//! across different test scenarios and environments.

/// Standard model configurations for testing
pub struct TestModelConfigurations;

impl TestModelConfigurations {
    /// Minimal model configuration for basic testing
    pub fn minimal_config() -> ModelConfig {
        ModelConfig {
            name: "test-minimal".to_string(),
            model_type: ModelType::SmallLanguageModel,
            memory_limit_mb: 100,
            processing_threads: 1,
            cache_enabled: false,
            optimization_level: OptimizationLevel::None,
        }
    }
    
    /// Standard model configuration for typical testing
    pub fn standard_config() -> ModelConfig {
        ModelConfig {
            name: "test-standard".to_string(),
            model_type: ModelType::MediumLanguageModel,
            memory_limit_mb: 500,
            processing_threads: 2,
            cache_enabled: true,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
    
    /// High-performance model configuration for stress testing
    pub fn high_performance_config() -> ModelConfig {
        ModelConfig {
            name: "test-high-perf".to_string(),
            model_type: ModelType::LargeLanguageModel,
            memory_limit_mb: 2000,
            processing_threads: 4,
            cache_enabled: true,
            optimization_level: OptimizationLevel::Performance,
        }
    }
    
    /// Mock embedding model configuration
    pub fn embedding_config() -> ModelConfig {
        ModelConfig {
            name: "test-embedding".to_string(),
            model_type: ModelType::EmbeddingModel,
            memory_limit_mb: 200,
            processing_threads: 1,
            cache_enabled: true,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

/// Mock model configuration structure
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub memory_limit_mb: u64,
    pub processing_threads: u32,
    pub cache_enabled: bool,
    pub optimization_level: OptimizationLevel,
}

/// Model types for testing
#[derive(Debug, Clone)]
pub enum ModelType {
    SmallLanguageModel,
    MediumLanguageModel, 
    LargeLanguageModel,
    EmbeddingModel,
    SpecializedModel,
}

/// Optimization levels for testing
#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    Memory,
    Speed,
    Balanced,
    Performance,
}

/// Resource constraints for testing different scenarios
pub struct ResourceConstraints;

impl ResourceConstraints {
    pub fn low_memory_constraint() -> ResourceConstraint {
        ResourceConstraint {
            max_memory_mb: 256,
            max_cpu_threads: 1,
            max_gpu_memory_mb: 0,
            disk_cache_enabled: false,
        }
    }
    
    pub fn standard_constraint() -> ResourceConstraint {
        ResourceConstraint {
            max_memory_mb: 1024,
            max_cpu_threads: 2,
            max_gpu_memory_mb: 512,
            disk_cache_enabled: true,
        }
    }
    
    pub fn high_resource_constraint() -> ResourceConstraint {
        ResourceConstraint {
            max_memory_mb: 4096,
            max_cpu_threads: 8,
            max_gpu_memory_mb: 2048,
            disk_cache_enabled: true,
        }
    }
}

/// Resource constraint configuration
#[derive(Debug, Clone)]
pub struct ResourceConstraint {
    pub max_memory_mb: u64,
    pub max_cpu_threads: u32,
    pub max_gpu_memory_mb: u64,
    pub disk_cache_enabled: bool,
}

/// Test model paths and identifiers
pub struct TestModelPaths;

impl TestModelPaths {
    pub fn mock_model_path() -> &'static str { "tests/fixtures/mock-model" }
    pub fn small_model_path() -> &'static str { "tests/fixtures/small-model" }
    pub fn embedding_model_path() -> &'static str { "tests/fixtures/embedding-model" }
    pub fn invalid_model_path() -> &'static str { "tests/fixtures/nonexistent-model" }
}