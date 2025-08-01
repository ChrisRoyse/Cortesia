//! Model-related mocks for testing
//! 
//! Provides mock implementations for model loading, resource management,
//! and processing components.

use std::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use mockall::mock;

// Mock traits for testable components
mock! {
    pub ModelBackend {}
    
    impl Clone for ModelBackend {
        fn clone(&self) -> Self;
    }
    
    #[async_trait::async_trait]
    impl ModelBackendTrait for ModelBackend {
        async fn load_model(&self, model_id: &str) -> Result<ModelHandle, ModelError>;
        async fn unload_model(&self, handle: ModelHandle) -> Result<(), ModelError>;
        async fn generate_text(&self, handle: &ModelHandle, prompt: &str, max_tokens: Option<u32>) -> Result<String, ModelError>;
        fn get_memory_usage(&self, handle: &ModelHandle) -> u64;
        fn get_model_info(&self, handle: &ModelHandle) -> ModelInfo;
    }
}

mock! {
    pub ResourceMonitor {}
    
    impl Clone for ResourceMonitor {
        fn clone(&self) -> Self;
    }
    
    impl ResourceMonitorTrait for ResourceMonitor {
        fn current_memory_usage(&self) -> u64;
        fn available_memory(&self) -> u64;
        fn active_model_count(&self) -> usize;
        fn add_memory_usage(&mut self, amount: u64);
        fn remove_memory_usage(&mut self, amount: u64);
    }
}

mock! {
    pub ModelCache {}
    
    impl Clone for ModelCache {
        fn clone(&self) -> Self;
    }
    
    impl ModelCacheTrait for ModelCache {
        fn get(&self, model_id: &str) -> Option<CachedModel>;
        fn insert(&mut self, model_id: String, model: CachedModel);
        fn remove(&mut self, model_id: &str) -> Option<CachedModel>;
        fn mark_used(&mut self, model_id: &str);
        fn get_least_recently_used(&self) -> Option<String>;
        fn clear_expired(&mut self, timeout: Duration);
    }
}

// Supporting types for the mocks
#[derive(Debug, Clone)]
pub struct ModelHandle {
    pub id: String,
    pub model_type: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: u64,
    pub memory_footprint: u64,
    pub complexity_level: String,
}

#[derive(Debug, Clone)]
pub struct CachedModel {
    pub handle: ModelHandle,
    pub info: ModelInfo,
    pub last_used: std::time::Instant,
    pub memory_usage: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    #[error("Model loading failed: {0}")]
    LoadingFailed(String),
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

// Trait definitions for mocking
#[async_trait::async_trait]
pub trait ModelBackendTrait {
    async fn load_model(&self, model_id: &str) -> Result<ModelHandle, ModelError>;
    async fn unload_model(&self, handle: ModelHandle) -> Result<(), ModelError>;
    async fn generate_text(&self, handle: &ModelHandle, prompt: &str, max_tokens: Option<u32>) -> Result<String, ModelError>;
    fn get_memory_usage(&self, handle: &ModelHandle) -> u64;
    fn get_model_info(&self, handle: &ModelHandle) -> ModelInfo;
}

pub trait ResourceMonitorTrait {
    fn current_memory_usage(&self) -> u64;
    fn available_memory(&self) -> u64;
    fn active_model_count(&self) -> usize;
    fn add_memory_usage(&mut self, amount: u64);
    fn remove_memory_usage(&mut self, amount: u64);
}

pub trait ModelCacheTrait {
    fn get(&self, model_id: &str) -> Option<CachedModel>;
    fn insert(&mut self, model_id: String, model: CachedModel);
    fn remove(&mut self, model_id: &str) -> Option<CachedModel>;
    fn mark_used(&mut self, model_id: &str);
    fn get_least_recently_used(&self) -> Option<String>;
    fn clear_expired(&mut self, timeout: Duration);
}

// Mock factory functions for common test scenarios
pub fn create_mock_model_backend_with_standard_models() -> MockModelBackend {
    let mut mock_backend = MockModelBackend::new();
    
    // Setup expectations for SmolLM models
    mock_backend
        .expect_load_model()
        .with(mockall::predicate::eq("smollm2_135m"))
        .returning(|_| Ok(ModelHandle {
            id: "smollm2_135m_handle".to_string(),
            model_type: "SmolLM2-135M".to_string(),
        }));
    
    mock_backend
        .expect_load_model() 
        .with(mockall::predicate::eq("smollm2_360m"))
        .returning(|_| Ok(ModelHandle {
            id: "smollm2_360m_handle".to_string(),
            model_type: "SmolLM2-360M".to_string(),
        }));
        
    mock_backend
        .expect_get_memory_usage()
        .returning(|handle| {
            match handle.model_type.as_str() {
                "SmolLM2-135M" => 270_000_000, // ~270MB
                "SmolLM2-360M" => 720_000_000, // ~720MB
                _ => 100_000_000, // Default
            }
        });
        
    mock_backend
        .expect_get_model_info()
        .returning(|handle| {
            match handle.model_type.as_str() {
                "SmolLM2-135M" => ModelInfo {
                    name: "SmolLM2-135M".to_string(),
                    parameters: 135_000_000,
                    memory_footprint: 270_000_000,
                    complexity_level: "Low".to_string(),
                },
                "SmolLM2-360M" => ModelInfo {
                    name: "SmolLM2-360M".to_string(),
                    parameters: 360_000_000,
                    memory_footprint: 720_000_000,
                    complexity_level: "Medium".to_string(),
                },
                _ => ModelInfo {
                    name: "Unknown".to_string(),
                    parameters: 0,
                    memory_footprint: 100_000_000,
                    complexity_level: "Low".to_string(),
                },
            }
        });
    
    mock_backend
}

pub fn create_mock_resource_monitor(initial_memory: u64, max_memory: u64) -> Arc<Mutex<MockResourceMonitor>> {
    let mut mock_monitor = MockResourceMonitor::new();
    
    mock_monitor
        .expect_current_memory_usage()
        .returning(move || initial_memory);
        
    mock_monitor
        .expect_available_memory()
        .returning(move || max_memory.saturating_sub(initial_memory));
        
    mock_monitor
        .expect_active_model_count()
        .returning(|| 0);
    
    Arc::new(Mutex::new(mock_monitor))
}

pub fn create_mock_model_cache() -> Arc<Mutex<MockModelCache>> {
    let mut mock_cache = MockModelCache::new();
    
    // Setup LRU cache behavior
    mock_cache
        .expect_get()
        .returning(|_| None); // Initially empty
        
    mock_cache
        .expect_insert()
        .returning(|_, _| ());
        
    mock_cache
        .expect_remove()
        .returning(|_| None);
        
    mock_cache
        .expect_mark_used()
        .returning(|_| ());
        
    mock_cache
        .expect_get_least_recently_used()
        .returning(|| None);
        
    mock_cache
        .expect_clear_expired()
        .returning(|_| ());
    
    Arc::new(Mutex::new(mock_cache))
}

// Test data builders for consistent test scenarios
pub struct ModelTestDataBuilder {
    models: HashMap<String, (ModelInfo, u64)>, // model_id -> (info, memory_usage)
}

impl Default for ModelTestDataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelTestDataBuilder {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    pub fn with_small_model(mut self, model_id: &str) -> Self {
        self.models.insert(
            model_id.to_string(),
            (
                ModelInfo {
                    name: format!("Small-{model_id}"),
                    parameters: 135_000_000,
                    memory_footprint: 270_000_000,
                    complexity_level: "Low".to_string(),
                },
                270_000_000,
            ),
        );
        self
    }
    
    pub fn with_medium_model(mut self, model_id: &str) -> Self {
        self.models.insert(
            model_id.to_string(),
            (
                ModelInfo {
                    name: format!("Medium-{model_id}"),
                    parameters: 360_000_000,
                    memory_footprint: 720_000_000,
                    complexity_level: "Medium".to_string(),
                },
                720_000_000,
            ),
        );
        self
    }
    
    pub fn with_large_model(mut self, model_id: &str) -> Self {
        self.models.insert(
            model_id.to_string(),
            (
                ModelInfo {
                    name: format!("Large-{model_id}"),
                    parameters: 1_700_000_000,
                    memory_footprint: 3_400_000_000,
                    complexity_level: "High".to_string(),
                },
                3_400_000_000,
            ),
        );
        self
    }
    
    pub fn build(self) -> HashMap<String, (ModelInfo, u64)> {
        self.models
    }
}