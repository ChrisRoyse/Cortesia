//! Model Loader
//! 
//! Handles asynchronous loading and initialization of small language models.
//! Provides caching, error handling, and resource management during model loading.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use crate::enhanced_knowledge_storage::types::*;
use crate::enhanced_knowledge_storage::model_management::{ModelHandle, ModelRegistry};

/// Trait for model backend implementations
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Load a model by ID, returning a handle to the loaded model
    async fn load_model(&self, model_id: &str) -> Result<ModelHandle>;
    
    /// Unload a model, freeing its resources
    async fn unload_model(&self, handle: ModelHandle) -> Result<()>;
    
    /// Generate text using a loaded model
    async fn generate_text(&self, handle: &ModelHandle, prompt: &str, max_tokens: Option<u32>) -> Result<String>;
    
    /// Get current memory usage of a model
    fn get_memory_usage(&self, handle: &ModelHandle) -> u64;
    
    /// Get model information
    fn get_model_info(&self, handle: &ModelHandle) -> ModelMetadata;
    
    /// Check if backend is healthy
    async fn health_check(&self) -> Result<()>;
}

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    /// Maximum time to wait for model loading
    pub load_timeout: Duration,
    /// Number of retry attempts for failed loads
    pub max_retries: u32,
    /// Delay between retry attempts
    pub retry_delay: Duration,
    /// Enable concurrent loading of multiple models
    pub enable_concurrent_loading: bool,
    /// Maximum number of models loading concurrently
    pub max_concurrent_loads: usize,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            load_timeout: Duration::from_secs(60),
            max_retries: 3,
            retry_delay: Duration::from_secs(2),
            enable_concurrent_loading: true,
            max_concurrent_loads: 2,
        }
    }
}

/// Statistics about model loading operations
#[derive(Debug, Clone, Default)]
pub struct LoaderStats {
    pub total_loads: u64,
    pub successful_loads: u64,
    pub failed_loads: u64,
    pub cache_hits: u64,
    pub total_load_time: Duration,
    pub currently_loading: usize,
}

impl LoaderStats {
    pub fn success_rate(&self) -> f32 {
        if self.total_loads == 0 {
            0.0
        } else {
            (self.successful_loads as f32 / self.total_loads as f32) * 100.0
        }
    }
    
    pub fn average_load_time(&self) -> Duration {
        if self.successful_loads == 0 {
            Duration::ZERO
        } else {
            self.total_load_time / self.successful_loads as u32
        }
    }
}

/// Asynchronous model loader with caching and resource management
pub struct ModelLoader {
    backend: Arc<dyn ModelBackend>,
    registry: Arc<RwLock<ModelRegistry>>,
    config: ModelLoaderConfig,
    stats: Arc<Mutex<LoaderStats>>,
    loading_models: Arc<Mutex<std::collections::HashSet<String>>>, // Track currently loading models
}

impl ModelLoader {
    /// Create a new model loader with the given backend and registry
    pub fn new(
        backend: Arc<dyn ModelBackend>,
        registry: Arc<RwLock<ModelRegistry>>,
        config: ModelLoaderConfig,
    ) -> Self {
        Self {
            backend,
            registry,
            config,
            stats: Arc::new(Mutex::new(LoaderStats::default())),
            loading_models: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }
    
    /// Load a model by ID, with caching and error handling
    pub async fn load_model(&self, model_id: &str) -> Result<ModelHandle> {
        let start_time = Instant::now();
        
        // Check if model is already being loaded
        {
            let loading = self.loading_models.lock().await;
            if loading.contains(model_id) {
                return Err(EnhancedStorageError::ModelLoadingFailed(
                    format!("Model {} is already being loaded", model_id)
                ));
            }
        }
        
        // Mark model as loading
        {
            let mut loading = self.loading_models.lock().await;
            loading.insert(model_id.to_string());
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_loads += 1;
            stats.currently_loading += 1;
        }
        
        // Perform the actual loading with retries
        let result = self.load_with_retries(model_id).await;
        
        // Clean up loading state
        {
            let mut loading = self.loading_models.lock().await;
            loading.remove(model_id);
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.currently_loading -= 1;
            
            match &result {
                Ok(_) => {
                    stats.successful_loads += 1;
                    stats.total_load_time += start_time.elapsed();
                }
                Err(_) => {
                    stats.failed_loads += 1;
                }
            }
        }
        
        result
    }
    
    /// Unload a model, freeing its resources
    pub async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        self.backend.unload_model(handle).await
    }
    
    /// Generate text using a loaded model
    pub async fn generate_text(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        self.backend.generate_text(handle, prompt, max_tokens).await
    }
    
    /// Get current loader statistics
    pub async fn stats(&self) -> LoaderStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }
    
    /// Get models currently being loaded
    pub async fn currently_loading(&self) -> Vec<String> {
        let loading = self.loading_models.lock().await;
        loading.iter().cloned().collect()
    }
    
    /// Check if a model is currently being loaded
    pub async fn is_loading(&self, model_id: &str) -> bool {
        let loading = self.loading_models.lock().await;
        loading.contains(model_id)
    }
    
    /// Perform health check on the backend
    pub async fn health_check(&self) -> Result<()> {
        self.backend.health_check().await
    }
    
    /// Load model with retry logic
    async fn load_with_retries(&self, model_id: &str) -> Result<ModelHandle> {
        let mut last_error = None;
        
        for attempt in 0..=self.config.max_retries {
            match self.try_load_model(model_id).await {
                Ok(handle) => return Ok(handle),
                Err(e) => {
                    last_error = Some(e);
                    
                    // Don't retry on the last attempt
                    if attempt < self.config.max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            EnhancedStorageError::ModelLoadingFailed(
                format!("Failed to load model {} after {} attempts", model_id, self.config.max_retries + 1)
            )
        }))
    }
    
    /// Single attempt to load a model
    async fn try_load_model(&self, model_id: &str) -> Result<ModelHandle> {
        // Check if model exists in registry
        let registry = self.registry.read().await;
        if !registry.has_model(model_id) {
            return Err(EnhancedStorageError::ModelNotFound(
                format!("Model {} not found in registry", model_id)
            ));
        }
        drop(registry);
        
        // Load model with timeout
        let load_future = self.backend.load_model(model_id);
        
        match tokio::time::timeout(self.config.load_timeout, load_future).await {
            Ok(result) => result,
            Err(_) => Err(EnhancedStorageError::ModelLoadingFailed(
                format!("Model {} loading timed out after {:?}", model_id, self.config.load_timeout)
            )),
        }
    }
}

/// Mock backend implementation for testing
#[derive(Debug)]
pub struct MockModelBackend {
    load_delay: Duration,
    fail_probability: f32,
}

impl MockModelBackend {
    pub fn new() -> Self {
        Self {
            load_delay: Duration::from_millis(100),
            fail_probability: 0.0,
        }
    }
    
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.load_delay = delay;
        self
    }
    
    pub fn with_failure_rate(mut self, fail_probability: f32) -> Self {
        self.fail_probability = fail_probability.clamp(0.0, 1.0);
        self
    }
}

#[async_trait::async_trait]
impl ModelBackend for MockModelBackend {
    async fn load_model(&self, model_id: &str) -> Result<ModelHandle> {
        // Simulate loading time
        tokio::time::sleep(self.load_delay).await;
        
        // Simulate random failures
        if self.fail_probability > 0.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            model_id.hash(&mut hasher);
            let hash = hasher.finish();
            let random = (hash % 1000) as f32 / 1000.0;
            
            if random < self.fail_probability {
                return Err(EnhancedStorageError::ModelLoadingFailed(
                    format!("Simulated failure loading model {}", model_id)
                ));
            }
        }
        
        // Create mock metadata
        let metadata = ModelMetadata {
            name: format!("Mock {}", model_id),
            parameters: match model_id {
                id if id.contains("135m") => 135_000_000,
                id if id.contains("360m") => 360_000_000,
                id if id.contains("1_7b") => 1_700_000_000,
                _ => 100_000_000,
            },
            memory_footprint: match model_id {
                id if id.contains("135m") => 270_000_000,
                id if id.contains("360m") => 720_000_000,
                id if id.contains("1_7b") => 3_400_000_000,
                _ => 200_000_000,
            },
            complexity_level: ComplexityLevel::Medium,
            model_type: "Mock Model".to_string(),
            huggingface_id: format!("mock/{}", model_id),
            supported_tasks: vec!["text_generation".to_string()],
        };
        
        Ok(ModelHandle::new(
            model_id.to_string(),
            "Mock".to_string(),
            metadata,
        ))
    }
    
    async fn unload_model(&self, _handle: ModelHandle) -> Result<()> {
        // Simulate unloading time
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }
    
    async fn generate_text(&self, handle: &ModelHandle, prompt: &str, _max_tokens: Option<u32>) -> Result<String> {
        // Simulate generation time
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(format!(
            "Generated by {} for prompt: {}",
            handle.metadata.name,
            prompt.chars().take(50).collect::<String>()
        ))
    }
    
    fn get_memory_usage(&self, handle: &ModelHandle) -> u64 {
        handle.memory_usage
    }
    
    fn get_model_info(&self, handle: &ModelHandle) -> ModelMetadata {
        handle.metadata.clone()
    }
    
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_loader_creation() {
        let backend = Arc::new(MockModelBackend::new());
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let config = ModelLoaderConfig::default();
        
        let loader = ModelLoader::new(backend, registry, config);
        let stats = loader.stats().await;
        
        assert_eq!(stats.total_loads, 0);
        assert_eq!(stats.currently_loading, 0);
    }
    
    #[tokio::test]
    async fn test_successful_model_loading() {
        let backend = Arc::new(MockModelBackend::new());
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let config = ModelLoaderConfig::default();
        
        let loader = ModelLoader::new(backend, registry, config);
        
        let handle = loader.load_model("smollm2_360m").await.unwrap();
        assert_eq!(handle.id, "smollm2_360m");
        
        let stats = loader.stats().await;
        assert_eq!(stats.total_loads, 1);
        assert_eq!(stats.successful_loads, 1);
        assert_eq!(stats.failed_loads, 0);
    }
    
    #[tokio::test]
    async fn test_model_loading_with_failures() {
        let backend = Arc::new(MockModelBackend::new().with_failure_rate(1.0)); // Always fail
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let mut config = ModelLoaderConfig::default();
        config.max_retries = 2;
        
        let loader = ModelLoader::new(backend, registry, config);
        
        let result = loader.load_model("smollm2_360m").await;
        assert!(result.is_err());
        
        let stats = loader.stats().await;
        assert_eq!(stats.total_loads, 1);
        assert_eq!(stats.successful_loads, 0);
        assert_eq!(stats.failed_loads, 1);
    }
    
    #[tokio::test]
    async fn test_concurrent_loading_prevention() {
        let backend = Arc::new(MockModelBackend::new().with_delay(Duration::from_millis(500)));
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let config = ModelLoaderConfig::default();
        
        let loader = Arc::new(ModelLoader::new(backend, registry, config));
        
        // Start loading the same model concurrently
        let loader1 = loader.clone();
        let loader2 = loader.clone();
        
        let handle1 = tokio::spawn(async move {
            loader1.load_model("smollm2_360m").await
        });
        
        // Small delay to ensure first loader starts
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let handle2 = tokio::spawn(async move {
            loader2.load_model("smollm2_360m").await
        });
        
        let results = tokio::join!(handle1, handle2);
        
        // One should succeed, one should fail with "already loading" error
        match (results.0.unwrap(), results.1.unwrap()) {
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => (),
            _ => panic!("Expected one success and one failure"),
        }
    }
    
    #[tokio::test]
    async fn test_text_generation() {
        let backend = Arc::new(MockModelBackend::new());
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let config = ModelLoaderConfig::default();
        
        let loader = ModelLoader::new(backend, registry, config);
        
        let handle = loader.load_model("smollm2_360m").await.unwrap();
        let result = loader.generate_text(&handle, "Test prompt", Some(100)).await.unwrap();
        
        assert!(result.contains("Generated by"));
        assert!(result.contains("Test prompt"));
    }
    
    #[tokio::test]
    async fn test_nonexistent_model_loading() {
        let backend = Arc::new(MockModelBackend::new());
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let config = ModelLoaderConfig::default();
        
        let loader = ModelLoader::new(backend, registry, config);
        
        let result = loader.load_model("nonexistent_model").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            EnhancedStorageError::ModelNotFound(_) => (),
            _ => panic!("Expected ModelNotFound error"),
        }
    }
}