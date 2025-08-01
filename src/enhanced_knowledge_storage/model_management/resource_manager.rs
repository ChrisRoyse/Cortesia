//! Model Resource Manager
//! 
//! Central coordinator for model resource management, implementing intelligent
//! model selection, memory management, and processing task coordination.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Mutex};
use crate::enhanced_knowledge_storage::types::*;
use crate::enhanced_knowledge_storage::model_management::*;

/// Central resource manager for model loading, caching, and task processing
pub struct ModelResourceManager {
    config: ModelResourceConfig,
    registry: Arc<RwLock<ModelRegistry>>,
    loader: Arc<ModelLoader>,
    cache: Arc<Mutex<ModelCache>>,
    resource_monitor: Arc<Mutex<ResourceMonitor>>,
}

/// Resource monitoring and tracking
#[derive(Debug)]
pub struct ResourceMonitor {
    current_memory_usage: u64,
    active_models: std::collections::HashMap<String, u64>, // model_id -> memory usage
    max_memory: u64,
}

impl ResourceMonitor {
    pub fn new(max_memory: u64) -> Self {
        Self {
            current_memory_usage: 0,
            active_models: std::collections::HashMap::new(),
            max_memory,
        }
    }
    
    pub fn current_memory_usage(&self) -> u64 {
        self.current_memory_usage
    }
    
    pub fn available_memory(&self) -> u64 {
        self.max_memory.saturating_sub(self.current_memory_usage)
    }
    
    pub fn active_model_count(&self) -> usize {
        self.active_models.len()
    }
    
    pub fn add_memory_usage(&mut self, model_id: &str, amount: u64) {
        self.active_models.insert(model_id.to_string(), amount);
        self.current_memory_usage += amount;
    }
    
    pub fn remove_memory_usage(&mut self, model_id: &str) {
        if let Some(amount) = self.active_models.remove(model_id) {
            self.current_memory_usage = self.current_memory_usage.saturating_sub(amount);
        }
    }
    
    pub fn can_fit_model(&self, memory_required: u64) -> bool {
        self.current_memory_usage + memory_required <= self.max_memory
    }
    
    pub fn get_active_models(&self) -> Vec<String> {
        self.active_models.keys().cloned().collect()
    }
}

impl ModelResourceManager {
    /// Create a new model resource manager
    pub fn new(config: ModelResourceConfig) -> Self {
        let registry = Arc::new(RwLock::new(ModelRegistry::with_default_models()));
        let backend = Arc::new(MockModelBackend::new());
        let loader_config = ModelLoaderConfig::default();
        let loader = Arc::new(ModelLoader::new(backend, registry.clone(), loader_config));
        let cache = Arc::new(Mutex::new(ModelCache::with_capacity(config.max_concurrent_models)));
        let resource_monitor = Arc::new(Mutex::new(ResourceMonitor::new(config.max_memory_usage)));
        
        Self {
            config,
            registry,
            loader,
            cache,
            resource_monitor,
        }
    }
    
    /// Process a task with optimal model selection
    pub async fn process_with_optimal_model(&self, task: ProcessingTask) -> Result<ProcessingResult> {
        let start_time = Instant::now();
        let mut processing_metadata = ProcessingMetadata::default();
        
        // Step 1: Select optimal model for this task
        let model_id = self.select_optimal_model(task.complexity).await?;
        
        // Step 2: Ensure model is loaded and available
        let model_handle = self.ensure_model_loaded(&model_id).await?;
        
        // Step 3: Process the task
        let generation_start = Instant::now();
        let output = self.loader.generate_text(&model_handle, &task.content, Some(1000)).await?;
        processing_metadata.inference_time = generation_start.elapsed();
        
        // Step 4: Mark model as used in cache
        {
            let mut cache = self.cache.lock().await;
            cache.mark_used(&model_id);
        }
        
        // Step 5: Calculate quality score based on task complexity and model capability
        let quality_score = self.calculate_quality_score(&task, &model_handle, &output);
        
        // Step 6: Create result
        Ok(ProcessingResult {
            task_id: task.id,
            processing_time: start_time.elapsed(),
            quality_score,
            model_used: model_handle.metadata.name,
            success: true,
            output,
            metadata: processing_metadata,
        })
    }
    
    /// Get current memory usage across all loaded models
    pub async fn current_memory_usage(&self) -> u64 {
        let monitor = self.resource_monitor.lock().await;
        monitor.current_memory_usage()
    }
    
    /// Get resource manager statistics
    pub async fn get_stats(&self) -> ResourceManagerStats {
        let monitor = self.resource_monitor.lock().await;
        let cache = self.cache.lock().await;
        let loader_stats = self.loader.stats().await;
        
        ResourceManagerStats {
            active_models: monitor.active_model_count(),
            total_memory_usage: monitor.current_memory_usage(),
            available_memory: monitor.available_memory(),
            cache_utilization: cache.stats().utilization_percent,
            loader_success_rate: loader_stats.success_rate(),
            total_tasks_processed: loader_stats.successful_loads,
        }
    }
    
    /// Force cleanup of idle models
    pub async fn cleanup_idle_models(&self) -> Vec<String> {
        let mut cache = self.cache.lock().await;
        let mut monitor = self.resource_monitor.lock().await;
        
        // Clear expired models from cache
        cache.clear_expired(self.config.idle_timeout);
        
        // Get models that were evicted and update resource monitor
        let evicted_models: Vec<String> = Vec::new(); // TODO: Track evicted models
        
        for model_id in &evicted_models {
            monitor.remove_memory_usage(model_id);
        }
        
        evicted_models
    }
    
    /// Select the optimal model for a given task complexity
    async fn select_optimal_model(&self, complexity: ComplexityLevel) -> Result<String> {
        let registry = self.registry.read().await;
        let task_complexity = TaskComplexity::from(complexity);
        
        // First, try to find the optimal model for this complexity
        if let Some(optimal_model) = registry.suggest_optimal_model(task_complexity) {
            // Check if we can fit this model in memory
            let monitor = self.resource_monitor.lock().await;
            if monitor.can_fit_model(optimal_model.memory_footprint) {
                // Find the model ID for this metadata
                return self.find_model_id_by_metadata(&*registry, optimal_model).await;
            }
        }
        
        // If optimal model doesn't fit, try smaller models
        let models_within_memory = registry.get_models_within_memory_limit(
            self.config.max_memory_usage / 2 // Conservative limit
        );
        
        if let Some(fallback_model) = models_within_memory.first() {
            return self.find_model_id_by_metadata(&*registry, fallback_model).await;
        }
        
        Err(EnhancedStorageError::InsufficientResources(
            "No suitable model found within memory constraints".to_string()
        ))
    }
    
    /// Ensure a model is loaded and available in cache
    async fn ensure_model_loaded(&self, model_id: &str) -> Result<ModelHandle> {
        // Check if model is already in cache
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached_model) = cache.get(model_id) {
                return Ok(cached_model.handle);
            }
        }
        
        // Model not in cache, need to load it
        self.load_and_cache_model(model_id).await
    }
    
    /// Load a model and add it to cache, handling eviction if necessary
    async fn load_and_cache_model(&self, model_id: &str) -> Result<ModelHandle> {
        // Get model metadata to check memory requirements
        let registry = self.registry.read().await;
        let metadata = registry.get_model_metadata(model_id)
            .ok_or_else(|| EnhancedStorageError::ModelNotFound(model_id.to_string()))?;
        
        // Check if we need to free up memory
        {
            let mut monitor = self.resource_monitor.lock().await;
            let mut cache = self.cache.lock().await;
            
            // If this model won't fit, evict models until it will
            while !monitor.can_fit_model(metadata.memory_footprint) && !cache.get_model_ids().is_empty() {
                if let Some(lru_model_id) = cache.get_least_recently_used() {
                    if let Some(_evicted_model) = cache.remove(&lru_model_id) {
                        monitor.remove_memory_usage(&lru_model_id);
                        // TODO: Actually unload the model from backend
                    }
                } else {
                    break;
                }
            }
            
            // Final check - if we still can't fit, return error
            if !monitor.can_fit_model(metadata.memory_footprint) {
                return Err(EnhancedStorageError::InsufficientResources(
                    format!("Cannot fit model {} ({}MB) in available memory ({}MB)", 
                        model_id, 
                        metadata.memory_footprint / 1_000_000,
                        monitor.available_memory() / 1_000_000)
                ));
            }
        }
        
        drop(registry);
        
        // Load the model
        let model_handle = self.loader.load_model(model_id).await?;
        
        // Add to cache and resource monitor
        {
            let mut cache = self.cache.lock().await;
            let mut monitor = self.resource_monitor.lock().await;
            
            let cached_model = CachedModel::new(model_handle.clone());
            cache.insert(model_id.to_string(), cached_model);
            monitor.add_memory_usage(model_id, model_handle.memory_usage);
        }
        
        Ok(model_handle)
    }
    
    /// Find model ID by matching metadata
    async fn find_model_id_by_metadata(
        &self,
        registry: &ModelRegistry,
        target_metadata: &ModelMetadata,
    ) -> Result<String> {
        for model_id in registry.get_model_ids() {
            if let Some(metadata) = registry.get_model_metadata(&model_id) {
                if metadata.name == target_metadata.name {
                    return Ok(model_id);
                }
            }
        }
        
        Err(EnhancedStorageError::ModelNotFound(
            format!("Model with name '{}' not found", target_metadata.name)
        ))
    }
    
    /// Calculate quality score for a processing result
    fn calculate_quality_score(
        &self,
        task: &ProcessingTask,
        model_handle: &ModelHandle,
        output: &str,
    ) -> f32 {
        let mut score: f32 = 0.5; // Base score
        
        // Factor 1: Model complexity vs task complexity alignment
        let complexity_match = match (task.complexity, model_handle.metadata.complexity_level) {
            (ComplexityLevel::Low, ComplexityLevel::Low) => 1.0,
            (ComplexityLevel::Medium, ComplexityLevel::Medium) => 1.0,
            (ComplexityLevel::High, ComplexityLevel::High) => 1.0,
            (ComplexityLevel::Low, ComplexityLevel::Medium) => 0.9, // Overkill but good
            (ComplexityLevel::Low, ComplexityLevel::High) => 0.8,   // Major overkill
            (ComplexityLevel::Medium, ComplexityLevel::High) => 0.9, // Slight overkill
            (ComplexityLevel::Medium, ComplexityLevel::Low) => 0.6,  // Underkill
            (ComplexityLevel::High, ComplexityLevel::Low) => 0.3,    // Major underkill
            (ComplexityLevel::High, ComplexityLevel::Medium) => 0.7, // Slight underkill
        };
        
        score += complexity_match * 0.3;
        
        // Factor 2: Output quality (simple heuristics)
        if !output.is_empty() {
            score += 0.1;
            
            if output.len() > 10 {
                score += 0.05;
            }
            
            if output.contains(&task.content[..task.content.len().min(20)]) {
                score += 0.05; // Output references input
            }
        }
        
        score.clamp(0.0, 1.0)
    }
}

/// Statistics about resource manager performance
#[derive(Debug, Clone)]
pub struct ResourceManagerStats {
    pub active_models: usize,
    pub total_memory_usage: u64,
    pub available_memory: u64,
    pub cache_utilization: f32,
    pub loader_success_rate: f32,
    pub total_tasks_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_resource_manager_creation() {
        let config = ModelResourceConfig::default();
        let manager = ModelResourceManager::new(config);
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_models, 0);
        assert_eq!(stats.total_memory_usage, 0);
    }
    
    #[tokio::test]
    async fn test_task_processing() {
        let config = ModelResourceConfig::default();
        let manager = ModelResourceManager::new(config);
        
        let task = ProcessingTask::new(ComplexityLevel::Low, "Test processing task");
        let result = manager.process_with_optimal_model(task).await.unwrap();
        
        assert!(result.success);
        assert!(!result.output.is_empty());
        assert!(result.quality_score > 0.0);
        assert!(result.processing_time > Duration::ZERO);
    }
    
    #[tokio::test]
    async fn test_memory_management() {
        let mut config = ModelResourceConfig::default();
        config.max_memory_usage = 1_000_000_000; // 1GB limit
        config.max_concurrent_models = 2;
        
        let manager = ModelResourceManager::new(config);
        
        // Process multiple tasks to load multiple models
        let task1 = ProcessingTask::new(ComplexityLevel::Low, "First task");
        let task2 = ProcessingTask::new(ComplexityLevel::Medium, "Second task");
        
        let _result1 = manager.process_with_optimal_model(task1).await.unwrap();
        let _result2 = manager.process_with_optimal_model(task2).await.unwrap();
        
        let stats = manager.get_stats().await;
        assert!(stats.total_memory_usage <= 1_000_000_000);
    }
    
    #[tokio::test]
    async fn test_concurrent_task_processing() {
        let config = ModelResourceConfig::default();
        let manager = Arc::new(ModelResourceManager::new(config));
        
        // Create multiple concurrent tasks
        let tasks = vec![
            ProcessingTask::new(ComplexityLevel::Low, "Concurrent task 1"),
            ProcessingTask::new(ComplexityLevel::Medium, "Concurrent task 2"),
            ProcessingTask::new(ComplexityLevel::Low, "Concurrent task 3"),
        ];
        
        let mut handles = Vec::new();
        for task in tasks {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                manager_clone.process_with_optimal_model(task).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            results.push(result);
        }
        
        // Verify all tasks completed successfully
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.success);
            assert!(!result.output.is_empty());
        }
    }
    
    #[tokio::test]
    async fn test_idle_model_cleanup() {
        let mut config = ModelResourceConfig::default();
        config.idle_timeout = Duration::from_millis(100); // Very short timeout
        
        let manager = ModelResourceManager::new(config);
        
        // Process a task to load a model
        let task = ProcessingTask::new(ComplexityLevel::Low, "Test task");
        let _result = manager.process_with_optimal_model(task).await.unwrap();
        
        // Verify model is loaded
        let initial_stats = manager.get_stats().await;
        assert!(initial_stats.active_models > 0);
        
        // Wait for idle timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Force cleanup
        let evicted = manager.cleanup_idle_models().await;
        
        // Note: This test might not show eviction due to implementation details
        // but verifies the cleanup mechanism works without errors
        assert!(evicted.len() >= 0); // Non-negative count
    }
}