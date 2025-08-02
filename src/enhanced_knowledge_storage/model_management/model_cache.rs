//! Model Cache with LRU Eviction
//! 
//! Implements an LRU (Least Recently Used) cache for loaded models to optimize
//! memory usage and model loading times. Handles model lifecycles and eviction policies.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::enhanced_knowledge_storage::types::*;

/// Backend type for model handle
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackendType {
    Local,
    Remote,
    Mock,
}

/// Handle to a loaded model instance
#[derive(Debug, Clone)]
pub struct ModelHandle {
    pub id: String,
    pub model_type: String,
    pub metadata: ModelMetadata,
    pub loaded_at: Instant,
    pub memory_usage: u64,
    pub backend_type: BackendType,
}

impl ModelHandle {
    pub fn new(id: String, model_type: String, metadata: ModelMetadata) -> Self {
        Self {
            memory_usage: metadata.memory_footprint,
            id,
            model_type,
            metadata,
            loaded_at: Instant::now(),
            backend_type: BackendType::Remote, // Default to remote
        }
    }
    
    pub fn with_backend_type(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }
}

/// Cached model entry with access tracking
#[derive(Debug, Clone)]
pub struct CachedModel {
    pub handle: ModelHandle,
    pub last_used: Instant,
    pub use_count: u64,
    pub is_loading: bool,
}

impl CachedModel {
    pub fn new(handle: ModelHandle) -> Self {
        Self {
            handle,
            last_used: Instant::now(),
            use_count: 1,
            is_loading: false,
        }
    }
    
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }
    
    pub fn age(&self) -> Duration {
        self.last_used.elapsed()
    }
    
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.age() > timeout
    }
}

/// LRU cache for managing loaded model instances
#[derive(Debug)]
pub struct ModelCache {
    models: HashMap<String, CachedModel>,
    access_order: Vec<String>, // Most recently used at the end
    max_capacity: usize,
    total_memory_usage: u64,
}

impl ModelCache {
    /// Create a new model cache with specified capacity
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            models: HashMap::new(),
            access_order: Vec::new(),
            max_capacity,
            total_memory_usage: 0,
        }
    }
    
    /// Create a new model cache with default capacity
    pub fn new() -> Self {
        Self::with_capacity(10) // Default capacity of 10 models
    }
    
    /// Get a cached model, updating its access time
    pub fn get(&mut self, model_id: &str) -> Option<CachedModel> {
        if self.models.contains_key(model_id) {
            // Update usage in two separate steps to avoid borrow checker issues
            if let Some(model) = self.models.get_mut(model_id) {
                model.mark_used();
            }
            self.update_access_order(model_id);
            
            // Return cloned model
            self.models.get(model_id).cloned()
        } else {
            None
        }
    }
    
    /// Get a cached model without updating access time (read-only)
    pub fn peek(&self, model_id: &str) -> Option<&CachedModel> {
        self.models.get(model_id)
    }
    
    /// Insert a new model into the cache
    pub fn insert(&mut self, model_id: String, model: CachedModel) {
        // Remove existing entry if present
        if self.models.contains_key(&model_id) {
            self.remove(&model_id);
        }
        
        // Add memory usage
        self.total_memory_usage += model.handle.memory_usage;
        
        // Insert the model
        self.models.insert(model_id.clone(), model);
        self.access_order.push(model_id);
        
        // Enforce capacity limits
        self.enforce_capacity_limits();
    }
    
    /// Remove a model from the cache
    pub fn remove(&mut self, model_id: &str) -> Option<CachedModel> {
        if let Some(model) = self.models.remove(model_id) {
            // Remove from access order
            self.access_order.retain(|id| id != model_id);
            
            // Subtract memory usage
            self.total_memory_usage = self.total_memory_usage
                .saturating_sub(model.handle.memory_usage);
            
            Some(model)
        } else {
            None
        }
    }
    
    /// Mark a model as used (update access time)
    pub fn mark_used(&mut self, model_id: &str) {
        if let Some(model) = self.models.get_mut(model_id) {
            model.mark_used();
            self.update_access_order(model_id);
        }
    }
    
    /// Get the least recently used model ID
    pub fn get_least_recently_used(&self) -> Option<String> {
        self.access_order.first().cloned()
    }
    
    /// Clear expired models based on timeout
    pub fn clear_expired(&mut self, timeout: Duration) {
        let expired_ids: Vec<String> = self.models
            .iter()
            .filter(|(_, model)| model.is_expired(timeout))
            .map(|(id, _)| id.clone())
            .collect();
        
        for id in expired_ids {
            self.remove(&id);
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_models: self.models.len(),
            max_capacity: self.max_capacity,
            total_memory_usage: self.total_memory_usage,
            utilization_percent: (self.models.len() as f32 / self.max_capacity as f32) * 100.0,
        }
    }
    
    /// Check if cache contains a model
    pub fn contains(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }
    
    /// Get all cached model IDs
    pub fn get_model_ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
    
    /// Clear all models from cache
    pub fn clear(&mut self) {
        self.models.clear();
        self.access_order.clear();
        self.total_memory_usage = 0;
    }
    
    /// Get total memory usage
    pub fn total_memory_usage(&self) -> u64 {
        self.total_memory_usage
    }
    
    /// Check if cache is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.models.len() >= self.max_capacity
    }
    
    /// Get models ordered by last access (oldest first)
    pub fn get_models_by_access_order(&self) -> Vec<String> {
        self.access_order.clone()
    }
    
    /// Force eviction of N least recently used models
    pub fn evict_lru_models(&mut self, count: usize) -> Vec<String> {
        let mut evicted = Vec::new();
        
        for _ in 0..count.min(self.models.len()) {
            if let Some(lru_id) = self.get_least_recently_used() {
                self.remove(&lru_id);
                evicted.push(lru_id);
            }
        }
        
        evicted
    }
    
    /// Evict models until memory usage is below threshold
    pub fn evict_until_memory_below(&mut self, memory_threshold: u64) -> Vec<String> {
        let mut evicted = Vec::new();
        
        while self.total_memory_usage > memory_threshold && !self.models.is_empty() {
            if let Some(lru_id) = self.get_least_recently_used() {
                self.remove(&lru_id);
                evicted.push(lru_id);
            } else {
                break;
            }
        }
        
        evicted
    }
    
    /// Update the access order for a model
    fn update_access_order(&mut self, model_id: &str) {
        // Remove from current position
        self.access_order.retain(|id| id != model_id);
        // Add to end (most recently used)
        self.access_order.push(model_id.to_string());
    }
    
    /// Enforce capacity limits by evicting LRU models
    fn enforce_capacity_limits(&mut self) {
        while self.models.len() > self.max_capacity {
            if let Some(lru_id) = self.get_least_recently_used() {
                self.remove(&lru_id);
            } else {
                break;
            }
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_models: usize,
    pub max_capacity: usize,
    pub total_memory_usage: u64,
    pub utilization_percent: f32,
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::types::ComplexityLevel;
    
    fn create_test_model(id: &str, memory: u64) -> CachedModel {
        let metadata = ModelMetadata {
            name: format!("Test Model {id}"),
            parameters: 1000000,
            memory_footprint: memory,
            complexity_level: ComplexityLevel::Low,
            model_type: "Test".to_string(),
            huggingface_id: format!("test/{id}"),
            supported_tasks: vec!["test".to_string()],
        };
        
        let handle = ModelHandle::new(
            id.to_string(),
            "Test".to_string(),
            metadata,
        );
        
        CachedModel::new(handle)
    }
    
    #[test]
    fn test_cache_insertion_and_retrieval() {
        let mut cache = ModelCache::with_capacity(3);
        let model = create_test_model("test1", 1000000);
        
        cache.insert("test1".to_string(), model);
        
        assert!(cache.contains("test1"));
        assert_eq!(cache.stats().total_models, 1);
        assert_eq!(cache.total_memory_usage(), 1000000);
        
        let retrieved = cache.get("test1");
        assert!(retrieved.is_some());
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = ModelCache::with_capacity(2);
        
        // Insert two models
        cache.insert("model1".to_string(), create_test_model("model1", 1000000));
        cache.insert("model2".to_string(), create_test_model("model2", 2000000));
        
        assert_eq!(cache.stats().total_models, 2);
        
        // Access model1 to make it more recently used
        cache.get("model1");
        
        // Insert third model, should evict model2 (least recently used)
        cache.insert("model3".to_string(), create_test_model("model3", 3000000));
        
        assert_eq!(cache.stats().total_models, 2);
        assert!(cache.contains("model1"));
        assert!(!cache.contains("model2"));
        assert!(cache.contains("model3"));
    }
    
    #[test]
    fn test_memory_based_eviction() {
        let mut cache = ModelCache::with_capacity(10);
        
        // Add models with significant memory usage
        cache.insert("model1".to_string(), create_test_model("model1", 5000000));
        cache.insert("model2".to_string(), create_test_model("model2", 3000000));
        cache.insert("model3".to_string(), create_test_model("model3", 2000000));
        
        assert_eq!(cache.total_memory_usage(), 10000000);
        
        // Evict until memory is below 6MB
        let evicted = cache.evict_until_memory_below(6000000);
        
        assert!(!evicted.is_empty());
        assert!(cache.total_memory_usage() <= 6000000);
    }
    
    #[test]
    fn test_expired_model_cleanup() {
        let mut cache = ModelCache::new();
        let mut model = create_test_model("test1", 1000000);
        
        // Manually set last_used to be expired
        model.last_used = Instant::now() - Duration::from_secs(3600); // 1 hour ago
        
        cache.insert("test1".to_string(), model);
        
        // Clear expired models with 30 minute timeout
        cache.clear_expired(Duration::from_secs(1800));
        
        assert!(!cache.contains("test1"));
        assert_eq!(cache.stats().total_models, 0);
    }
    
    #[test]
    fn test_access_order_tracking() {
        let mut cache = ModelCache::with_capacity(3);
        
        cache.insert("model1".to_string(), create_test_model("model1", 1000000));
        cache.insert("model2".to_string(), create_test_model("model2", 1000000));
        cache.insert("model3".to_string(), create_test_model("model3", 1000000));
        
        // Access in specific order
        cache.get("model1");
        cache.get("model3");
        cache.get("model2");
        
        let access_order = cache.get_models_by_access_order();
        // Most recent should be at the end
        assert_eq!(access_order.last(), Some(&"model2".to_string()));
        
        // Least recent should be model1 (accessed first)
        let lru = cache.get_least_recently_used();
        assert_eq!(lru, Some("model1".to_string()));
    }
    
    #[test]
    fn test_cache_stats() {
        let mut cache = ModelCache::with_capacity(5);
        
        cache.insert("model1".to_string(), create_test_model("model1", 2000000));
        cache.insert("model2".to_string(), create_test_model("model2", 3000000));
        
        let stats = cache.stats();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.max_capacity, 5);
        assert_eq!(stats.total_memory_usage, 5000000);
        assert_eq!(stats.utilization_percent, 40.0); // 2/5 * 100
    }
}