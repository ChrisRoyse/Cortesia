//! Intelligent Caching Layer
//! 
//! Multi-level caching system for AI components with intelligent cache strategies,
//! LRU eviction, and optional distributed caching via Redis.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use lru::LruCache;
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn, error, instrument};

use super::types::*;

/// Configuration for caching layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Maximum entries in L1 (in-memory) cache
    pub l1_max_entries: usize,
    
    /// TTL for entity cache entries (seconds)
    pub entity_ttl_seconds: u64,
    
    /// TTL for embedding cache entries (seconds)
    pub embedding_ttl_seconds: u64,
    
    /// TTL for reasoning cache entries (seconds)
    pub reasoning_ttl_seconds: u64,
    
    /// Maximum size for individual cache entries (bytes)
    pub max_entry_size: usize,
    
    /// Redis connection URL (if distributed caching enabled)
    pub redis_url: Option<String>,
    
    /// Enable cache compression
    pub enable_compression: bool,
    
    /// Cache warming strategy
    pub warming_strategy: CacheWarmingStrategy,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 10000,
            entity_ttl_seconds: 3600,      // 1 hour
            embedding_ttl_seconds: 7200,   // 2 hours
            reasoning_ttl_seconds: 1800,   // 30 minutes
            max_entry_size: 1024 * 1024,   // 1MB
            redis_url: None,
            enable_compression: true,
            warming_strategy: CacheWarmingStrategy::Lazy,
        }
    }
}

/// Cache warming strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheWarmingStrategy {
    /// Load cache entries on-demand
    Lazy,
    /// Pre-populate cache with frequent queries
    Eager,
    /// Predict and cache likely queries
    Predictive,
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    data: CachedResult,
    created_at: SystemTime,
    last_accessed: SystemTime,
    access_count: u64,
    ttl: Duration,
}

impl CacheEntry {
    fn new(data: CachedResult, ttl: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }
    
    fn is_expired(&self) -> bool {
        self.created_at.elapsed().unwrap_or(Duration::MAX) > self.ttl
    }
    
    fn access(&mut self) -> &CachedResult {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
        &self.data
    }
}

/// Statistics for cache performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub expired_entries: u64,
    pub current_size: usize,
    pub max_size: usize,
    pub average_hit_time: Duration,
    pub average_miss_time: Duration,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.cache_hits as f32 / self.total_requests as f32) * 100.0
        }
    }
    
    pub fn miss_rate(&self) -> f32 {
        100.0 - self.hit_rate()
    }
}

/// Multi-level intelligent caching system
pub struct IntelligentCachingLayer {
    // L1 Cache - In-memory LRU cache
    l1_entities: Arc<Mutex<LruCache<String, CacheEntry>>>,
    l1_embeddings: Arc<Mutex<LruCache<String, CacheEntry>>>,
    l1_reasoning: Arc<Mutex<LruCache<String, CacheEntry>>>,
    
    // L2 Cache - Optional Redis distributed cache
    #[cfg(feature = "redis")]
    redis_client: Option<redis::Client>,
    
    config: CachingConfig,
    stats: Arc<RwLock<CacheStats>>,
    
    // Cache warming and prediction
    frequent_queries: Arc<RwLock<HashMap<String, u64>>>,
    last_cleanup: Arc<Mutex<Instant>>,
}

impl IntelligentCachingLayer {
    /// Create new intelligent caching layer
    pub fn new() -> AIResult<Self> {
        let config = CachingConfig::default();
        Self::with_config(config)
    }
    
    /// Create with custom configuration
    pub fn with_config(config: CachingConfig) -> AIResult<Self> {
        let l1_entities = Arc::new(Mutex::new(
            LruCache::new(std::num::NonZeroUsize::new(config.l1_max_entries / 3).unwrap())
        ));
        let l1_embeddings = Arc::new(Mutex::new(
            LruCache::new(std::num::NonZeroUsize::new(config.l1_max_entries / 3).unwrap())
        ));
        let l1_reasoning = Arc::new(Mutex::new(
            LruCache::new(std::num::NonZeroUsize::new(config.l1_max_entries / 3).unwrap())
        ));
        
        #[cfg(feature = "redis")]
        let redis_client = if let Some(ref redis_url) = config.redis_url {
            Some(redis::Client::open(redis_url.as_str())
                .map_err(|e| AIComponentError::CachingError(format!("Redis connection failed: {e}")))?)
        } else {
            None
        };
        
        let mut stats = CacheStats::default();
        stats.max_size = config.l1_max_entries;
        
        Ok(Self {
            l1_entities,
            l1_embeddings,
            l1_reasoning,
            #[cfg(feature = "redis")]
            redis_client,
            config,
            stats: Arc::new(RwLock::new(stats)),
            frequent_queries: Arc::new(RwLock::new(HashMap::new())),
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
        })
    }
    
    /// Cache entity extraction results
    #[instrument(skip(self, entities), fields(entity_count = entities.len()))]
    pub async fn cache_entities(&self, key: &str, entities: &[Entity]) -> AIResult<()> {
        let start_time = Instant::now();
        
        // Validate entry size
        let serialized_size = bincode::serialized_size(entities)
            .map_err(|e| AIComponentError::CachingError(format!("Serialization error: {e}")))?;
        
        if serialized_size as usize > self.config.max_entry_size {
            warn!("Entity cache entry too large: {} bytes, skipping", serialized_size);
            return Ok(());
        }
        
        let ttl = Duration::from_secs(self.config.entity_ttl_seconds);
        let entry = CacheEntry::new(
            CachedResult::Entities(entities.to_vec()),
            ttl
        );
        
        // Store in L1 cache
        {
            let mut l1_cache = self.l1_entities.lock().await;
            l1_cache.put(key.to_string(), entry.clone());
        }
        
        // Store in L2 cache if enabled
        #[cfg(feature = "redis")]
        if let Some(ref client) = self.redis_client {
            let serialized = bincode::serialize(&entry)
                .map_err(|e| AIComponentError::CachingError(format!("Serialization error: {e}")))?;
            
            let mut conn = client.get_connection()
                .map_err(|e| AIComponentError::CachingError(format!("Redis connection error: {e}")))?;
            
            use redis::Commands;
            let _: () = conn.setex(
                format!("entities:{}", key),
                self.config.entity_ttl_seconds,
                serialized
            ).map_err(|e| AIComponentError::CachingError(format!("Redis set error: {e}")))?;
        }
        
        self.update_stats_for_set(start_time.elapsed()).await;
        debug!("Cached {} entities for key: {}", entities.len(), key);
        
        // Trigger cleanup if needed
        self.maybe_cleanup().await?;
        
        Ok(())
    }
    
    /// Get cached entity extraction results
    #[instrument(skip(self), fields(key = %key))]
    pub async fn get_entities(&self, key: &str) -> AIResult<Option<Vec<Entity>>> {
        let start_time = Instant::now();
        self.update_query_frequency(key).await;
        
        // Try L1 cache first
        {
            let mut l1_cache = self.l1_entities.lock().await;
            if let Some(entry) = l1_cache.get_mut(key) {
                if !entry.is_expired() {
                    let result = match entry.access() {
                        CachedResult::Entities(entities) => Some(entities.clone()),
                        _ => None,
                    };
                    
                    if result.is_some() {
                        self.update_stats_for_hit(start_time.elapsed()).await;
                        debug!("L1 cache hit for entities: {}", key);
                        return Ok(result);
                    }
                } else {
                    // Remove expired entry
                    l1_cache.pop(key);
                    self.increment_expired_count().await;
                }
            }
        }
        
        // Try L2 cache if enabled
        #[cfg(feature = "redis")]
        if let Some(ref client) = self.redis_client {
            match self.get_from_redis(client, &format!("entities:{}", key)).await {
                Ok(Some(entry)) => {
                    if !entry.is_expired() {
                        let result = match &entry.data {
                            CachedResult::Entities(entities) => Some(entities.clone()),
                            _ => None,
                        };
                        
                        if result.is_some() {
                            // Promote to L1
                            let mut l1_cache = self.l1_entities.lock().await;
                            l1_cache.put(key.to_string(), entry);
                            
                            self.update_stats_for_hit(start_time.elapsed()).await;
                            debug!("L2 cache hit for entities: {}", key);
                            return Ok(result);
                        }
                    }
                },
                Ok(None) => {},
                Err(e) => warn!("Redis get error: {}", e),
            }
        }
        
        self.update_stats_for_miss(start_time.elapsed()).await;
        debug!("Cache miss for entities: {}", key);
        Ok(None)
    }
    
    /// Cache embedding results
    #[instrument(skip(self, embedding), fields(embedding_dim = embedding.len()))]
    pub async fn cache_embedding(&self, key: &str, embedding: &[f32]) -> AIResult<()> {
        let start_time = Instant::now();
        
        let ttl = Duration::from_secs(self.config.embedding_ttl_seconds);
        let entry = CacheEntry::new(
            CachedResult::Embeddings(embedding.to_vec()),
            ttl
        );
        
        // Store in L1 cache
        {
            let mut l1_cache = self.l1_embeddings.lock().await;
            l1_cache.put(key.to_string(), entry.clone());
        }
        
        // Store in L2 if enabled
        #[cfg(feature = "redis")]
        if let Some(ref client) = self.redis_client {
            if let Ok(serialized) = bincode::serialize(&entry) {
                if let Ok(mut conn) = client.get_connection() {
                    use redis::Commands;
                    let _: Result<(), _> = conn.setex(
                        format!("embeddings:{}", key),
                        self.config.embedding_ttl_seconds,
                        serialized
                    );
                }
            }
        }
        
        self.update_stats_for_set(start_time.elapsed()).await;
        debug!("Cached embedding with {} dimensions for key: {}", embedding.len(), key);
        
        Ok(())
    }
    
    /// Get cached embedding
    #[instrument(skip(self), fields(key = %key))]
    pub async fn get_embedding(&self, key: &str) -> AIResult<Option<Vec<f32>>> {
        let start_time = Instant::now();
        self.update_query_frequency(key).await;
        
        // Try L1 cache first
        {
            let mut l1_cache = self.l1_embeddings.lock().await;
            if let Some(entry) = l1_cache.get_mut(key) {
                if !entry.is_expired() {
                    let result = match entry.access() {
                        CachedResult::Embeddings(embedding) => Some(embedding.clone()),
                        _ => None,
                    };
                    
                    if result.is_some() {
                        self.update_stats_for_hit(start_time.elapsed()).await;
                        return Ok(result);
                    }
                }
            }
        }
        
        // Try L2 cache
        #[cfg(feature = "redis")]
        if let Some(ref client) = self.redis_client {
            if let Ok(Some(entry)) = self.get_from_redis(client, &format!("embeddings:{}", key)).await {
                if !entry.is_expired() {
                    let result = match &entry.data {
                        CachedResult::Embeddings(embedding) => Some(embedding.clone()),
                        _ => None,
                    };
                    
                    if result.is_some() {
                        // Promote to L1
                        let mut l1_cache = self.l1_embeddings.lock().await;
                        l1_cache.put(key.to_string(), entry);
                        
                        self.update_stats_for_hit(start_time.elapsed()).await;
                        return Ok(result);
                    }
                }
            }
        }
        
        self.update_stats_for_miss(start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Cache reasoning results
    pub async fn cache_reasoning(&self, key: &str, result: &ReasoningResult) -> AIResult<()> {
        let ttl = Duration::from_secs(self.config.reasoning_ttl_seconds);
        let entry = CacheEntry::new(
            CachedResult::Reasoning(result.clone()),
            ttl
        );
        
        let mut l1_cache = self.l1_reasoning.lock().await;
        l1_cache.put(key.to_string(), entry);
        
        debug!("Cached reasoning result for key: {}", key);
        Ok(())
    }
    
    /// Get cached reasoning results
    pub async fn get_reasoning(&self, key: &str) -> AIResult<Option<ReasoningResult>> {
        let start_time = Instant::now();
        self.update_query_frequency(key).await;
        
        let mut l1_cache = self.l1_reasoning.lock().await;
        if let Some(entry) = l1_cache.get_mut(key) {
            if !entry.is_expired() {
                let result = match entry.access() {
                    CachedResult::Reasoning(reasoning) => Some(reasoning.clone()),
                    _ => None,
                };
                
                if result.is_some() {
                    self.update_stats_for_hit(start_time.elapsed()).await;
                    return Ok(result);
                }
            }
        }
        
        self.update_stats_for_miss(start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// Clear all caches
    pub async fn clear(&self) -> AIResult<()> {
        {
            let mut l1_entities = self.l1_entities.lock().await;
            l1_entities.clear();
        }
        {
            let mut l1_embeddings = self.l1_embeddings.lock().await;
            l1_embeddings.clear();
        }
        {
            let mut l1_reasoning = self.l1_reasoning.lock().await;
            l1_reasoning.clear();
        }
        
        // Clear L2 cache if enabled
        #[cfg(feature = "redis")]
        if let Some(ref client) = self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                use redis::Commands;
                let _: Result<(), _> = conn.flushdb();
            }
        }
        
        debug!("Cleared all caches");
        Ok(())
    }
    
    /// Get from Redis L2 cache
    #[cfg(feature = "redis")]
    async fn get_from_redis(&self, client: &redis::Client, key: &str) -> AIResult<Option<CacheEntry>> {
        let mut conn = client.get_connection()
            .map_err(|e| AIComponentError::CachingError(format!("Redis connection error: {e}")))?;
        
        use redis::Commands;
        let data: Option<Vec<u8>> = conn.get(key)
            .map_err(|e| AIComponentError::CachingError(format!("Redis get error: {e}")))?;
        
        if let Some(data) = data {
            let entry: CacheEntry = bincode::deserialize(&data)
                .map_err(|e| AIComponentError::CachingError(format!("Deserialization error: {e}")))?;
            Ok(Some(entry))
        } else {
            Ok(None)  
        }
    }
    
    /// Update query frequency for cache warming
    async fn update_query_frequency(&self, key: &str) {
        let mut freq = self.frequent_queries.write().await;
        *freq.entry(key.to_string()).or_insert(0) += 1;
    }
    
    /// Update statistics for cache hit
    async fn update_stats_for_hit(&self, elapsed: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.cache_hits += 1;
        stats.average_hit_time = Duration::from_nanos(
            ((stats.average_hit_time.as_nanos() as f64 * (stats.cache_hits - 1) as f64) 
            + elapsed.as_nanos() as f64) as u64 / stats.cache_hits
        );
    }
    
    /// Update statistics for cache miss
    async fn update_stats_for_miss(&self, elapsed: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.cache_misses += 1;
        stats.average_miss_time = Duration::from_nanos(
            ((stats.average_miss_time.as_nanos() as f64 * (stats.cache_misses - 1) as f64) 
            + elapsed.as_nanos() as f64) as u64 / stats.cache_misses
        );
    }
    
    /// Update statistics for cache set operation
    async fn update_stats_for_set(&self, _elapsed: Duration) {
        let mut stats = self.stats.write().await;
        stats.current_size = stats.current_size.saturating_add(1).min(stats.max_size);
    }
    
    /// Increment expired entry count
    async fn increment_expired_count(&self) {
        let mut stats = self.stats.write().await;
        stats.expired_entries += 1;
    }
    
    /// Maybe perform cache cleanup
    async fn maybe_cleanup(&self) -> AIResult<()> {
        let mut last_cleanup = self.last_cleanup.lock().await;
        let now = Instant::now();
        
        // Cleanup every 10 minutes
        if now.duration_since(*last_cleanup) > Duration::from_secs(600) {
            self.cleanup_expired_entries().await?;
            *last_cleanup = now;
        }
        
        Ok(())
    }
    
    /// Clean up expired entries from all caches
    async fn cleanup_expired_entries(&self) -> AIResult<()> {
        let start_time = Instant::now();
        let mut total_removed = 0;
        
        // Cleanup L1 caches
        {
            let mut l1_entities = self.l1_entities.lock().await;
            let keys_to_remove: Vec<String> = l1_entities.iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in keys_to_remove {
                l1_entities.pop(&key);
                total_removed += 1;
            }
        }
        
        {
            let mut l1_embeddings = self.l1_embeddings.lock().await;
            let keys_to_remove: Vec<String> = l1_embeddings.iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in keys_to_remove {
                l1_embeddings.pop(&key);
                total_removed += 1;
            }
        }
        
        {
            let mut l1_reasoning = self.l1_reasoning.lock().await;
            let keys_to_remove: Vec<String> = l1_reasoning.iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in keys_to_remove {
                l1_reasoning.pop(&key);
                total_removed += 1;
            }
        }
        
        debug!("Cache cleanup removed {} expired entries in {:?}", 
               total_removed, start_time.elapsed());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_creation() {
        let cache = IntelligentCachingLayer::new().unwrap();
        let stats = cache.get_stats().await;
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.cache_hits, 0);
    }
    
    #[tokio::test]
    async fn test_entity_caching() {
        let cache = IntelligentCachingLayer::new().unwrap();
        
        let entities = vec![
            Entity {
                name: "Test Entity".to_string(),
                entity_type: EntityType::Person,
                start_pos: 0,
                end_pos: 11,
                confidence: 0.9,
                context: "Test context".to_string(),
                attributes: HashMap::new(),
                extracted_at: 1234567890,
            }
        ];
        
        // Cache entities
        cache.cache_entities("test_key", &entities).await.unwrap();
        
        // Retrieve entities
        let cached = cache.get_entities("test_key").await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.total_requests, 1);
    }
    
    #[tokio::test]
    async fn test_embedding_caching() {
        let cache = IntelligentCachingLayer::new().unwrap();
        
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        // Cache embedding
        cache.cache_embedding("embed_key", &embedding).await.unwrap();
        
        // Retrieve embedding
        let cached = cache.get_embedding("embed_key").await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), embedding);
    }
    
    #[tokio::test]
    async fn test_cache_miss() {
        let cache = IntelligentCachingLayer::new().unwrap();
        
        let result = cache.get_entities("nonexistent").await.unwrap();
        assert!(result.is_none());
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let cache = IntelligentCachingLayer::new().unwrap();
        let stats = cache.get_stats().await;
        
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.miss_rate(), 100.0);
        
        // Add some hits and misses
        let _ = cache.get_entities("miss1").await;
        let _ = cache.get_entities("miss2").await;
        
        let entities = vec![];
        cache.cache_entities("hit1", &entities).await.unwrap();
        let _ = cache.get_entities("hit1").await;
        
        let stats = cache.get_stats().await;
        assert!(stats.hit_rate() > 0.0 && stats.hit_rate() < 100.0);
        assert!(stats.miss_rate() > 0.0 && stats.miss_rate() < 100.0);
    }
}