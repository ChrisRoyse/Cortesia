//! Data Cache System
//! 
//! Provides intelligent caching for test data to improve performance.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: serde_json::Value,
    pub size_bytes: usize,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}

/// Intelligent data cache for test data
pub struct DataCache {
    cache: RwLock<HashMap<String, CacheEntry>>,
    max_size_bytes: usize,
    max_age: Duration,
    stats: RwLock<CacheStats>,
}

impl DataCache {
    /// Create a new data cache
    pub fn new(max_size_bytes: usize, max_age: Duration) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size_bytes,
            max_age,
            stats: RwLock::new(CacheStats {
                total_entries: 0,
                total_size_bytes: 0,
                hit_count: 0,
                miss_count: 0,
                eviction_count: 0,
            }),
        }
    }

    /// Get data from cache
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is still valid
            if entry.created_at.elapsed() < self.max_age {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
                stats.hit_count += 1;
                Some(entry.data.clone())
            } else {
                // Entry expired, remove it
                cache.remove(key);
                stats.total_entries = cache.len();
                stats.total_size_bytes = cache.values().map(|e| e.size_bytes).sum();
                stats.miss_count += 1;
                None
            }
        } else {
            stats.miss_count += 1;
            None
        }
    }

    /// Put data into cache
    pub async fn put(&self, key: String, data: serde_json::Value) -> Result<()> {
        let serialized = serde_json::to_string(&data)?;
        let size_bytes = serialized.len();

        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        // Check if we need to evict entries
        self.evict_if_needed(&mut cache, &mut stats, size_bytes).await;

        let entry = CacheEntry {
            data,
            size_bytes,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
        };

        cache.insert(key, entry);
        stats.total_entries = cache.len();
        stats.total_size_bytes = cache.values().map(|e| e.size_bytes).sum();

        Ok(())
    }

    /// Clear all cached data
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        cache.clear();
        stats.total_entries = 0;
        stats.total_size_bytes = 0;
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        let total = stats.hit_count + stats.miss_count;
        if total > 0 {
            stats.hit_count as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Evict entries if needed to make room
    async fn evict_if_needed(
        &self,
        cache: &mut HashMap<String, CacheEntry>,
        stats: &mut CacheStats,
        new_entry_size: usize,
    ) {
        let current_size: usize = cache.values().map(|e| e.size_bytes).sum();
        
        if current_size + new_entry_size > self.max_size_bytes {
            // Sort entries by LRU (least recently used)
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, entry)| entry.last_accessed);

            // Evict oldest entries until we have enough space
            let mut freed_space = 0;
            let mut keys_to_remove = Vec::new();

            for (key, entry) in entries {
                if current_size + new_entry_size - freed_space <= self.max_size_bytes {
                    break;
                }
                
                freed_space += entry.size_bytes;
                keys_to_remove.push(key.clone());
                stats.eviction_count += 1;
            }

            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_cache_put_get() {
        let cache = DataCache::new(1024 * 1024, Duration::from_secs(60));
        let test_data = json!({"test": "data", "number": 42});

        cache.put("test_key".to_string(), test_data.clone()).await.unwrap();
        let retrieved = cache.get("test_key").await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), test_data);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = DataCache::new(1024 * 1024, Duration::from_millis(10));
        let test_data = json!({"test": "data"});

        cache.put("test_key".to_string(), test_data).await.unwrap();
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        let retrieved = cache.get("test_key").await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = DataCache::new(100, Duration::from_secs(60)); // Very small cache
        
        // Add large data that should trigger eviction
        let large_data = json!({"data": "x".repeat(200)});
        cache.put("large_key".to_string(), large_data).await.unwrap();

        let stats = cache.stats().await;
        assert!(stats.eviction_count > 0 || stats.total_size_bytes <= 100);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = DataCache::new(1024 * 1024, Duration::from_secs(60));
        let test_data = json!({"test": "data"});

        // Put data
        cache.put("test_key".to_string(), test_data).await.unwrap();

        // Get data (hit)
        let _ = cache.get("test_key").await;

        // Try to get non-existent data (miss)
        let _ = cache.get("non_existent").await;

        let stats = cache.stats().await;
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert_eq!(stats.total_entries, 1);

        let hit_rate = cache.hit_rate().await;
        assert_eq!(hit_rate, 0.5);
    }
}