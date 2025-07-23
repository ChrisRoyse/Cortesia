use std::collections::HashMap;
use std::hash::Hash;

/// Simple LRU cache for similarity search results
pub struct LruCache<K, V> {
    map: HashMap<K, (V, usize)>,
    access_order: Vec<K>,
    capacity: usize,
    access_counter: usize,
}

impl<K: Clone + Eq + Hash, V: Clone> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            access_order: Vec::with_capacity(capacity),
            capacity,
            access_counter: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.get(key) {
            let value_clone = value.clone();
            self.access_counter += 1;
            self.map.insert(key.clone(), (value_clone.clone(), self.access_counter));
            Some(value_clone)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.access_counter += 1;
        
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            self.evict_lru();
        }
        
        self.map.insert(key.clone(), (value, self.access_counter));
        
        // Keep track of insertion order for potential optimization
        if !self.access_order.contains(&key) {
            self.access_order.push(key);
        }
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.find_lru_key() {
            self.map.remove(&lru_key);
            self.access_order.retain(|k| k != &lru_key);
        }
    }

    fn find_lru_key(&self) -> Option<K> {
        self.map
            .iter()
            .min_by_key(|(_, (_, access_time))| *access_time)
            .map(|(key, _)| key.clone())
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.access_order.clear();
        self.access_counter = 0;
    }

    pub fn hit_rate(&self) -> f64 {
        if self.access_counter == 0 {
            0.0
        } else {
            // This is a simplified hit rate calculation
            // In a real implementation, we'd track hits vs misses separately
            self.map.len() as f64 / self.access_counter as f64
        }
    }
}

/// Query cache key for similarity search
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    // Quantized query embedding for approximate matching
    quantized_query: Vec<u8>,
    k: usize,
}

impl QueryCacheKey {
    pub fn new(query_embedding: &[f32], k: usize, quantization_levels: u8) -> Self {
        let quantized_query = quantize_embedding(query_embedding, quantization_levels);
        Self {
            quantized_query,
            k,
        }
    }
}

/// Cache for similarity search results
pub type SimilarityCache = LruCache<QueryCacheKey, Vec<(u32, f32)>>;

/// Quantize a floating-point embedding to reduce precision for cache keys
fn quantize_embedding(embedding: &[f32], levels: u8) -> Vec<u8> {
    let scale = (levels - 1) as f32;
    embedding
        .iter()
        .map(|&x| {
            // Normalize to [0, 1] assuming input is roughly in [-1, 1]
            let normalized = (x + 1.0) / 2.0;
            let quantized = (normalized * scale).round().max(0.0).min(scale);
            quantized as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LruCache::new(2);
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LruCache::new(2);
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3); // Should evict "a"
        
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.get(&"c"), Some(3));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_access_order() {
        let mut cache = LruCache::new(2);
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        
        // Access "a" to make it more recently used
        cache.get(&"a");
        
        cache.insert("c", 3); // Should evict "b", not "a"
        
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn test_query_cache_key() {
        let query1 = vec![0.5, 0.3, 0.8];
        let query2 = vec![0.5, 0.3, 0.8]; // Same query
        let query3 = vec![0.5, 0.3, 0.9]; // Different query
        
        let key1 = QueryCacheKey::new(&query1, 10, 16);
        let key2 = QueryCacheKey::new(&query2, 10, 16);
        let key3 = QueryCacheKey::new(&query3, 10, 16);
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_quantize_embedding() {
        let embedding = vec![-1.0, 0.0, 1.0];
        let quantized = quantize_embedding(&embedding, 255);
        
        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0], 0);   // -1.0 -> 0
        assert_eq!(quantized[1], 127); // 0.0 -> middle (254/2 = 127)
        assert_eq!(quantized[2], 254); // 1.0 -> max (255-1 = 254)
    }
}