use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tokio::sync::{RwLock, Mutex, Semaphore};
use tokio::time::sleep;
use lru::LruCache;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use std::io::{Write, Read};
use std::path::{Path, PathBuf};
// use std::fs;
use tokio::fs as async_fs;
use regex::Regex;
use log::info;

/// Cache entry metadata for intelligent management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub size_bytes: usize,
    pub version: u64,
    pub ttl: Option<Duration>,
    pub cache_level: CacheLevel,
    pub compression_ratio: Option<f64>,
}

/// Cache levels in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheLevel {
    L1Memory,
    L2Disk,
    L3Distributed,
}

/// Cache entry with metadata and data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub data: T,
    pub metadata: CacheMetadata,
}

/// Cache statistics for monitoring and optimization
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l1_evictions: u64,
    pub l1_size_bytes: usize,
    pub l1_entry_count: usize,
    
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l2_evictions: u64,
    pub l2_size_bytes: usize,
    pub l2_entry_count: usize,
    
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub l3_size_bytes: usize,
    pub l3_entry_count: usize,
    
    pub compression_savings: u64,
    pub total_requests: u64,
    pub stampede_preventions: u64,
    pub cache_warmup_operations: u64,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total_misses = self.l1_misses + self.l2_misses + self.l3_misses;
        let total = total_hits + total_misses;
        if total == 0 { 0.0 } else { total_hits as f64 / total as f64 }
    }
    
    pub fn compression_ratio(&self) -> f64 {
        if self.compression_savings == 0 { 1.0 } 
        else { self.compression_savings as f64 / (self.l1_size_bytes + self.l2_size_bytes) as f64 }
    }
}

/// Cache invalidation strategy
#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    TimeToLive(Duration),
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
    AdaptiveTtl { base_ttl: Duration, access_multiplier: f64 },
    PatternBased(String),
}

/// Cache write strategy
#[derive(Debug, Clone, Copy)]
pub enum WriteStrategy {
    WriteThrough,
    WriteBack,
    WriteBehind { delay: Duration },
}

/// L1 Memory Cache implementation
pub struct L1MemoryCache {
    cache: LruCache<String, Vec<u8>>,
    metadata: HashMap<String, CacheMetadata>,
    max_size_bytes: usize,
    current_size_bytes: usize,
}

impl L1MemoryCache {
    pub fn new(capacity: usize, max_size_bytes: usize) -> Self {
        Self {
            cache: LruCache::new(capacity.try_into().unwrap()),
            metadata: HashMap::new(),
            max_size_bytes,
            current_size_bytes: 0,
        }
    }
    
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(data) = self.cache.get(key) {
            // Update access metadata
            if let Some(meta) = self.metadata.get_mut(key) {
                meta.last_accessed = current_timestamp();
                meta.access_count += 1;
            }
            Some(data.clone())
        } else {
            None
        }
    }
    
    pub fn put(&mut self, key: String, data: Vec<u8>, metadata: CacheMetadata) -> bool {
        let data_size = data.len();
        
        // Check if we need to evict entries
        while self.current_size_bytes + data_size > self.max_size_bytes && !self.cache.is_empty() {
            if let Some((evicted_key, evicted_data)) = self.cache.pop_lru() {
                self.current_size_bytes -= evicted_data.len();
                self.metadata.remove(&evicted_key);
            }
        }
        
        // Only insert if it fits
        if self.current_size_bytes + data_size <= self.max_size_bytes {
            self.cache.put(key.clone(), data);
            self.metadata.insert(key, metadata);
            self.current_size_bytes += data_size;
            true
        } else {
            false
        }
    }
    
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(data) = self.cache.pop(key) {
            self.current_size_bytes -= data.len();
            self.metadata.remove(key);
            Some(data)
        } else {
            None
        }
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
        self.metadata.clear();
        self.current_size_bytes = 0;
    }
    
    pub fn size(&self) -> usize {
        self.cache.len()
    }
    
    pub fn size_bytes(&self) -> usize {
        self.current_size_bytes
    }
}

/// L2 Disk Cache implementation
pub struct L2DiskCache {
    cache_dir: PathBuf,
    max_size_bytes: usize,
    current_size_bytes: Arc<RwLock<usize>>,
    metadata: Arc<RwLock<HashMap<String, CacheMetadata>>>,
    compression_level: u32,
}

impl L2DiskCache {
    pub async fn new(cache_dir: PathBuf, max_size_bytes: usize, compression_level: u32) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            async_fs::create_dir_all(&cache_dir).await?;
        }
        
        let mut cache = Self {
            cache_dir,
            max_size_bytes,
            current_size_bytes: Arc::new(RwLock::new(0)),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            compression_level,
        };
        
        // Load existing metadata
        cache.load_metadata().await?;
        
        Ok(cache)
    }
    
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let file_path = self.get_cache_file_path(key);
        
        if file_path.exists() {
            match async_fs::read(&file_path).await {
                Ok(compressed_data) => {
                    // Update access metadata
                    {
                        let mut metadata = self.metadata.write().await;
                        if let Some(meta) = metadata.get_mut(key) {
                            meta.last_accessed = current_timestamp();
                            meta.access_count += 1;
                        }
                    }
                    
                    // Decompress data
                    self.decompress_data(&compressed_data).ok()
                }
                Err(_) => None,
            }
        } else {
            None
        }
    }
    
    pub async fn put(&self, key: String, data: Vec<u8>, mut metadata: CacheMetadata) -> bool {
        // Compress data
        let compressed_data = match self.compress_data(&data) {
            Ok(compressed) => compressed,
            Err(_) => return false,
        };
        
        let compressed_size = compressed_data.len();
        metadata.size_bytes = compressed_size;
        metadata.compression_ratio = Some(data.len() as f64 / compressed_size as f64);
        
        // Check space and evict if necessary
        self.ensure_space(compressed_size).await;
        
        let file_path = self.get_cache_file_path(&key);
        
        match async_fs::write(&file_path, &compressed_data).await {
            Ok(_) => {
                // Update metadata
                {
                    let mut meta_map = self.metadata.write().await;
                    meta_map.insert(key, metadata);
                }
                
                // Update size
                {
                    let mut size = self.current_size_bytes.write().await;
                    *size += compressed_size;
                }
                
                true
            }
            Err(_) => false,
        }
    }
    
    pub async fn remove(&self, key: &str) -> bool {
        let file_path = self.get_cache_file_path(key);
        
        if file_path.exists() {
            if let Some(metadata) = {
                let mut meta_map = self.metadata.write().await;
                meta_map.remove(key)
            } {
                // Update size
                {
                    let mut size = self.current_size_bytes.write().await;
                    *size = size.saturating_sub(metadata.size_bytes);
                }
                
                // Remove file
                let _ = async_fs::remove_file(&file_path).await;
                true
            } else {
                false
            }
        } else {
            false
        }
    }
    
    pub async fn clear(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Remove all cache files
        let mut entries = async_fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                let _ = async_fs::remove_file(entry.path()).await;
            }
        }
        
        // Clear metadata and size
        {
            let mut metadata = self.metadata.write().await;
            metadata.clear();
        }
        {
            let mut size = self.current_size_bytes.write().await;
            *size = 0;
        }
        
        Ok(())
    }
    
    pub async fn size(&self) -> usize {
        let metadata = self.metadata.read().await;
        metadata.len()
    }
    
    pub async fn size_bytes(&self) -> usize {
        let size = self.current_size_bytes.read().await;
        *size
    }
    
    async fn ensure_space(&self, needed_bytes: usize) {
        let current_size = {
            let size = self.current_size_bytes.read().await;
            *size
        };
        
        if current_size + needed_bytes > self.max_size_bytes {
            // Find LRU entries to evict
            let mut entries_to_evict = Vec::new();
            {
                let metadata = self.metadata.read().await;
                let mut sorted_entries: Vec<_> = metadata.iter().collect();
                sorted_entries.sort_by_key(|(_, meta)| meta.last_accessed);
                
                let mut bytes_to_free = (current_size + needed_bytes) - self.max_size_bytes;
                for (key, meta) in sorted_entries {
                    if bytes_to_free == 0 {
                        break;
                    }
                    entries_to_evict.push(key.clone());
                    bytes_to_free = bytes_to_free.saturating_sub(meta.size_bytes);
                }
            }
            
            // Evict selected entries
            for key in entries_to_evict {
                self.remove(&key).await;
            }
        }
    }
    
    fn get_cache_file_path(&self, key: &str) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.cache_dir.join(format!("{:x}.cache", hash))
    }
    
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }
    
    fn decompress_data(&self, compressed_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
    
    async fn load_metadata(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let metadata_file = self.cache_dir.join("metadata.json");
        if metadata_file.exists() {
            let content = async_fs::read_to_string(&metadata_file).await?;
            let loaded_metadata: HashMap<String, CacheMetadata> = serde_json::from_str(&content)?;
            
            // Calculate current size
            let total_size = loaded_metadata.values().map(|m| m.size_bytes).sum();
            
            {
                let mut metadata = self.metadata.write().await;
                *metadata = loaded_metadata;
            }
            {
                let mut size = self.current_size_bytes.write().await;
                *size = total_size;
            }
        }
        Ok(())
    }
    
    pub async fn save_metadata(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let metadata_file = self.cache_dir.join("metadata.json");
        let metadata = self.metadata.read().await;
        let content = serde_json::to_string_pretty(&*metadata)?;
        async_fs::write(&metadata_file, content).await?;
        Ok(())
    }
}

/// L3 Distributed Cache (Redis/Memcached) interface
#[async_trait::async_trait]
pub trait L3DistributedCache: Send + Sync {
    async fn get(&self, key: &str) -> Option<Vec<u8>>;
    async fn put(&self, key: String, data: Vec<u8>, ttl: Option<Duration>) -> bool;
    async fn remove(&self, key: &str) -> bool;
    async fn clear(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn size(&self) -> usize;
    async fn invalidate_pattern(&self, pattern: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

/// Multi-level intelligent cache system
pub struct MultiLevelCache {
    l1_cache: Arc<RwLock<L1MemoryCache>>,
    l2_cache: Arc<L2DiskCache>,
    l3_cache: Option<Arc<dyn L3DistributedCache>>,
    stats: Arc<RwLock<CacheStatistics>>,
    write_strategy: WriteStrategy,
    stampede_protection: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
    version_counter: Arc<RwLock<u64>>,
    adaptive_ttl_enabled: bool,
}

impl MultiLevelCache {
    pub async fn new(
        l1_capacity: usize,
        l1_max_bytes: usize,
        l2_cache_dir: PathBuf,
        l2_max_bytes: usize,
        l3_cache: Option<Arc<dyn L3DistributedCache>>,
        write_strategy: WriteStrategy,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let l1_cache = Arc::new(RwLock::new(L1MemoryCache::new(l1_capacity, l1_max_bytes)));
        let l2_cache = Arc::new(L2DiskCache::new(l2_cache_dir, l2_max_bytes, 6).await?);
        
        Ok(Self {
            l1_cache,
            l2_cache,
            l3_cache,
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
            write_strategy,
            stampede_protection: Arc::new(Mutex::new(HashMap::new())),
            version_counter: Arc::new(RwLock::new(0)),
            adaptive_ttl_enabled: true,
        })
    }
    
    /// Get value from cache, checking all levels
    pub async fn get<T>(&self, key: &str) -> Option<T>
    where
        T: DeserializeOwned + Send + Sync,
    {
        // Prevent cache stampede
        let semaphore = {
            let mut protection = self.stampede_protection.lock().await;
            protection.entry(key.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(1)))
                .clone()
        };
        
        let _permit = semaphore.acquire().await.ok()?;
        
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        drop(stats);
        
        // Try L1 cache first
        if let Some(data) = {
            let mut l1 = self.l1_cache.write().await;
            l1.get(key)
        } {
            let mut stats = self.stats.write().await;
            stats.l1_hits += 1;
            drop(stats);
            
            return self.deserialize_data(&data).ok();
        } else {
            let mut stats = self.stats.write().await;
            stats.l1_misses += 1;
            drop(stats);
        }
        
        // Try L2 cache
        if let Some(data) = self.l2_cache.get(key).await {
            let mut stats = self.stats.write().await;
            stats.l2_hits += 1;
            drop(stats);
            
            // Promote to L1
            let metadata = CacheMetadata {
                created_at: current_timestamp(),
                last_accessed: current_timestamp(),
                access_count: 1,
                size_bytes: data.len(),
                version: self.get_next_version().await,
                ttl: None,
                cache_level: CacheLevel::L1Memory,
                compression_ratio: None,
            };
            
            {
                let mut l1 = self.l1_cache.write().await;
                l1.put(key.to_string(), data.clone(), metadata);
            }
            
            return self.deserialize_data(&data).ok();
        } else {
            let mut stats = self.stats.write().await;
            stats.l2_misses += 1;
            drop(stats);
        }
        
        // Try L3 cache if available
        if let Some(l3) = &self.l3_cache {
            if let Some(data) = l3.get(key).await {
                let mut stats = self.stats.write().await;
                stats.l3_hits += 1;
                drop(stats);
                
                // Promote to L2 and L1
                let metadata = CacheMetadata {
                    created_at: current_timestamp(),
                    last_accessed: current_timestamp(),
                    access_count: 1,
                    size_bytes: data.len(),
                    version: self.get_next_version().await,
                    ttl: None,
                    cache_level: CacheLevel::L2Disk,
                    compression_ratio: None,
                };
                
                self.l2_cache.put(key.to_string(), data.clone(), metadata.clone()).await;
                
                {
                    let mut l1 = self.l1_cache.write().await;
                    let l1_metadata = CacheMetadata {
                        cache_level: CacheLevel::L1Memory,
                        ..metadata
                    };
                    l1.put(key.to_string(), data.clone(), l1_metadata);
                }
                
                return self.deserialize_data(&data).ok();
            } else {
                let mut stats = self.stats.write().await;
                stats.l3_misses += 1;
                drop(stats);
            }
        }
        
        None
    }
    
    /// Put value into cache using configured write strategy
    pub async fn put<T>(&self, key: String, value: T, ttl: Option<Duration>)
    where
        T: Serialize + Send + Sync,
    {
        let data = match self.serialize_data(&value) {
            Ok(data) => data,
            Err(_) => return,
        };
        
        let base_metadata = CacheMetadata {
            created_at: current_timestamp(),
            last_accessed: current_timestamp(),
            access_count: 0,
            size_bytes: data.len(),
            version: self.get_next_version().await,
            ttl,
            cache_level: CacheLevel::L1Memory,
            compression_ratio: None,
        };
        
        match self.write_strategy {
            WriteStrategy::WriteThrough => {
                self.write_through(&key, data, base_metadata).await;
            }
            WriteStrategy::WriteBack => {
                self.write_back(&key, data, base_metadata).await;
            }
            WriteStrategy::WriteBehind { delay } => {
                self.write_behind(&key, data, base_metadata, delay).await;
            }
        }
    }
    
    /// Remove key from all cache levels
    pub async fn remove(&self, key: &str) {
        // Remove from L1
        {
            let mut l1 = self.l1_cache.write().await;
            l1.remove(key);
        }
        
        // Remove from L2
        self.l2_cache.remove(key).await;
        
        // Remove from L3 if available
        if let Some(l3) = &self.l3_cache {
            l3.remove(key).await;
        }
    }
    
    /// Invalidate cache entries matching a pattern
    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let regex = Regex::new(pattern)?;
        let mut invalidated = 0u64;
        
        // Invalidate L1 entries
        {
            let mut l1 = self.l1_cache.write().await;
            let keys_to_remove: Vec<String> = l1.cache.iter()
                .filter_map(|(k, _)| if regex.is_match(k) { Some(k.clone()) } else { None })
                .collect();
            
            for key in keys_to_remove {
                l1.remove(&key);
                invalidated += 1;
            }
        }
        
        // Invalidate L2 entries
        {
            let metadata = self.l2_cache.metadata.read().await;
            let keys_to_remove: Vec<String> = metadata.keys()
                .filter(|k| regex.is_match(k))
                .cloned()
                .collect();
            drop(metadata);
            
            for key in keys_to_remove {
                self.l2_cache.remove(&key).await;
                invalidated += 1;
            }
        }
        
        // Invalidate L3 entries if available
        if let Some(l3) = &self.l3_cache {
            invalidated += l3.invalidate_pattern(pattern).await?;
        }
        
        Ok(invalidated)
    }
    
    /// Warm cache with preloaded data
    pub async fn warm_cache(&self, entries: Vec<(String, Vec<u8>)>) {
        let mut stats = self.stats.write().await;
        stats.cache_warmup_operations += entries.len() as u64;
        drop(stats);
        
        for (key, data) in entries {
            let metadata = CacheMetadata {
                created_at: current_timestamp(),
                last_accessed: current_timestamp(),
                access_count: 0,
                size_bytes: data.len(),
                version: self.get_next_version().await,
                ttl: Some(Duration::from_secs(3600)), // 1 hour default
                cache_level: CacheLevel::L1Memory,
                compression_ratio: None,
            };
            
            // Put in L1 first for immediate access
            {
                let mut l1 = self.l1_cache.write().await;
                l1.put(key.clone(), data.clone(), metadata.clone());
            }
            
            // Also put in L2 for persistence
            let l2_metadata = CacheMetadata {
                cache_level: CacheLevel::L2Disk,
                ..metadata
            };
            self.l2_cache.put(key, data, l2_metadata).await;
        }
        
        info!("Cache warming completed");
    }
    
    /// Get cache statistics
    pub async fn get_statistics(&self) -> CacheStatistics {
        let mut stats = self.stats.read().await.clone();
        
        // Update current sizes
        {
            let l1 = self.l1_cache.read().await;
            stats.l1_size_bytes = l1.size_bytes();
            stats.l1_entry_count = l1.size();
        }
        
        stats.l2_size_bytes = self.l2_cache.size_bytes().await;
        stats.l2_entry_count = self.l2_cache.size().await;
        
        if let Some(l3) = &self.l3_cache {
            stats.l3_entry_count = l3.size().await;
        }
        
        stats
    }
    
    /// Clear all cache levels
    pub async fn clear(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Clear L1
        {
            let mut l1 = self.l1_cache.write().await;
            l1.clear();
        }
        
        // Clear L2
        self.l2_cache.clear().await?;
        
        // Clear L3 if available
        if let Some(l3) = &self.l3_cache {
            l3.clear().await?;
        }
        
        // Reset statistics
        {
            let mut stats = self.stats.write().await;
            *stats = CacheStatistics::default();
        }
        
        Ok(())
    }
    
    /// Calculate adaptive TTL based on access patterns
    fn calculate_adaptive_ttl(&self, access_count: u64, base_ttl: Duration) -> Duration {
        if !self.adaptive_ttl_enabled {
            return base_ttl;
        }
        
        let multiplier = 1.0 + (access_count as f64).ln() * 0.1;
        Duration::from_secs((base_ttl.as_secs() as f64 * multiplier) as u64)
    }
    
    async fn write_through(&self, key: &str, data: Vec<u8>, metadata: CacheMetadata) {
        // Write to all levels simultaneously
        let l1_metadata = CacheMetadata {
            cache_level: CacheLevel::L1Memory,
            ..metadata.clone()
        };
        
        {
            let mut l1 = self.l1_cache.write().await;
            l1.put(key.to_string(), data.clone(), l1_metadata);
        }
        
        let l2_metadata = CacheMetadata {
            cache_level: CacheLevel::L2Disk,
            ..metadata.clone()
        };
        self.l2_cache.put(key.to_string(), data.clone(), l2_metadata).await;
        
        if let Some(l3) = &self.l3_cache {
            l3.put(key.to_string(), data, metadata.ttl).await;
        }
    }
    
    async fn write_back(&self, key: &str, data: Vec<u8>, metadata: CacheMetadata) {
        // Write to L1 immediately, defer L2/L3 writes
        let l1_metadata = CacheMetadata {
            cache_level: CacheLevel::L1Memory,
            ..metadata.clone()
        };
        
        {
            let mut l1 = self.l1_cache.write().await;
            l1.put(key.to_string(), data.clone(), l1_metadata);
        }
        
        // Schedule background writes to L2 and L3
        let l2_cache = self.l2_cache.clone();
        let l3_cache = self.l3_cache.clone();
        let key = key.to_string();
        tokio::spawn(async move {
            let l2_metadata = CacheMetadata {
                cache_level: CacheLevel::L2Disk,
                ..metadata.clone()
            };
            l2_cache.put(key.clone(), data.clone(), l2_metadata).await;
            
            if let Some(l3) = l3_cache {
                l3.put(key, data, metadata.ttl).await;
            }
        });
    }
    
    async fn write_behind(&self, key: &str, data: Vec<u8>, metadata: CacheMetadata, delay: Duration) {
        // Write to L1 immediately
        let l1_metadata = CacheMetadata {
            cache_level: CacheLevel::L1Memory,
            ..metadata.clone()
        };
        
        {
            let mut l1 = self.l1_cache.write().await;
            l1.put(key.to_string(), data.clone(), l1_metadata);
        }
        
        // Schedule delayed writes to L2 and L3
        let l2_cache = self.l2_cache.clone();
        let l3_cache = self.l3_cache.clone();
        let key = key.to_string();
        tokio::spawn(async move {
            sleep(delay).await;
            
            let l2_metadata = CacheMetadata {
                cache_level: CacheLevel::L2Disk,
                ..metadata.clone()
            };
            l2_cache.put(key.clone(), data.clone(), l2_metadata).await;
            
            if let Some(l3) = l3_cache {
                l3.put(key, data, metadata.ttl).await;
            }
        });
    }
    
    async fn get_next_version(&self) -> u64 {
        let mut version = self.version_counter.write().await;
        *version += 1;
        *version
    }
    
    fn serialize_data<T>(&self, value: &T) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Serialize,
    {
        Ok(bincode::serialize(value)?)
    }
    
    fn deserialize_data<T>(&self, data: &[u8]) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
    where
        T: DeserializeOwned,
    {
        Ok(bincode::deserialize(data)?)
    }
}

/// Cache configuration builder
pub struct CacheConfigBuilder {
    l1_capacity: usize,
    l1_max_bytes: usize,
    l2_cache_dir: Option<PathBuf>,
    l2_max_bytes: usize,
    l3_cache: Option<Arc<dyn L3DistributedCache>>,
    write_strategy: WriteStrategy,
    compression_level: u32,
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self {
            l1_capacity: 10000,
            l1_max_bytes: 100 * 1024 * 1024, // 100MB
            l2_cache_dir: None,
            l2_max_bytes: 1024 * 1024 * 1024, // 1GB
            l3_cache: None,
            write_strategy: WriteStrategy::WriteThrough,
            compression_level: 6,
        }
    }
}

impl CacheConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn l1_capacity(mut self, capacity: usize) -> Self {
        self.l1_capacity = capacity;
        self
    }
    
    pub fn l1_max_bytes(mut self, max_bytes: usize) -> Self {
        self.l1_max_bytes = max_bytes;
        self
    }
    
    pub fn l2_cache_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.l2_cache_dir = Some(dir.as_ref().to_path_buf());
        self
    }
    
    pub fn l2_max_bytes(mut self, max_bytes: usize) -> Self {
        self.l2_max_bytes = max_bytes;
        self
    }
    
    pub fn l3_cache(mut self, cache: Arc<dyn L3DistributedCache>) -> Self {
        self.l3_cache = Some(cache);
        self
    }
    
    pub fn write_strategy(mut self, strategy: WriteStrategy) -> Self {
        self.write_strategy = strategy;
        self
    }
    
    pub fn compression_level(mut self, level: u32) -> Self {
        self.compression_level = level;
        self
    }
    
    pub async fn build(self) -> Result<MultiLevelCache, Box<dyn std::error::Error + Send + Sync>> {
        let l2_dir = self.l2_cache_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("cortesia_cache")
        });
        
        MultiLevelCache::new(
            self.l1_capacity,
            self.l1_max_bytes,
            l2_dir,
            self.l2_max_bytes,
            self.l3_cache,
            self.write_strategy,
        ).await
    }
}

// Utility functions
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    use tempfile::TempDir;
    
    #[test]
    async fn test_l1_cache_basic_operations() {
        let mut cache = L1MemoryCache::new(10, 1024);
        let data = b"test data".to_vec();
        let metadata = CacheMetadata {
            created_at: current_timestamp(),
            last_accessed: current_timestamp(),
            access_count: 0,
            size_bytes: data.len(),
            version: 1,
            ttl: None,
            cache_level: CacheLevel::L1Memory,
            compression_ratio: None,
        };
        
        // Test put and get
        assert!(cache.put("test_key".to_string(), data.clone(), metadata));
        assert_eq!(cache.get("test_key"), Some(data));
        
        // Test remove
        assert!(cache.remove("test_key").is_some());
        assert_eq!(cache.get("test_key"), None);
    }
    
    #[test]
    async fn test_l2_cache_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let cache = L2DiskCache::new(temp_dir.path().to_path_buf(), 1024 * 1024, 6).await.unwrap();
        
        let data = b"persistent test data".to_vec();
        let metadata = CacheMetadata {
            created_at: current_timestamp(),
            last_accessed: current_timestamp(),
            access_count: 0,
            size_bytes: data.len(),
            version: 1,
            ttl: None,
            cache_level: CacheLevel::L2Disk,
            compression_ratio: None,
        };
        
        // Test put and get
        assert!(cache.put("persist_key".to_string(), data.clone(), metadata).await);
        assert_eq!(cache.get("persist_key").await, Some(data));
        
        // Test persistence across instance recreation
        drop(cache);
        let cache2 = L2DiskCache::new(temp_dir.path().to_path_buf(), 1024 * 1024, 6).await.unwrap();
        assert!(cache2.get("persist_key").await.is_some());
    }
    
    #[test]
    async fn test_multi_level_cache_integration() {
        let temp_dir = TempDir::new().unwrap();
        let cache = CacheConfigBuilder::new()
            .l1_capacity(10)
            .l1_max_bytes(1024)
            .l2_cache_dir(temp_dir.path())
            .l2_max_bytes(1024 * 1024)
            .write_strategy(WriteStrategy::WriteThrough)
            .build()
            .await
            .unwrap();
        
        // Test storing and retrieving data
        cache.put("test_key".to_string(), "test_value".to_string(), None).await;
        let retrieved: Option<String> = cache.get("test_key").await;
        assert_eq!(retrieved, Some("test_value".to_string()));
        
        // Test cache statistics
        let stats = cache.get_statistics().await;
        assert!(stats.total_requests > 0);
    }
    
    #[test]
    async fn test_cache_invalidation_pattern() {
        let temp_dir = TempDir::new().unwrap();
        let cache = CacheConfigBuilder::new()
            .l2_cache_dir(temp_dir.path())
            .build()
            .await
            .unwrap();
        
        // Put multiple keys
        cache.put("user:123".to_string(), "user data 1".to_string(), None).await;
        cache.put("user:456".to_string(), "user data 2".to_string(), None).await;
        cache.put("session:789".to_string(), "session data".to_string(), None).await;
        
        // Invalidate user keys (WriteThrough means data is in both L1 and L2)
        let invalidated = cache.invalidate_pattern(r"^user:").await.unwrap();
        assert_eq!(invalidated, 4); // 2 keys * 2 cache levels
        
        // Verify invalidation
        let user1: Option<String> = cache.get("user:123").await;
        let user2: Option<String> = cache.get("user:456").await;
        let session: Option<String> = cache.get("session:789").await;
        
        assert!(user1.is_none());
        assert!(user2.is_none());
        assert!(session.is_some());
    }
}