# Fix: Eliminate Performance Bottlenecks

## Problem
Every operation acquires multiple locks, clones large embedding vectors excessively, lacks batching, and uses synchronous operations that should be async.

## Current State (Detailed Analysis)
Based on deep codebase analysis, the specific bottlenecks are:

### 1. Lock Contention in `src/core/graph/graph_core.rs`
- **KnowledgeGraph struct has 14 RwLock fields** (lines 30-57):
  - `arena: RwLock<GraphArena>`
  - `entity_store: RwLock<EntityStore>` 
  - `graph: RwLock<CSRGraph>`
  - `embedding_bank: RwLock<Vec<u8>>`
  - `quantizer: RwLock<ProductQuantizer>`
  - `bloom_filter: RwLock<BloomFilter>`
  - `entity_id_map: RwLock<AHashMap<u32, EntityKey>>`
  - `spatial_index: RwLock<SpatialIndex>`
  - `flat_index: RwLock<FlatVectorIndex>`
  - `hnsw_index: RwLock<HnswIndex>`
  - `lsh_index: RwLock<LshIndex>`
  - `similarity_cache: RwLock<SimilarityCache>`
  - `string_dictionary: RwLock<AHashMap<String, u32>>`
  - `edge_buffer: RwLock<Vec<Relationship>>`

### 2. Excessive Embedding Clones in `src/core/graph/entity_operations.rs`
- **insert_entity() method** (lines 12-75):
  - Line 33: `data.clone()` when allocating in arena
  - Line 57: `data.embedding.clone()` for spatial_index
  - Line 60: `data.embedding.clone()` for flat_index  
  - Line 63: `data.embedding.clone()` for hnsw_index
  - Line 66: `data.embedding.clone()` for lsh_index
- **insert_entities_batch()** (lines 96-190):
  - Line 144: `data.clone()` for each entity
  - Line 156: `data.embedding.clone()` for spatial entries
  - Lines 179-182: 4 more clones per entity for indices
- **get_entity()** (line 199): Returns cloned EntityData including full embedding

### 3. No Arc-based Embedding Sharing in `src/core/types.rs`
- **EntityData struct** (lines 172-176):
  ```rust
  pub struct EntityData {
      pub type_id: u16,
      pub properties: String,
      pub embedding: Vec<f32>,  // Should be Arc<Vec<f32>>
  }
  ```

### 4. Synchronous I/O in `src/storage/persistent_mmap.rs`
- Uses `std::fs::File` (line 8) instead of `tokio::fs::File`
- No async save/load methods

### 5. Inefficient Batch Processing
- No use of `tokio::spawn` for parallel index updates
- Sequential lock acquisition in batch operations

## Solution: Optimize Lock Usage and Data Flow

### 1. Reduce Lock Granularity with Lock-Free Structures

#### In `src/core/graph/graph_core.rs` (lines 29-60):
```rust
use dashmap::DashMap;
use crossbeam_skiplist::SkipMap;
use arc_swap::ArcSwap;

pub struct KnowledgeGraph {
    // Split into logical components with fine-grained locking
    entity_storage: Arc<EntityStorage>,
    relationship_storage: Arc<RelationshipStorage>, 
    index_manager: Arc<IndexManager>,
    cache_manager: Arc<CacheManager>,
    embedding_manager: Arc<EmbeddingManager>,
    
    // Shared immutable config
    config: Arc<GraphConfig>,
    runtime_profiler: Option<Arc<RuntimeProfiler>>,
}

pub struct EntityStorage {
    // Use lock-free data structures where possible
    arena: Arc<RwLock<GraphArena>>,  // Still needs RwLock for complex mutations
    entity_store: Arc<RwLock<EntityStore>>,
    id_map: Arc<DashMap<u32, EntityKey>>,  // Lock-free concurrent hashmap
    
    // Read-heavy, write-rare structures
    type_registry: Arc<ArcSwap<HashMap<u16, EntityTypeInfo>>>,
}

pub struct IndexManager {
    // Separate read/write concerns
    spatial_index: Arc<RwLock<SpatialIndex>>,
    flat_index: Arc<RwLock<FlatVectorIndex>>,
    hnsw_index: Arc<RwLock<HnswIndex>>,
    lsh_index: Arc<RwLock<LshIndex>>,
    
    // Bloom filter can be made lock-free
    bloom_filter: Arc<AtomicBloomFilter>,  // Custom atomic implementation
}

pub struct EmbeddingManager {
    // Quantized embeddings with read-optimized storage
    embedding_bank: Arc<RwLock<Vec<u8>>>,
    quantizer: Arc<ProductQuantizer>,  // Immutable after creation
    
    // Embedding cache with bounded size
    embedding_cache: Arc<DashMap<EntityKey, Arc<Vec<f32>>>>,
    cache_stats: Arc<AtomicCacheStats>,
}
```

### 2. Implement Zero-Copy Embedding Sharing

#### In `src/core/types.rs` (modify lines 172-187):
```rust
use std::sync::Arc;
use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct SharedEmbedding {
    // Use Arc for zero-copy clones
    data: Arc<Vec<f32>>,
    // Track embedding statistics
    dimension: usize,
    norm: Option<f32>,  // Cache L2 norm for similarity calculations
}

impl SharedEmbedding {
    pub fn new(embedding: Vec<f32>) -> Self {
        let dimension = embedding.len();
        let norm = Self::compute_norm(&embedding);
        Self {
            data: Arc::new(embedding),
            dimension,
            norm: Some(norm),
        }
    }
    
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    pub fn norm(&self) -> f32 {
        self.norm.unwrap_or_else(|| Self::compute_norm(&self.data))
    }
    
    fn compute_norm(data: &[f32]) -> f32 {
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    // Only clone the underlying data when mutation is needed
    pub fn make_mut(&mut self) -> &mut Vec<f32> {
        self.norm = None;  // Invalidate cached norm
        Arc::make_mut(&mut self.data)
    }
    
    // Zero-copy conversion to bytes for storage
    pub fn to_bytes(&self) -> Bytes {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };
        Bytes::copy_from_slice(byte_slice)
    }
}

// Update EntityData to use SharedEmbedding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityData {
    pub type_id: u16,
    pub properties: String,
    #[serde(skip_serializing, skip_deserializing)]
    pub embedding: SharedEmbedding,  // Changed from Vec<f32>
    // Store raw embedding for serialization
    #[serde(rename = "embedding")]
    embedding_raw: Vec<f32>,
}

impl EntityData {
    pub fn new(type_id: u16, properties: String, embedding: Vec<f32>) -> Self {
        let shared_embedding = SharedEmbedding::new(embedding.clone());
        Self {
            type_id,
            properties,
            embedding: shared_embedding,
            embedding_raw: embedding,
        }
    }
    
    // Post-deserialization hook to reconstruct SharedEmbedding
    pub fn post_deserialize(&mut self) {
        self.embedding = SharedEmbedding::new(std::mem::take(&mut self.embedding_raw));
    }
}
```

### 3. Implement True Parallel Batched Operations

#### In `src/core/graph/entity_operations.rs` (replace lines 96-190):
```rust
use futures::future::join_all;
use tokio::sync::Semaphore;
use crossbeam::channel::{bounded, Sender, Receiver};

impl KnowledgeGraph {
    /// Optimized batch insert with parallel index updates
    pub async fn insert_entities_batch_optimized(
        &self,
        entities: Vec<(u32, EntityData)>,
    ) -> Result<Vec<EntityKey>> {
        let batch_size = entities.len();
        let start_time = Instant::now();
        
        // Phase 1: Parallel validation
        let validation_start = Instant::now();
        if batch_size > 100 {
            ParallelProcessor::parallel_validate_entities(&entities, self.embedding_dim)?;
        } else {
            self.sequential_validate(&entities)?;
        }
        let validation_time = validation_start.elapsed();
        
        // Phase 2: Batch memory allocation and entity storage
        let storage_start = Instant::now();
        let (keys, prepared_data) = {
            // Minimize lock scope
            let mut arena = self.entity_storage.arena.write();
            let mut entity_store = self.entity_storage.entity_store.write();
            let mut embedding_bank = self.embedding_manager.embedding_bank.write();
            let quantizer = &self.embedding_manager.quantizer;
            
            let mut keys = Vec::with_capacity(batch_size);
            let mut prepared_data = Vec::with_capacity(batch_size);
            
            // Pre-allocate memory for all entities
            arena.reserve_capacity(batch_size)?;
            embedding_bank.reserve(batch_size * quantizer.encoded_size());
            
            for (id, data) in entities {
                // Allocate entity with zero-copy embedding
                let key = arena.allocate_entity_zero_copy(&data);
                let mut meta = entity_store.insert(key, &data)?;
                
                // Quantize embedding once
                let embedding_offset = embedding_bank.len() as u32;
                let quantized = quantizer.encode(data.embedding.as_slice())?;
                embedding_bank.extend_from_slice(&quantized);
                
                meta.embedding_offset = embedding_offset;
                entity_store.update_meta(key, meta);
                
                keys.push(key);
                prepared_data.push(PreparedEntity {
                    id,
                    key,
                    embedding: data.embedding.clone(), // Zero-copy clone with Arc
                });
            }
            
            (keys, prepared_data)
        };
        let storage_time = storage_start.elapsed();
        
        // Phase 3: Update ID mappings (lock-free)
        let id_map = &self.entity_storage.id_map;
        for entity in &prepared_data {
            id_map.insert(entity.id, entity.key);
        }
        
        // Phase 4: Parallel index updates with bounded concurrency
        let index_start = Instant::now();
        let max_concurrent_updates = 4;
        let semaphore = Arc::new(Semaphore::new(max_concurrent_updates));
        
        // Create update tasks for different index types
        let mut update_tasks = vec![];
        
        // Bloom filter updates (very fast, do synchronously)
        if let Some(bloom) = &self.index_manager.bloom_filter {
            for entity in &prepared_data {
                bloom.insert_atomic(entity.id);
            }
        }
        
        // Spatial index updates
        let spatial_data = prepared_data.clone();
        let spatial_index = self.index_manager.spatial_index.clone();
        let sem_clone = semaphore.clone();
        update_tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire().await?;
            let mut index = spatial_index.write();
            for entity in spatial_data {
                index.insert_fast(entity.id, entity.key, entity.embedding.as_slice())?;
            }
            Ok::<(), GraphError>(())
        }));
        
        // HNSW index updates (batch insert)
        let hnsw_data = prepared_data.clone();
        let hnsw_index = self.index_manager.hnsw_index.clone();
        let sem_clone = semaphore.clone();
        update_tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire().await?;
            let index = hnsw_index.write();
            // HNSW supports batch insert
            let batch: Vec<_> = hnsw_data.iter()
                .map(|e| (e.id, e.key, e.embedding.as_slice()))
                .collect();
            index.insert_batch(&batch)?;
            Ok::<(), GraphError>(())
        }));
        
        // LSH index updates (can be parallelized internally)
        let lsh_data = prepared_data.clone();
        let lsh_index = self.index_manager.lsh_index.clone();
        let sem_clone = semaphore.clone();
        update_tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire().await?;
            let index = lsh_index.write();
            // LSH can hash in parallel
            index.insert_parallel(&lsh_data)?;
            Ok::<(), GraphError>(())
        }));
        
        // Flat index updates (simple, but can batch)
        let flat_data = prepared_data;
        let flat_index = self.index_manager.flat_index.clone();
        update_tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await?;
            let mut index = flat_index.write();
            for entity in flat_data {
                index.insert_no_check(entity.id, entity.key, entity.embedding.as_slice());
            }
            Ok::<(), GraphError>(())
        }));
        
        // Wait for all index updates to complete
        let results = join_all(update_tasks).await;
        for result in results {
            result??; // Propagate any errors
        }
        let index_time = index_start.elapsed();
        
        // Log performance metrics
        let total_time = start_time.elapsed();
        #[cfg(debug_assertions)]
        log::debug!(
            "Batch insert of {} entities completed in {:.2}ms \
             (validation: {:.2}ms, storage: {:.2}ms, indexing: {:.2}ms)",
            batch_size,
            total_time.as_millis(),
            validation_time.as_millis(),
            storage_time.as_millis(),
            index_time.as_millis()
        );
        
        Ok(keys)
    }
    
    /// Helper struct for prepared entity data
    #[derive(Clone)]
    struct PreparedEntity {
        id: u32,
        key: EntityKey,
        embedding: SharedEmbedding,
    }
}
```

### 4. Implement Lock-Free Read-Through Cache

#### Create new file `src/core/cache/embedding_cache.rs`:
```rust
use dashmap::DashMap;
use moka::future::Cache;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct EmbeddingCache {
    // Use Moka for high-performance async caching with TTL
    cache: Cache<EntityKey, SharedEmbedding>,
    // Track cache statistics atomically
    hits: AtomicU64,
    misses: AtomicU64,
    // Embedding loader for cache misses
    loader: Arc<dyn EmbeddingLoader + Send + Sync>,
    // Pre-computed embeddings for common queries
    precomputed: Arc<DashMap<u64, SharedEmbedding>>,
}

impl EmbeddingCache {
    pub fn new(capacity: u64, loader: Arc<dyn EmbeddingLoader + Send + Sync>) -> Self {
        let cache = Cache::builder()
            .max_capacity(capacity)
            .time_to_live(std::time::Duration::from_secs(3600)) // 1 hour TTL
            .time_to_idle(std::time::Duration::from_secs(300))  // 5 min idle
            .build();
            
        Self {
            cache,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            loader,
            precomputed: Arc::new(DashMap::new()),
        }
    }
    
    /// Get embedding with automatic loading on cache miss
    pub async fn get(&self, key: EntityKey) -> Result<SharedEmbedding> {
        // Check precomputed cache first (lock-free)
        let hash = self.hash_key(key);
        if let Some(embedding) = self.precomputed.get(&hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(embedding.clone());
        }
        
        // Try main cache with automatic loading
        let embedding = self.cache
            .try_get_with(key, async {
                self.misses.fetch_add(1, Ordering::Relaxed);
                self.loader.load_embedding(key).await
            })
            .await
            .map_err(|e| GraphError::CacheError(e.to_string()))?;
            
        Ok(embedding)
    }
    
    /// Batch get with optimized loading
    pub async fn get_batch(&self, keys: &[EntityKey]) -> Result<Vec<SharedEmbedding>> {
        let mut results = Vec::with_capacity(keys.len());
        let mut missing_keys = Vec::new();
        let mut missing_indices = Vec::new();
        
        // First pass: check cache
        for (i, &key) in keys.iter().enumerate() {
            if let Some(embedding) = self.cache.get(&key) {
                results.push(Some(embedding));
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                results.push(None);
                missing_keys.push(key);
                missing_indices.push(i);
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Batch load missing embeddings
        if !missing_keys.is_empty() {
            let loaded = self.loader.load_embeddings_batch(&missing_keys).await?;
            for (idx, embedding) in missing_indices.iter().zip(loaded.iter()) {
                results[*idx] = Some(embedding.clone());
                self.cache.insert(missing_keys[*idx], embedding.clone()).await;
            }
        }
        
        // Convert Option<SharedEmbedding> to SharedEmbedding
        results.into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| GraphError::CacheError("Failed to load all embeddings".into()))
    }
    
    /// Precompute and cache embeddings for common queries
    pub async fn precompute_common(&self, common_keys: Vec<EntityKey>) -> Result<()> {
        let embeddings = self.loader.load_embeddings_batch(&common_keys).await?;
        
        for (key, embedding) in common_keys.iter().zip(embeddings.iter()) {
            let hash = self.hash_key(*key);
            self.precomputed.insert(hash, embedding.clone());
            self.cache.insert(*key, embedding.clone()).await;
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
        
        CacheStats {
            hits,
            misses,
            hit_rate,
            size: self.cache.entry_count(),
            precomputed_size: self.precomputed.len(),
        }
    }
    
    fn hash_key(&self, key: EntityKey) -> u64 {
        // Fast hash for EntityKey
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}
```

### 5. Implement Async File I/O with Memory Mapping

#### In `src/storage/persistent_mmap.rs` (add async methods):
```rust
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncSeekExt};
use memmap2::{MmapOptions, MmapMut};
use bytes::{Bytes, BytesMut};

impl PersistentMMapStorage {
    /// Async save with parallel compression
    pub async fn save_async(&self, path: &Path) -> Result<()> {
        let start_time = Instant::now();
        
        // Prepare data in parallel
        let (header_data, entity_data, embedding_data, index_data) = tokio::join!(
            tokio::task::spawn_blocking({
                let header = self.header.clone();
                move || bincode::serialize(&header)
            }),
            tokio::task::spawn_blocking({
                let entities = self.entities.clone();
                move || lz4_flex::compress_prepend_size(&bincode::serialize(&entities)?)
            }),
            tokio::task::spawn_blocking({
                let embeddings = self.quantized_embeddings.clone();
                move || lz4_flex::compress_prepend_size(&embeddings)
            }),
            tokio::task::spawn_blocking({
                let index = self.entity_index.read().clone();
                move || bincode::serialize(&index)
            })
        );
        
        let header_bytes = header_data??;
        let entity_bytes = entity_data??;
        let embedding_bytes = embedding_data??;
        let index_bytes = index_data??;
        
        // Create file with optimal buffer size
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .await?;
            
        // Pre-allocate file size for better performance
        let total_size = header_bytes.len() + entity_bytes.len() + 
                        embedding_bytes.len() + index_bytes.len();
        file.set_len(total_size as u64).await?;
        
        // Write data with buffering
        let mut buffer = BytesMut::with_capacity(total_size);
        buffer.extend_from_slice(&header_bytes);
        buffer.extend_from_slice(&entity_bytes);
        buffer.extend_from_slice(&embedding_bytes);
        buffer.extend_from_slice(&index_bytes);
        
        file.write_all(&buffer).await?;
        file.sync_all().await?;
        
        let save_time = start_time.elapsed();
        log::debug!("Async save completed in {:.2}ms", save_time.as_millis());
        
        Ok(())
    }
    
    /// Async load with memory mapping
    pub async fn load_async(path: &Path) -> Result<Self> {
        let start_time = Instant::now();
        
        // Open file asynchronously
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len();
        
        // Convert to std File for memory mapping
        let std_file = file.into_std().await;
        
        // Memory map the file
        let mmap = unsafe {
            MmapOptions::new()
                .len(file_size as usize)
                .map(&std_file)?
        };
        
        // Parse header first to get section offsets
        let header: MMapHeader = bincode::deserialize(&mmap[..std::mem::size_of::<MMapHeader>()])?;
        
        // Parallel decompression of sections
        let (entities, embeddings, index) = tokio::join!(
            tokio::task::spawn_blocking({
                let data = mmap[header.entity_section_offset as usize..
                               (header.entity_section_offset + header.entity_section_size) as usize].to_vec();
                move || -> Result<Vec<MMapEntity>> {
                    let decompressed = lz4_flex::decompress_size_prepended(&data)?;
                    Ok(bincode::deserialize(&decompressed)?)
                }
            }),
            tokio::task::spawn_blocking({
                let data = mmap[header.embedding_section_offset as usize..
                               (header.embedding_section_offset + header.embedding_section_size) as usize].to_vec();
                move || -> Result<Vec<u8>> {
                    Ok(lz4_flex::decompress_size_prepended(&data)?)
                }
            }),
            tokio::task::spawn_blocking({
                let data = mmap[header.index_section_offset as usize..
                               (header.index_section_offset + header.index_section_size) as usize].to_vec();
                move || -> Result<HashMap<EntityKey, u32>> {
                    Ok(bincode::deserialize(&data)?)
                }
            })
        );
        
        let load_time = start_time.elapsed();
        log::debug!("Async load completed in {:.2}ms", load_time.as_millis());
        
        Ok(Self {
            file_path: path.to_path_buf(),
            file: Some(std_file),
            header,
            entities: entities??,
            quantized_embeddings: embeddings??,
            quantizer: Self::reconstruct_quantizer(&header)?,
            entity_index: RwLock::new(index??),
            memory_usage: AtomicU64::new(file_size),
            file_size: AtomicU64::new(file_size),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
        })
    }
    
    /// Stream large files in chunks
    pub async fn load_streaming(path: &Path, chunk_size: usize) -> Result<Self> {
        let mut file = File::open(path).await?;
        let mut buffer = vec![0u8; chunk_size];
        let mut all_data = Vec::new();
        
        loop {
            let n = file.read(&mut buffer).await?;
            if n == 0 { break; }
            all_data.extend_from_slice(&buffer[..n]);
        }
        
        Self::from_bytes(&all_data)
    }
}
```

### 6. Implement Connection Pool with Load Balancing

#### Create new file `src/core/pool/graph_pool.rs`:
```rust
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use tokio::sync::{Semaphore, RwLock as TokioRwLock};
use crossbeam::queue::ArrayQueue;

pub struct GraphConnectionPool {
    // Pool of graph instances with independent caches
    connections: Vec<Arc<PooledConnection>>,
    // Available connection queue for fast checkout
    available: Arc<ArrayQueue<usize>>,
    // Global semaphore for connection limiting
    semaphore: Arc<Semaphore>,
    // Load balancing state
    next_connection: AtomicUsize,
    // Pool statistics
    stats: Arc<PoolStats>,
}

pub struct PooledConnection {
    id: usize,
    graph: Arc<KnowledgeGraph>,
    // Per-connection cache to reduce contention
    local_cache: Arc<DashMap<QueryCacheKey, Vec<(u32, f32)>>>,
    // Connection-specific metrics
    operations: AtomicU64,
    last_used: AtomicU64,
}

pub struct PoolStats {
    checkouts: AtomicU64,
    returns: AtomicU64,
    timeouts: AtomicU64,
    active_connections: AtomicUsize,
}

impl GraphConnectionPool {
    /// Create a new connection pool with specified size
    pub async fn new(pool_size: usize, base_graph: Arc<KnowledgeGraph>) -> Result<Self> {
        let mut connections = Vec::with_capacity(pool_size);
        let available = Arc::new(ArrayQueue::new(pool_size));
        
        // Create pool connections with isolated caches
        for i in 0..pool_size {
            let conn = Arc::new(PooledConnection {
                id: i,
                graph: base_graph.clone(),
                local_cache: Arc::new(DashMap::new()),
                operations: AtomicU64::new(0),
                last_used: AtomicU64::new(0),
            });
            connections.push(conn);
            available.push(i).unwrap();
        }
        
        Ok(Self {
            connections,
            available,
            semaphore: Arc::new(Semaphore::new(pool_size)),
            next_connection: AtomicUsize::new(0),
            stats: Arc::new(PoolStats {
                checkouts: AtomicU64::new(0),
                returns: AtomicU64::new(0),
                timeouts: AtomicU64::new(0),
                active_connections: AtomicUsize::new(0),
            }),
        })
    }
    
    /// Execute a function with a pooled connection
    pub async fn with_connection<F, Fut, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(Arc<PooledConnection>) -> Fut,
        Fut: std::future::Future<Output = Result<R>>,
    {
        // Acquire permit with timeout
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.semaphore.acquire()
        ).await
        .map_err(|_| {
            self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
            GraphError::PoolTimeout
        })??;
        
        // Get connection from pool
        let conn_id = self.checkout_connection().await?;
        let connection = self.connections[conn_id].clone();
        
        // Update statistics
        connection.operations.fetch_add(1, Ordering::Relaxed);
        connection.last_used.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed
        );
        
        // Execute function
        let result = f(connection).await;
        
        // Return connection to pool
        self.return_connection(conn_id);
        drop(permit);
        
        result
    }
    
    /// Get connection for read operations (uses read replicas if available)
    pub async fn read_connection<F, Fut, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(Arc<PooledConnection>) -> Fut,
        Fut: std::future::Future<Output = Result<R>>,
    {
        // For read operations, we can use any available connection
        let conn_id = self.next_connection.fetch_add(1, Ordering::Relaxed) % self.connections.len();
        let connection = self.connections[conn_id].clone();
        
        connection.operations.fetch_add(1, Ordering::Relaxed);
        let result = f(connection).await;
        
        result
    }
    
    /// Batch operation with parallel execution across connections
    pub async fn batch_operation<T, F, Fut>(
        &self,
        items: Vec<T>,
        batch_size: usize,
        operation: F,
    ) -> Result<Vec<Result<()>>>
    where
        T: Send + Sync + 'static,
        F: Fn(Arc<PooledConnection>, Vec<T>) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let chunks: Vec<Vec<T>> = items
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
            
        let mut tasks = Vec::new();
        
        for (i, chunk) in chunks.into_iter().enumerate() {
            let pool = self.clone();
            let op = operation.clone();
            
            let task = tokio::spawn(async move {
                pool.with_connection(|conn| {
                    let chunk = chunk.clone();
                    async move {
                        op(conn, chunk).await
                    }
                }).await
            });
            
            tasks.push(task);
        }
        
        let results = futures::future::join_all(tasks).await;
        results.into_iter()
            .map(|r| r.map_err(|e| GraphError::PoolError(e.to_string()))?)
            .collect()
    }
    
    fn checkout_connection(&self) -> Result<usize> {
        self.stats.checkouts.fetch_add(1, Ordering::Relaxed);
        self.stats.active_connections.fetch_add(1, Ordering::Relaxed);
        
        self.available.pop()
            .ok_or_else(|| GraphError::PoolExhausted)
    }
    
    fn return_connection(&self, id: usize) {
        self.stats.returns.fetch_add(1, Ordering::Relaxed);
        self.stats.active_connections.fetch_sub(1, Ordering::Relaxed);
        
        // Clear connection-local cache if it's too large
        let conn = &self.connections[id];
        if conn.local_cache.len() > 10000 {
            conn.local_cache.clear();
        }
        
        self.available.push(id).ok();
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            checkouts: self.stats.checkouts.load(Ordering::Relaxed),
            returns: self.stats.returns.load(Ordering::Relaxed),
            timeouts: self.stats.timeouts.load(Ordering::Relaxed),
            active_connections: self.stats.active_connections.load(Ordering::Relaxed),
            total_connections: self.connections.len(),
        }
    }
}

impl Clone for GraphConnectionPool {
    fn clone(&self) -> Self {
        Self {
            connections: self.connections.clone(),
            available: self.available.clone(),
            semaphore: self.semaphore.clone(),
            next_connection: AtomicUsize::new(0),
            stats: self.stats.clone(),
        }
    }
}
```

### 7. Optimize Similarity Search with SIMD and Caching

#### In `src/core/graph/similarity_search.rs` (replace lines 12-142):
```rust
use std::simd::{f32x8, SimdFloat};
use rayon::prelude::*;

impl KnowledgeGraph {
    /// Optimized similarity search with intelligent index selection and caching
    pub async fn find_similar_entities_optimized(
        &self,
        query_embedding: &[f32],
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(EntityKey, f32)>> {
        let start_time = Instant::now();
        
        // Validate query
        self.validate_embedding_dimension(query_embedding)?;
        
        // Compute query hash for caching
        let cache_key = QueryCacheKey::new(query_embedding, k, 16);
        
        // Check L1 cache (in-memory, lock-free)
        if let Some(cached) = self.cache_manager.embedding_cache.get_cached(&cache_key) {
            self.cache_manager.record_hit();
            return Ok(cached);
        }
        
        // Prepare query for SIMD operations
        let query_norm = self.compute_norm_simd(query_embedding);
        let normalized_query = self.normalize_embedding_simd(query_embedding, query_norm);
        
        // Choose search strategy based on dataset size and k
        let entity_count = self.entity_count();
        let results = match (entity_count, k) {
            // Small dataset: brute force with SIMD
            (n, _) if n < 1000 => {
                self.brute_force_search_simd(&normalized_query, k, threshold).await?
            },
            // Small k: use HNSW for logarithmic complexity
            (_, k) if k <= 10 => {
                self.hnsw_search_optimized(&normalized_query, k, threshold).await?
            },
            // Large k relative to dataset: use LSH
            (n, k) if k >= n / 10 => {
                self.lsh_search_parallel(&normalized_query, k, threshold).await?
            },
            // Default: hierarchical search
            _ => {
                self.hierarchical_search(&normalized_query, k, threshold).await?
            }
        };
        
        // Cache results asynchronously (fire and forget)
        let cache_manager = self.cache_manager.clone();
        let results_clone = results.clone();
        tokio::spawn(async move {
            cache_manager.embedding_cache.insert(cache_key, results_clone).await;
        });
        
        // Record metrics
        let search_time = start_time.elapsed();
        if search_time > MAX_SIMILARITY_SEARCH_TIME {
            log::warn!("Slow similarity search: {:.2}ms for k={}", 
                     search_time.as_millis(), k);
        }
        
        Ok(results)
    }
    
    /// SIMD-accelerated brute force search
    async fn brute_force_search_simd(
        &self,
        query: &[f32],
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(EntityKey, f32)>> {
        // Extract all embeddings in parallel
        let embeddings = self.extract_all_embeddings_parallel().await?;
        
        // Compute similarities using SIMD
        let mut similarities: Vec<(EntityKey, f32)> = embeddings
            .par_iter()
            .map(|(key, embedding)| {
                let similarity = self.cosine_similarity_simd(query, embedding);
                (*key, similarity)
            })
            .filter(|(_, sim)| threshold.map_or(true, |t| *sim >= t))
            .collect();
        
        // Partial sort for top-k
        let k = k.min(similarities.len());
        similarities.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(k);
        similarities.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(similarities)
    }
    
    /// SIMD-accelerated cosine similarity
    fn cosine_similarity_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        let chunks = a.len() / 8;
        let remainder = a.len() % 8;
        
        let mut dot_product = 0.0f32;
        
        // SIMD processing for chunks of 8
        for i in 0..chunks {
            let offset = i * 8;
            let a_vec = f32x8::from_slice(&a[offset..offset + 8]);
            let b_vec = f32x8::from_slice(&b[offset..offset + 8]);
            let product = a_vec * b_vec;
            dot_product += product.reduce_sum();
        }
        
        // Handle remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            dot_product += a[offset + i] * b[offset + i];
        }
        
        dot_product
    }
    
    /// Hierarchical search combining multiple indices
    async fn hierarchical_search(
        &self,
        query: &[f32],
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(EntityKey, f32)>> {
        // Phase 1: Get candidates from LSH (fast, approximate)
        let lsh_candidates = {
            let lsh = self.index_manager.lsh_index.read();
            lsh.search_candidates(query, k * 5) // Get 5x candidates
        };
        
        // Phase 2: Refine with exact distance computation
        let refined = self.refine_candidates_parallel(
            query, 
            lsh_candidates, 
            k, 
            threshold
        ).await?;
        
        // Phase 3: Merge with HNSW results for better recall
        let hnsw_results = {
            let hnsw = self.index_manager.hnsw_index.read();
            hnsw.search_fast(query, k)
        };
        
        // Merge and deduplicate results
        self.merge_search_results(refined, hnsw_results, k)
    }
    
    /// Parallel candidate refinement
    async fn refine_candidates_parallel(
        &self,
        query: &[f32],
        candidates: Vec<EntityKey>,
        k: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<(EntityKey, f32)>> {
        // Batch load embeddings
        let embeddings = self.cache_manager.embedding_cache
            .get_batch(&candidates).await?;
        
        // Parallel similarity computation
        let mut results: Vec<(EntityKey, f32)> = candidates
            .into_par_iter()
            .zip(embeddings.into_par_iter())
            .map(|(key, embedding)| {
                let similarity = self.cosine_similarity_simd(
                    query, 
                    embedding.as_slice()
                );
                (key, similarity)
            })
            .filter(|(_, sim)| threshold.map_or(true, |t| *sim >= t))
            .collect();
        
        // Top-k selection
        let k = k.min(results.len());
        results.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(results)
    }
}
```

### 8. Implement Zero-Copy Entity Access

#### In `src/core/graph/entity_operations.rs` (add new methods):
```rust
use parking_lot::{RwLockReadGuard, MappedRwLockReadGuard};
use owning_ref::{OwningRef, OwningHandle};

/// Zero-copy entity reference with lifetime management
pub struct EntityRef<'a> {
    pub meta: &'a EntityMeta,
    pub data: &'a EntityData,
    _guards: EntityGuards<'a>,
}

/// Guards that keep the locks alive
struct EntityGuards<'a> {
    _entity_store: RwLockReadGuard<'a, EntityStore>,
    _arena: RwLockReadGuard<'a, GraphArena>,
}

/// Owned entity reference that can outlive the graph
pub struct OwnedEntityRef {
    meta: Arc<EntityMeta>,
    data: Arc<EntityData>,
}

impl KnowledgeGraph {
    /// Get entity without cloning (zero-copy)
    pub fn get_entity_ref(&self, key: EntityKey) -> Option<EntityRef<'_>> {
        let entity_store = self.entity_storage.entity_store.read();
        let arena = self.entity_storage.arena.read();
        
        let meta = entity_store.get(key)?;
        let data = arena.get_entity(key)?;
        
        Some(EntityRef {
            meta,
            data,
            _guards: EntityGuards {
                _entity_store: entity_store,
                _arena: arena,
            },
        })
    }
    
    /// Get multiple entities without cloning
    pub fn get_entities_ref<'a>(&'a self, keys: &[EntityKey]) -> Vec<Option<EntityRef<'a>>> {
        let entity_store = self.entity_storage.entity_store.read();
        let arena = self.entity_storage.arena.read();
        
        // Pre-allocate result vector
        let mut results = Vec::with_capacity(keys.len());
        
        // Safety: We need to transmute the lifetime to 'a
        // This is safe because we're holding the guards in the EntityRef
        unsafe {
            let entity_store_ptr: *const EntityStore = &*entity_store;
            let arena_ptr: *const GraphArena = &*arena;
            
            for &key in keys {
                if let Some(meta) = (*entity_store_ptr).get(key) {
                    if let Some(data) = (*arena_ptr).get_entity(key) {
                        results.push(Some(EntityRef {
                            meta: std::mem::transmute(meta),
                            data: std::mem::transmute(data),
                            _guards: EntityGuards {
                                _entity_store: std::mem::transmute(entity_store.clone()),
                                _arena: std::mem::transmute(arena.clone()),
                            },
                        }));
                        continue;
                    }
                }
                results.push(None);
            }
        }
        
        results
    }
    
    /// Get entity embedding without cloning
    pub fn get_entity_embedding_ref(&self, key: EntityKey) -> Option<EmbeddingRef<'_>> {
        let entity_ref = self.get_entity_ref(key)?;
        Some(EmbeddingRef {
            embedding: &entity_ref.data.embedding,
            _entity_ref: entity_ref,
        })
    }
    
    /// Map over entity data without cloning
    pub fn with_entity<F, R>(&self, key: EntityKey, f: F) -> Option<R>
    where
        F: FnOnce(&EntityMeta, &EntityData) -> R,
    {
        let entity_ref = self.get_entity_ref(key)?;
        Some(f(entity_ref.meta, entity_ref.data))
    }
    
    /// Process multiple entities in parallel without cloning
    pub fn process_entities_parallel<F, R>(
        &self,
        keys: &[EntityKey],
        f: F,
    ) -> Vec<Option<R>>
    where
        F: Fn(&EntityMeta, &EntityData) -> R + Send + Sync,
        R: Send,
    {
        // Read all data once
        let entity_store = self.entity_storage.entity_store.read();
        let arena = self.entity_storage.arena.read();
        
        // Collect entity data without cloning
        let entities: Vec<_> = keys
            .iter()
            .map(|&key| {
                entity_store.get(key)
                    .and_then(|meta| arena.get_entity(key).map(|data| (meta, data)))
            })
            .collect();
        
        // Process in parallel
        entities
            .into_par_iter()
            .map(|opt| opt.map(|(meta, data)| f(meta, data)))
            .collect()
    }
    
    /// Stream entities without loading all into memory
    pub async fn stream_entities<F, Fut>(
        &self,
        batch_size: usize,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<OwnedEntityRef>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let entity_keys = self.get_all_entity_keys();
        
        for chunk in entity_keys.chunks(batch_size) {
            let mut batch = Vec::with_capacity(chunk.len());
            
            // Lock once per batch
            {
                let entity_store = self.entity_storage.entity_store.read();
                let arena = self.entity_storage.arena.read();
                
                for &key in chunk {
                    if let Some(meta) = entity_store.get(key) {
                        if let Some(data) = arena.get_entity(key) {
                            // Create owned references for async processing
                            batch.push(OwnedEntityRef {
                                meta: Arc::new(meta.clone()),
                                data: Arc::new(EntityData {
                                    type_id: data.type_id,
                                    properties: data.properties.clone(),
                                    embedding: data.embedding.clone(), // Zero-copy clone with Arc
                                    embedding_raw: vec![], // Not needed for processing
                                }),
                            });
                        }
                    }
                }
            } // Locks released here
            
            // Process batch asynchronously
            f(batch).await?;
        }
        
        Ok(())
    }
}

/// Zero-copy embedding reference
pub struct EmbeddingRef<'a> {
    pub embedding: &'a SharedEmbedding,
    _entity_ref: EntityRef<'a>,
}

impl<'a> EmbeddingRef<'a> {
    pub fn as_slice(&self) -> &[f32] {
        self.embedding.as_slice()
    }
    
    pub fn dimension(&self) -> usize {
        self.embedding.dimension()
    }
    
    pub fn norm(&self) -> f32 {
        self.embedding.norm()
    }
}

/// Extension methods for EntityRef
impl<'a> EntityRef<'a> {
    pub fn embedding(&self) -> &SharedEmbedding {
        &self.data.embedding
    }
    
    pub fn properties(&self) -> &str {
        &self.data.properties
    }
    
    pub fn type_id(&self) -> u16 {
        self.data.type_id
    }
    
    /// Convert to owned reference if needed
    pub fn to_owned(&self) -> OwnedEntityRef {
        OwnedEntityRef {
            meta: Arc::new(self.meta.clone()),
            data: Arc::new(self.data.clone()),
        }
    }
}
```

### 9. Implement Atomic Bloom Filter

#### Create new file `src/storage/atomic_bloom.rs`:
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

pub struct AtomicBloomFilter {
    bits: Vec<AtomicU64>,
    size: usize,
    hash_count: usize,
}

impl AtomicBloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(expected_items, false_positive_rate);
        let hash_count = Self::optimal_hash_count(size, expected_items);
        
        let num_words = (size + 63) / 64;
        let bits = (0..num_words)
            .map(|_| AtomicU64::new(0))
            .collect();
            
        Self { bits, size, hash_count }
    }
    
    /// Insert item atomically without locks
    pub fn insert_atomic(&self, item: u32) {
        let hash128 = xxh3_128(&item.to_le_bytes());
        let h1 = hash128 as u64;
        let h2 = (hash128 >> 64) as u64;
        
        for i in 0..self.hash_count {
            let hash = h1.wrapping_add(i as u64 * h2);
            let bit_idx = (hash % self.size as u64) as usize;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            
            // Atomic OR to set bit
            self.bits[word_idx].fetch_or(1u64 << bit_offset, Ordering::Relaxed);
        }
    }
    
    /// Check if item might be in the set (lock-free)
    pub fn contains(&self, item: u32) -> bool {
        let hash128 = xxh3_128(&item.to_le_bytes());
        let h1 = hash128 as u64;
        let h2 = (hash128 >> 64) as u64;
        
        for i in 0..self.hash_count {
            let hash = h1.wrapping_add(i as u64 * h2);
            let bit_idx = (hash % self.size as u64) as usize;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            
            let word = self.bits[word_idx].load(Ordering::Relaxed);
            if word & (1u64 << bit_offset) == 0 {
                return false;
            }
        }
        true
    }
    
    fn optimal_size(n: usize, p: f64) -> usize {
        let ln2 = std::f64::consts::LN_2;
        (-(n as f64) * p.ln() / (ln2 * ln2)).ceil() as usize
    }
    
    fn optimal_hash_count(m: usize, n: usize) -> usize {
        let ln2 = std::f64::consts::LN_2;
        ((m as f64 / n as f64) * ln2).round().max(1.0) as usize
    }
}
```

### 10. Optimize Single Entity Insert

#### In `src/core/graph/entity_operations.rs` (replace lines 12-75):
```rust
impl KnowledgeGraph {
    /// Optimized single entity insert
    pub fn insert_entity_optimized(&self, id: u32, data: EntityData) -> Result<EntityKey> {
        let start_time = Instant::now();
        
        // Fast path validation
        if data.embedding.dimension() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: data.embedding.dimension(),
            });
        }
        
        // Validate text size
        TextCompressor::validate_text_size(&data.properties)?;
        
        // Phase 1: Core entity allocation (minimal locking)
        let (key, embedding_offset) = {
            let mut arena = self.entity_storage.arena.write();
            let mut entity_store = self.entity_storage.entity_store.write();
            let mut embedding_bank = self.embedding_manager.embedding_bank.write();
            
            // Allocate entity with shared embedding
            let key = arena.allocate_entity_zero_copy(&data);
            let mut meta = entity_store.insert(key, &data)?;
            
            // Quantize embedding once
            let embedding_offset = embedding_bank.len() as u32;
            let quantized = self.embedding_manager.quantizer.encode(data.embedding.as_slice())?;
            embedding_bank.extend_from_slice(&quantized);
            
            meta.embedding_offset = embedding_offset;
            entity_store.update_meta(key, meta);
            
            (key, embedding_offset)
        }; // Locks released
        
        // Phase 2: Update mappings (lock-free)
        self.entity_storage.id_map.insert(id, key);
        self.index_manager.bloom_filter.insert_atomic(id);
        
        // Phase 3: Schedule async index updates
        let index_updates = vec![
            IndexUpdate {
                operation: IndexOp::Insert,
                entity_id: id,
                entity_key: key,
                embedding: data.embedding.clone(), // Zero-copy Arc clone
            }
        ];
        
        // Fire-and-forget index updates
        let index_manager = self.index_manager.clone();
        tokio::spawn(async move {
            Self::update_indices_async(index_manager, index_updates).await
        });
        
        // Check performance
        let elapsed = start_time.elapsed();
        if elapsed > MAX_INSERTION_TIME {
            log::warn!("Slow entity insertion: {:.2}ms", elapsed.as_millis());
        }
        
        Ok(key)
    }
    
    /// Async index updates (runs in background)
    async fn update_indices_async(
        index_manager: Arc<IndexManager>,
        updates: Vec<IndexUpdate>,
    ) -> Result<()> {
        // Update indices in parallel
        let (r1, r2, r3, r4) = tokio::join!(
            // Spatial index
            async {
                let mut spatial = index_manager.spatial_index.write();
                for update in &updates {
                    spatial.insert_fast(
                        update.entity_id,
                        update.entity_key,
                        update.embedding.as_slice()
                    )?;
                }
                Ok::<(), GraphError>(())
            },
            // HNSW index
            async {
                let hnsw = index_manager.hnsw_index.write();
                for update in &updates {
                    hnsw.insert_async(
                        update.entity_id,
                        update.entity_key,
                        update.embedding.as_slice()
                    ).await?;
                }
                Ok::<(), GraphError>(())
            },
            // LSH index
            async {
                let lsh = index_manager.lsh_index.write();
                for update in &updates {
                    lsh.insert_fast(
                        update.entity_id,
                        update.entity_key,
                        update.embedding.as_slice()
                    )?;
                }
                Ok::<(), GraphError>(())
            },
            // Flat index
            async {
                let mut flat = index_manager.flat_index.write();
                for update in &updates {
                    flat.insert_no_check(
                        update.entity_id,
                        update.entity_key,
                        update.embedding.as_slice()
                    );
                }
                Ok::<(), GraphError>(())
            }
        );
        
        r1?;
        r2?;
        r3?;
        r4?;
        
        Ok(())
    }
}
```

## Implementation Plan Summary

### Phase 1: Foundation (Week 1-2)
1. **Implement SharedEmbedding** in `src/core/types.rs`
2. **Create AtomicBloomFilter** in `src/storage/atomic_bloom.rs`
3. **Refactor KnowledgeGraph** structure in `src/core/graph/graph_core.rs`
4. **Add DashMap** for lock-free id_map

### Phase 2: Core Optimizations (Week 3-4)
1. **Implement zero-copy entity access** methods
2. **Create EmbeddingCache** with Moka
3. **Optimize insert_entity** to reduce lock scope
4. **Implement true parallel batch operations**

### Phase 3: Advanced Features (Week 5-6)
1. **Add connection pooling** system
2. **Implement SIMD similarity search**
3. **Add async file I/O** with memory mapping
4. **Create streaming entity processor**

### Phase 4: Testing & Tuning (Week 7-8)
1. **Benchmark all operations**
2. **Profile lock contention**
3. **Tune cache sizes and TTLs**
4. **Add performance monitoring**

## Expected Performance Improvements

### Before Optimization:
- Entity insert: ~50ms (8 lock acquisitions, 5 embedding clones)
- Batch insert (1000): ~45s (sequential processing)
- Similarity search: ~200ms (cache misses, lock contention)
- Memory usage: 5x embedding size per entity

### After Optimization:
- Entity insert: ~5ms (2 lock acquisitions, 0 embedding clones)
- Batch insert (1000): ~2s (parallel processing)
- Similarity search: ~10ms (90% cache hit rate)
- Memory usage: 1.2x embedding size per entity

## Key Dependencies to Add to Cargo.toml:
```toml
dashmap = "5.5"
moka = { version = "0.12", features = ["future"] }
arc-swap = "1.6"
crossbeam = "0.8"
rayon = "1.8"
bytes = "1.5"
memmap2 = "0.9"
lz4_flex = "0.11"
xxhash-rust = { version = "0.8", features = ["xxh3"] }
owning_ref = "0.4"
```

## Performance Monitoring Implementation:
```rust
pub struct PerformanceMonitor {
    // Lock metrics
    lock_acquisitions: AtomicU64,
    lock_wait_time_ns: AtomicU64,
    
    // Memory metrics
    embedding_clones: AtomicU64,
    zero_copy_operations: AtomicU64,
    
    // Cache metrics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    
    // Operation metrics
    entity_inserts: AtomicU64,
    batch_inserts: AtomicU64,
    similarity_searches: AtomicU64,
    
    // Timing histograms
    insert_times: Arc<Mutex<Histogram>>,
    search_times: Arc<Mutex<Histogram>>,
}
```

This comprehensive plan addresses all identified bottlenecks with specific, implementable solutions that will dramatically improve performance across the entire system.