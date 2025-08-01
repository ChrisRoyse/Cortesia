# Multi-Level Intelligent Caching System

## Overview

The Enhanced Knowledge Storage System includes a sophisticated multi-level caching system designed to optimize performance for document processing, query results, and model inference operations. The caching system implements a three-tier hierarchy with intelligent eviction, compression, and adaptive TTL management.

## Architecture

### Cache Hierarchy

```
┌─────────────────┐
│   L1 Memory     │  ← Hot data, fastest access
│   (LRU Cache)   │
└─────────────────┘
         │
┌─────────────────┐
│   L2 Disk       │  ← Warm data, compressed storage
│  (Compressed)   │
└─────────────────┘
         │
┌─────────────────┐
│  L3 Distributed │  ← Cold data, Redis/Memcached
│ (Redis/Memcached)│
└─────────────────┘
```

### Components

1. **L1 Memory Cache**: In-memory LRU cache for hot data
2. **L2 Disk Cache**: Compressed disk-based storage for warm data
3. **L3 Distributed Cache**: Optional Redis/Memcached integration
4. **Cache Statistics**: Comprehensive monitoring and metrics
5. **Intelligent Invalidation**: Pattern-based and TTL-based cleanup

## Key Features

### 1. Multi-Level Hierarchy

- **L1 (Memory)**: Ultra-fast access with configurable LRU eviction
- **L2 (Disk)**: Persistent storage with compression (gzip)
- **L3 (Distributed)**: Scalable distributed caching support

### 2. Write Strategies

```rust
pub enum WriteStrategy {
    WriteThrough,                              // Synchronous writes to all levels
    WriteBack,                                 // Asynchronous background writes
    WriteBehind { delay: Duration },           // Delayed writes with configurable delay
}
```

### 3. Intelligent Features

- **Adaptive TTL**: TTL adjusts based on access patterns
- **Cache Stampede Prevention**: Semaphore-based protection
- **Compression**: Automatic compression for L2 storage
- **Pattern Invalidation**: Regex-based cache invalidation
- **Cache Warming**: Preload frequently accessed data

### 4. Statistics and Monitoring

```rust
pub struct CacheStatistics {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub compression_savings: u64,
    pub total_requests: u64,
    pub stampede_preventions: u64,
    // ... more metrics
}
```

## Usage Examples

### Basic Setup

```rust
use llmkg::enhanced_knowledge_storage::production::caching::*;

// Create a multi-level cache
let cache = CacheConfigBuilder::new()
    .l1_capacity(10000)                        // 10K entries in L1
    .l1_max_bytes(100 * 1024 * 1024)           // 100MB L1 limit
    .l2_cache_dir("./cache")                   // L2 directory
    .l2_max_bytes(1024 * 1024 * 1024)          // 1GB L2 limit
    .write_strategy(WriteStrategy::WriteThrough)
    .build()
    .await?;
```

### Storing and Retrieving Data

```rust
// Store data with optional TTL
cache.put("user:123".to_string(), user_data, Some(Duration::from_secs(3600))).await;

// Retrieve data (type-safe)
let user_data: Option<UserProfile> = cache.get("user:123").await;
```

### Cache Warming

```rust
// Preload frequently accessed data
let warmup_data = vec![
    ("doc:1".to_string(), document1_bytes),
    ("doc:2".to_string(), document2_bytes),
];
cache.warm_cache(warmup_data).await;
```

### Pattern-Based Invalidation

```rust
// Invalidate all user session data
let invalidated = cache.invalidate_pattern(r"^session:").await?;
println!("Invalidated {} cache entries", invalidated);
```

## Configuration Options

### Cache Builder Configuration

```rust
CacheConfigBuilder::new()
    .l1_capacity(usize)                        // L1 entry limit
    .l1_max_bytes(usize)                       // L1 memory limit
    .l2_cache_dir<P: AsRef<Path>>(P)          // L2 directory
    .l2_max_bytes(usize)                       // L2 disk limit
    .l3_cache(Arc<dyn L3DistributedCache>)    // L3 implementation
    .write_strategy(WriteStrategy)             // Write behavior
    .compression_level(u32)                    // Gzip compression level (0-9)
    .build()
```

### Write Strategy Details

1. **WriteThrough**: 
   - ✅ Data consistency guaranteed
   - ❌ Higher latency for writes
   - 🎯 Use for: Critical data that must be immediately persistent

2. **WriteBack**:
   - ✅ Lower write latency
   - ❌ Risk of data loss on failure
   - 🎯 Use for: High-throughput scenarios with acceptable risk

3. **WriteBehind**:
   - ✅ Configurable write delay
   - ⚖️ Balance between performance and consistency
   - 🎯 Use for: Batch processing scenarios

## Performance Optimizations

### 1. Cache Sizing

```rust
// Recommended L1 sizing
let available_memory = get_available_memory();
let l1_size = available_memory / 10;  // Use 10% of available memory

// L2 sizing based on storage
let l2_size = available_disk_space / 20;  // Use 5% of available disk
```

### 2. Compression Configuration

```rust
// Balance compression ratio vs CPU usage
let compression_level = match workload_type {
    WorkloadType::CpuBound => 1,        // Fast compression
    WorkloadType::IoBound => 6,         // Balanced
    WorkloadType::StorageBound => 9,    // Maximum compression
};
```

### 3. Adaptive TTL

The cache automatically adjusts TTL based on access patterns:

```rust
// Formula: adaptive_ttl = base_ttl * (1 + ln(access_count) * 0.1)
let adaptive_ttl = cache.calculate_adaptive_ttl(access_count, base_ttl);
```

## Cache Use Cases in Enhanced Knowledge Storage

### 1. Document Processing Results

```rust
// Cache processed document chunks
let doc_id = "doc_12345";
let processed_chunks = process_document(&raw_document).await?;
cache.put(
    format!("processed_chunks:{}", doc_id),
    processed_chunks,
    Some(Duration::from_secs(86400)) // 24 hours
).await;
```

### 2. Query Results with Reasoning Chains

```rust
// Cache complex reasoning results
let query_hash = calculate_query_hash(&query);
let reasoning_result = perform_reasoning(&query).await?;
cache.put(
    format!("reasoning:{}", query_hash),
    reasoning_result,
    Some(Duration::from_secs(3600)) // 1 hour
).await;
```

### 3. Model Inference Results

```rust
// Cache embedding computations
let text_hash = calculate_text_hash(&text);
let embeddings = compute_embeddings(&text).await?;
cache.put(
    format!("embeddings:{}", text_hash),
    embeddings,
    Some(Duration::from_secs(7200)) // 2 hours
).await;
```

### 4. Semantic Chunking Results

```rust
// Cache semantic chunks for reuse
let content_hash = calculate_content_hash(&content);
let semantic_chunks = semantic_chunker.chunk(&content).await?;
cache.put(
    format!("semantic_chunks:{}", content_hash),
    semantic_chunks,
    Some(Duration::from_secs(1800)) // 30 minutes
).await;
```

## Monitoring and Debugging

### Cache Statistics

```rust
let stats = cache.get_statistics().await;
println!("Cache Performance:");
println!("  Hit Rate: {:.2}%", stats.hit_rate() * 100.0);
println!("  L1: {} entries, {} bytes", stats.l1_entry_count, stats.l1_size_bytes);
println!("  L2: {} entries, {} bytes", stats.l2_entry_count, stats.l2_size_bytes);
println!("  Compression Ratio: {:.2}", stats.compression_ratio());
```

### Cache Health Monitoring

```rust
// Recommended health checks
fn check_cache_health(stats: &CacheStatistics) -> CacheHealth {
    let hit_rate = stats.hit_rate();
    let l1_utilization = stats.l1_size_bytes as f64 / max_l1_bytes as f64;
    
    match (hit_rate, l1_utilization) {
        (hr, _) if hr < 0.5 => CacheHealth::Poor,
        (hr, util) if hr > 0.8 && util < 0.9 => CacheHealth::Excellent,
        _ => CacheHealth::Good,
    }
}
```

## Integration with Production System

### System Architecture Integration

```rust
pub struct ProductionSystem {
    cache: Arc<MultiLevelCache>,
    document_processor: DocumentProcessor,
    reasoning_engine: ReasoningEngine,
    // ... other components
}

impl ProductionSystem {
    pub async fn process_document_with_cache(&self, document: Document) -> Result<ProcessedDocument> {
        let cache_key = format!("processed_doc:{}", document.id);
        
        // Try cache first
        if let Some(cached) = self.cache.get(&cache_key).await {
            return Ok(cached);
        }
        
        // Process and cache result
        let processed = self.document_processor.process(document).await?;
        self.cache.put(cache_key, processed.clone(), Some(Duration::from_secs(3600))).await;
        
        Ok(processed)
    }
}
```

## Best Practices

### 1. Key Design

```rust
// Use hierarchical keys for better invalidation
let key = format!("{}:{}:{}", namespace, entity_type, entity_id);

// Examples:
// "user:profile:123"
// "document:processed:456" 
// "query:reasoning:789abc"
```

### 2. TTL Strategy

```rust
// Different TTL for different data types
let ttl = match data_type {
    DataType::UserSession => Duration::from_secs(1800),      // 30 minutes
    DataType::ProcessedDocument => Duration::from_secs(86400), // 24 hours
    DataType::ModelInference => Duration::from_secs(3600),    // 1 hour
    DataType::StaticReference => Duration::from_secs(604800), // 7 days
};
```

### 3. Error Handling

```rust
// Graceful cache failures
async fn get_with_fallback<T>(&self, key: &str) -> Result<T> 
where 
    T: DeserializeOwned + Send + Sync,
{
    match self.cache.get(key).await {
        Some(data) => Ok(data),
        None => {
            warn!("Cache miss for key: {}", key);
            self.compute_and_cache(key).await
        }
    }
}
```

### 4. Cache Partitioning

```rust
// Partition by data type for better management
let cache_key = match data_type {
    DataType::UserData => format!("user:{}", key),
    DataType::DocumentData => format!("doc:{}", key),
    DataType::QueryResults => format!("query:{}", key),
};
```

## Performance Benchmarks

### Typical Performance Characteristics

| Operation | L1 Cache | L2 Cache | L3 Cache |
|-----------|----------|----------|----------|
| Get (hit) | < 1μs | < 100μs | < 1ms |
| Put | < 10μs | < 1ms | < 5ms |
| Pattern Invalidation | < 1ms | < 10ms | < 50ms |

### Memory Usage

- **L1**: ~8 bytes overhead per entry + data size
- **L2**: ~40% compression ratio for typical text data
- **Metadata**: ~200 bytes per cache entry

## Troubleshooting

### Common Issues

1. **High Cache Miss Rate**
   - Check TTL settings
   - Verify key consistency
   - Monitor invalidation patterns

2. **L2 Disk Usage Growing**
   - Verify L2 size limits
   - Check compression settings
   - Monitor eviction policies

3. **Memory Pressure**
   - Adjust L1 capacity
   - Implement custom eviction policies
   - Monitor object sizes

### Debug Tools

```rust
// Enable detailed logging
env_logger::init_from_env(env_logger::Env::default().default_filter_or("debug"));

// Cache introspection
let stats = cache.get_statistics().await;
println!("Cache Debug Info: {:#?}", stats);
```

## Future Enhancements

### Planned Features

1. **Intelligent Prefetching**: ML-based cache warming
2. **Cross-Node Consistency**: Distributed cache synchronization  
3. **Custom Eviction Policies**: Pluggable eviction strategies
4. **Cache Analytics**: Advanced usage pattern analysis
5. **Automatic Scaling**: Dynamic cache size adjustment

### Extension Points

```rust
// Custom L3 implementation
#[async_trait::async_trait]
impl L3DistributedCache for CustomDistributedCache {
    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        // Custom implementation
    }
    // ... other methods
}
```

This caching system provides the foundation for high-performance knowledge storage and retrieval in the Enhanced Knowledge Storage System, with intelligent optimization and production-ready reliability.