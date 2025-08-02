# Micro Phase 2.5: Exception Storage Optimization

**Estimated Time**: 30 minutes
**Dependencies**: Micro 2.4 (Exception Pattern Learning)
**Objective**: Implement optimized storage structures and indexing for efficient exception management

## Task Description

Create high-performance storage optimization layer that minimizes memory usage and maximizes access speed for exceptions. This includes compression techniques, intelligent indexing, and memory pool management to handle large-scale exception data efficiently.

The optimizer focuses on reducing memory fragmentation, improving cache locality, and providing fast bulk operations for exception management.

## Deliverables

Create `src/exceptions/storage_optimizer.rs` with:

1. **StorageOptimizer struct**: Main optimization controller
2. **Compression algorithms**: Reduce memory footprint of exception data
3. **Index optimization**: Efficient multi-dimensional indexing
4. **Memory pools**: Reduce allocation overhead
5. **Bulk operations**: Optimized batch processing for exceptions

## Success Criteria

- [ ] Memory usage reduced by >50% compared to naive storage
- [ ] Index lookups perform in O(log n) or better
- [ ] Bulk operations process >10,000 exceptions/second
- [ ] Memory fragmentation minimized through pooling
- [ ] Compression ratio >3:1 for typical exception data
- [ ] Cache hit rate >95% for hot exception data

## Implementation Requirements

```rust
pub struct StorageOptimizer {
    compressed_store: CompressedExceptionStore,
    index_manager: IndexManager,
    memory_pool: MemoryPool,
    cache: LruCache<ExceptionKey, CachedData>,
    compression_stats: CompressionStatistics,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ExceptionKey {
    node_id: NodeId,
    property_hash: u64, // Hash of property name for space efficiency
}

pub struct CompressedExceptionStore {
    compressed_blocks: Vec<CompressedBlock>,
    block_index: BTreeMap<ExceptionKey, BlockAddress>,
    compression_algorithm: CompressionAlgorithm,
}

#[derive(Debug, Clone)]
pub struct CompressedBlock {
    compressed_data: Vec<u8>,
    original_size: usize,
    exception_count: u32,
    checksum: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockAddress {
    block_index: u32,
    offset: u32,
    length: u32,
}

pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Custom(Box<dyn CustomCompressor>),
}

pub trait CustomCompressor: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    fn compression_ratio(&self) -> f32;
}

pub struct IndexManager {
    primary_index: BTreeMap<ExceptionKey, BlockAddress>,
    property_index: HashMap<u64, Vec<NodeId>>, // property_hash -> nodes
    source_index: HashMap<ExceptionSource, Vec<ExceptionKey>>,
    confidence_index: SkipList<OrderedFloat<f32>, Vec<ExceptionKey>>,
    temporal_index: BTreeMap<SystemTime, Vec<ExceptionKey>>,
}

pub struct MemoryPool {
    exception_pool: Pool<Exception>,
    string_pool: Pool<String>,
    vector_pool: Pool<Vec<u8>>,
    allocation_stats: AllocationStatistics,
}

impl StorageOptimizer {
    pub fn new(initial_capacity: usize) -> Self;
    
    pub fn store_exception(&mut self, key: ExceptionKey, exception: Exception) -> Result<(), StorageError>;
    
    pub fn retrieve_exception(&self, key: &ExceptionKey) -> Result<Option<Exception>, StorageError>;
    
    pub fn bulk_store(&mut self, exceptions: Vec<(ExceptionKey, Exception)>) -> Result<usize, StorageError>;
    
    pub fn bulk_retrieve(&self, keys: &[ExceptionKey]) -> Result<Vec<Option<Exception>>, StorageError>;
    
    pub fn optimize_storage(&mut self) -> OptimizationResults;
    
    pub fn rebuild_indexes(&mut self) -> Result<(), StorageError>;
    
    pub fn compact_storage(&mut self) -> CompactionResults;
    
    pub fn get_storage_statistics(&self) -> StorageStatistics;
    
    pub fn query_by_property(&self, property_hash: u64) -> Vec<(NodeId, Exception)>;
    
    pub fn query_by_confidence_range(&self, min: f32, max: f32) -> Vec<(ExceptionKey, Exception)>;
    
    pub fn query_by_time_range(&self, start: SystemTime, end: SystemTime) -> Vec<(ExceptionKey, Exception)>;
}

#[derive(Debug)]
pub struct OptimizationResults {
    pub memory_saved: usize,
    pub compression_ratio: f32,
    pub index_rebuild_time: Duration,
    pub defragmentation_savings: usize,
}

#[derive(Debug)]
pub struct CompactionResults {
    pub blocks_compacted: usize,
    pub space_reclaimed: usize,
    pub fragmentation_reduced: f32,
    pub compaction_time: Duration,
}

#[derive(Debug, Default)]
pub struct StorageStatistics {
    pub total_exceptions: AtomicUsize,
    pub total_compressed_size: AtomicUsize,
    pub total_uncompressed_size: AtomicUsize,
    pub compression_ratio: AtomicU32, // Fixed-point representation
    pub index_memory_usage: AtomicUsize,
    pub cache_hit_rate: AtomicU32, // Percentage * 100
    pub average_access_time_nanos: AtomicU64,
}

#[derive(Debug, Default)]
pub struct CompressionStatistics {
    pub total_compressions: AtomicU64,
    pub total_decompressions: AtomicU64,
    pub bytes_compressed: AtomicU64,
    pub bytes_decompressed: AtomicU64,
    pub compression_time: AtomicU64, // Total nanoseconds
    pub decompression_time: AtomicU64,
}

#[derive(Debug, Default)]
pub struct AllocationStatistics {
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
    pub allocations_avoided: AtomicU64,
    pub memory_reused: AtomicUsize,
}
```

## Test Requirements

Must pass storage optimization tests:
```rust
#[test]
fn test_compression_efficiency() {
    let mut optimizer = StorageOptimizer::new(1000);
    
    // Create similar exceptions that should compress well
    let base_exception = Exception {
        inherited_value: PropertyValue::String("default_value".to_string()),
        actual_value: PropertyValue::String("override_value".to_string()),
        reason: "Standard override pattern".to_string(),
        source: ExceptionSource::Detected,
        created_at: Instant::now(),
        confidence: 0.8,
    };
    
    // Store 1000 similar exceptions
    for i in 0..1000 {
        let key = ExceptionKey {
            node_id: NodeId(i),
            property_hash: hash_string("test_property"),
        };
        
        let mut exception = base_exception.clone();
        exception.actual_value = PropertyValue::String(format!("override_value_{}", i));
        
        optimizer.store_exception(key, exception).expect("Failed to store");
    }
    
    let stats = optimizer.get_storage_statistics();
    let compression_ratio = stats.compression_ratio.load(Ordering::Relaxed) as f32 / 100.0;
    
    assert!(compression_ratio > 3.0); // >3:1 compression ratio
    
    let memory_efficiency = stats.total_compressed_size.load(Ordering::Relaxed) as f32 
                          / stats.total_uncompressed_size.load(Ordering::Relaxed) as f32;
    assert!(memory_efficiency < 0.5); // >50% memory reduction
}

#[test]
fn test_bulk_operations_performance() {
    let mut optimizer = StorageOptimizer::new(10000);
    
    // Prepare bulk data
    let mut exceptions = Vec::new();
    for i in 0..10000 {
        let key = ExceptionKey {
            node_id: NodeId(i),
            property_hash: hash_string(&format!("prop_{}", i % 100)),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: format!("Exception {}", i),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        
        exceptions.push((key, exception));
    }
    
    // Test bulk store performance
    let start = Instant::now();
    let stored_count = optimizer.bulk_store(exceptions).expect("Bulk store failed");
    let store_time = start.elapsed();
    
    assert_eq!(stored_count, 10000);
    
    let throughput = 10000.0 / store_time.as_secs_f64();
    assert!(throughput > 10000.0); // >10,000 exceptions/second
    
    // Test bulk retrieve performance
    let keys: Vec<ExceptionKey> = (0..10000).map(|i| ExceptionKey {
        node_id: NodeId(i),
        property_hash: hash_string(&format!("prop_{}", i % 100)),
    }).collect();
    
    let start = Instant::now();
    let retrieved = optimizer.bulk_retrieve(&keys).expect("Bulk retrieve failed");
    let retrieve_time = start.elapsed();
    
    assert_eq!(retrieved.len(), 10000);
    assert!(retrieved.iter().all(|opt| opt.is_some()));
    
    let retrieve_throughput = 10000.0 / retrieve_time.as_secs_f64();
    assert!(retrieve_throughput > 10000.0);
}

#[test]
fn test_index_performance() {
    let mut optimizer = StorageOptimizer::new(1000);
    
    // Store exceptions with various properties
    for i in 0..1000 {
        let key = ExceptionKey {
            node_id: NodeId(i),
            property_hash: hash_string(&format!("prop_{}", i % 10)),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("value_{}", i)),
            reason: "Test".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: (i as f32) / 1000.0,
        };
        
        optimizer.store_exception(key, exception).expect("Failed to store");
    }
    
    // Test property index query
    let start = Instant::now();
    let property_results = optimizer.query_by_property(hash_string("prop_5"));
    let property_query_time = start.elapsed();
    
    assert_eq!(property_results.len(), 100); // Should find 100 matches
    assert!(property_query_time < Duration::from_millis(1)); // <1ms lookup
    
    // Test confidence range query
    let start = Instant::now();
    let confidence_results = optimizer.query_by_confidence_range(0.5, 0.7);
    let confidence_query_time = start.elapsed();
    
    assert!(!confidence_results.is_empty());
    assert!(confidence_query_time < Duration::from_millis(5)); // <5ms range query
}

#[test]
fn test_memory_pool_efficiency() {
    let mut optimizer = StorageOptimizer::new(100);
    
    // Store and remove exceptions to test pooling
    for round in 0..10 {
        for i in 0..100 {
            let key = ExceptionKey {
                node_id: NodeId(i + round * 100),
                property_hash: hash_string("test_prop"),
            };
            
            let exception = Exception {
                inherited_value: PropertyValue::String("default".to_string()),
                actual_value: PropertyValue::String(format!("value_{}_{}", round, i)),
                reason: "Test pooling".to_string(),
                source: ExceptionSource::Detected,
                created_at: Instant::now(),
                confidence: 0.8,
            };
            
            optimizer.store_exception(key, exception).expect("Failed to store");
        }
        
        // Remove half the exceptions to create reusable pool objects
        for i in 0..50 {
            let key = ExceptionKey {
                node_id: NodeId(i + round * 100),
                property_hash: hash_string("test_prop"),
            };
            optimizer.retrieve_exception(&key).expect("Failed to retrieve");
        }
    }
    
    let stats = optimizer.get_storage_statistics();
    // After several rounds, pool should be providing significant reuse
    // This is implementation-dependent, but we expect some efficiency gains
}

#[test]
fn test_storage_optimization() {
    let mut optimizer = StorageOptimizer::new(1000);
    
    // Fill with somewhat fragmented data
    for i in 0..1000 {
        let key = ExceptionKey {
            node_id: NodeId(i),
            property_hash: hash_string(&format!("prop_{}", i % 10)),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::String("base".to_string()),
            actual_value: PropertyValue::String(format!("{}_{}", "value", i)),
            reason: "Fragmentation test".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        
        optimizer.store_exception(key, exception).expect("Failed to store");
    }
    
    let pre_optimization_stats = optimizer.get_storage_statistics();
    
    // Run optimization
    let optimization_results = optimizer.optimize_storage();
    let post_optimization_stats = optimizer.get_storage_statistics();
    
    // Should have some improvement
    assert!(optimization_results.memory_saved > 0);
    assert!(optimization_results.compression_ratio > 1.0);
    
    // Storage should be more efficient after optimization
    let pre_size = pre_optimization_stats.total_compressed_size.load(Ordering::Relaxed);
    let post_size = post_optimization_stats.total_compressed_size.load(Ordering::Relaxed);
    assert!(post_size <= pre_size); // Should not increase size
}

fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}
```

## File Location
`src/exceptions/storage_optimizer.rs`

## Next Micro Phase
After completion, proceed to Micro 2.6: Exception Integration Tests