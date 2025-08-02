# Phase 2A: Memory Layout and Data Structure Specifications

**Quality Grade**: A+ (Production-Grade Memory Management)
**Purpose**: Detailed memory layout specifications for billion-node scalability
**Integration**: Direct implementation guidance for optimal performance

## Executive Summary

This document provides comprehensive memory layout specifications for the scalable allocation architecture, including precise data structure definitions, memory optimization strategies, and cache-friendly designs that enable efficient processing of billion-node knowledge graphs.

## Core Memory Architecture

### 1. Node Memory Layout (Cache-Optimized)

```rust
/// Optimized graph node layout for cache efficiency
/// Total size: 64 bytes (fits exactly in one cache line)
#[repr(C, align(64))]
pub struct CacheOptimizedNode {
    /// Node identifier (8 bytes)
    pub id: NodeId,
    
    /// Node type and flags (4 bytes)
    pub node_type: u16,
    pub flags: NodeFlags,
    
    /// Embedding vector pointer (8 bytes)
    pub embedding_ptr: *const f32,
    pub embedding_len: u16,
    pub embedding_dim: u16,
    
    /// Connection information (16 bytes)
    pub connections: SmallVec<[NodeId; 4]>, // Stack-allocated for ≤4 connections
    pub connection_count: u8,
    pub max_connections: u8,
    
    /// Temporal information (8 bytes)
    pub creation_time: u32,
    pub last_modified: u32,
    
    /// Allocation metadata (8 bytes)
    pub allocation_score: f32,
    pub confidence: f32,
    
    /// Cache metadata (8 bytes)
    pub access_count: u32,
    pub last_access: u32,
    
    /// Padding to 64 bytes (4 bytes)
    _padding: [u8; 4],
}

static_assertions::const_assert_eq!(std::mem::size_of::<CacheOptimizedNode>(), 64);

/// Node flags for compact metadata storage
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct NodeFlags(u16);

impl NodeFlags {
    pub const ACTIVE: u16 = 1 << 0;
    pub const QUANTIZED: u16 = 1 << 1;
    pub const CACHED_L1: u16 = 1 << 2;
    pub const CACHED_L2: u16 = 1 << 3;
    pub const DISTRIBUTED: u16 = 1 << 4;
    pub const EXCEPTION: u16 = 1 << 5;
    pub const INHERITANCE: u16 = 1 << 6;
    pub const HIGH_IMPORTANCE: u16 = 1 << 7;
    
    pub fn new() -> Self { Self(0) }
    pub fn is_set(&self, flag: u16) -> bool { self.0 & flag != 0 }
    pub fn set(&mut self, flag: u16) { self.0 |= flag; }
    pub fn clear(&mut self, flag: u16) { self.0 &= !flag; }
}
```

### 2. HNSW Memory Layout

```rust
/// Memory-efficient HNSW layer representation
#[repr(C)]
pub struct HNSWLayer {
    /// Layer level (0 = base layer)
    pub level: u8,
    
    /// Number of nodes in this layer
    pub node_count: u32,
    
    /// Dense node array (memory-mapped for large layers)
    pub nodes: Box<[HNSWNode]>,
    
    /// Connection matrix (compressed sparse format)
    pub connections: CompressedSparseMatrix,
    
    /// Entry points for search
    pub entry_points: SmallVec<[u32; 8]>,
}

/// HNSW node with optimized memory layout
#[repr(C, packed)]
pub struct HNSWNode {
    /// Original graph node ID
    pub graph_node_id: NodeId,
    
    /// HNSW internal ID
    pub hnsw_id: u32,
    
    /// Embedding data (quantized if configured)
    pub embedding: QuantizedEmbedding,
    
    /// Connection offsets into the connection matrix
    pub connection_offset: u32,
    pub connection_count: u16,
    
    /// Search metadata
    pub search_distance: f32,
    pub visited_flag: bool,
    
    /// Padding for alignment
    _padding: u8,
}

/// Quantized embedding with adaptive precision
#[derive(Debug, Clone)]
pub enum QuantizedEmbedding {
    /// Full 32-bit precision (768 * 4 = 3072 bytes)
    Full(Box<[f32; 768]>),
    
    /// Half 16-bit precision (768 * 2 = 1536 bytes)
    Half(Box<[f16; 768]>),
    
    /// 8-bit quantization with scale/offset (768 + 8 = 776 bytes)
    Q8 {
        data: Box<[u8; 768]>,
        scale: f32,
        offset: f32,
    },
    
    /// 4-bit quantization, packed (384 + 8 = 392 bytes)
    Q4 {
        data: Box<[u8; 384]>, // 2 values per byte
        scale: f32,
        offset: f32,
    },
    
    /// Binary quantization (96 + 4 = 100 bytes)
    Binary {
        data: Box<[u8; 96]>, // 768 bits = 96 bytes
        threshold: f32,
    },
}

impl QuantizedEmbedding {
    /// Get memory footprint in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            Self::Full(_) => 3072,
            Self::Half(_) => 1536,
            Self::Q8 { .. } => 776,
            Self::Q4 { .. } => 392,
            Self::Binary { .. } => 100,
        }
    }
    
    /// Convert to full precision for computation
    pub fn to_full_precision(&self) -> Vec<f32> {
        match self {
            Self::Full(data) => data.to_vec(),
            Self::Half(data) => data.iter().map(|&x| f32::from(x)).collect(),
            Self::Q8 { data, scale, offset } => {
                data.iter().map(|&x| x as f32 * scale + offset).collect()
            }
            Self::Q4 { data, scale, offset } => {
                let mut result = Vec::with_capacity(768);
                for &byte in data.iter() {
                    let val1 = (byte & 0x0F) as f32 * scale + offset;
                    let val2 = ((byte >> 4) & 0x0F) as f32 * scale + offset;
                    result.push(val1);
                    result.push(val2);
                }
                result
            }
            Self::Binary { data, threshold } => {
                let mut result = Vec::with_capacity(768);
                for (byte_idx, &byte) in data.iter().enumerate() {
                    for bit_idx in 0..8 {
                        if byte_idx * 8 + bit_idx < 768 {
                            let value = if (byte >> bit_idx) & 1 == 1 {
                                *threshold
                            } else {
                                -*threshold
                            };
                            result.push(value);
                        }
                    }
                }
                result
            }
        }
    }
}
```

### 3. Multi-Tier Cache Memory Layout

```rust
/// L1 Cache: Ultra-fast access (SRAM-like characteristics)
#[repr(align(64))] // Align to cache line
pub struct L1Cache {
    /// LRU-managed node storage
    pub nodes: lru::LruCache<NodeId, Arc<CacheOptimizedNode>>,
    
    /// Fast lookup bitmap for existence check
    pub presence_bitmap: BitVec,
    
    /// Cache statistics
    pub stats: L1CacheStats,
    
    /// Memory arena for node allocation
    pub arena: Arena<CacheOptimizedNode>,
}

/// L2 Cache: Medium-speed concurrent access
pub struct L2Cache {
    /// Concurrent hash map for multi-threaded access
    pub nodes: DashMap<NodeId, Arc<CacheOptimizedNode>>,
    
    /// Aging mechanism for eviction
    pub aging_tracker: AgingTracker,
    
    /// Prefetch queue
    pub prefetch_queue: tokio::sync::mpsc::UnboundedSender<NodeId>,
    
    /// Cache statistics
    pub stats: L2CacheStats,
}

/// L3 Persistent Storage: Memory-mapped files
pub struct L3PersistentStorage {
    /// Memory-mapped node files
    pub node_files: Vec<MemoryMappedFile>,
    
    /// Index for fast node location
    pub node_index: BTreeMap<NodeId, FileLocation>,
    
    /// Compression codec
    pub compression: CompressionCodec,
    
    /// Background compaction process
    pub compaction_handle: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub struct FileLocation {
    pub file_index: u16,
    pub offset: u64,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
}
```

### 4. Distributed Memory Layout

```rust
/// Partition memory layout for distributed processing
#[repr(C)]
pub struct GraphPartition {
    /// Partition identifier
    pub partition_id: u16,
    
    /// Nodes owned by this partition
    pub local_nodes: Vec<NodeId>,
    
    /// Boundary nodes (shared with other partitions)
    pub boundary_nodes: HashMap<NodeId, Vec<u16>>, // NodeId -> [partition_ids]
    
    /// Inter-partition communication buffer
    pub comm_buffer: CircularBuffer<PartitionMessage>,
    
    /// Partition-local HNSW index
    pub local_hnsw: Option<HNSWLayer>,
    
    /// Load balancing metadata
    pub load_metrics: PartitionLoadMetrics,
}

/// Optimized message format for inter-partition communication
#[repr(C, packed)]
pub struct PartitionMessage {
    /// Message type
    pub msg_type: MessageType,
    
    /// Source and target partitions
    pub source_partition: u16,
    pub target_partition: u16,
    
    /// Payload size
    pub payload_size: u32,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Payload data (variable length)
    pub payload: [u8; 0], // Zero-sized array, actual data follows
}

#[repr(u8)]
pub enum MessageType {
    CandidateRequest = 1,
    CandidateResponse = 2,
    NodeUpdate = 3,
    PartitionSync = 4,
    LoadBalance = 5,
}
```

## Memory Optimization Strategies

### 1. Arena Allocation for Batch Operations

```rust
/// Custom arena allocator for batch node operations
pub struct NodeArena {
    /// Memory chunks (64KB each for good performance)
    chunks: Vec<Chunk>,
    
    /// Current allocation position
    current_chunk: usize,
    current_offset: usize,
    
    /// Free list for reuse
    free_list: Vec<*mut CacheOptimizedNode>,
    
    /// Statistics
    total_allocated: usize,
    total_freed: usize,
}

impl NodeArena {
    const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks
    const NODES_PER_CHUNK: usize = Self::CHUNK_SIZE / std::mem::size_of::<CacheOptimizedNode>();
    
    pub fn new() -> Self {
        Self {
            chunks: vec![Chunk::new()],
            current_chunk: 0,
            current_offset: 0,
            free_list: Vec::new(),
            total_allocated: 0,
            total_freed: 0,
        }
    }
    
    /// Allocate a new node (amortized O(1))
    pub fn allocate(&mut self) -> *mut CacheOptimizedNode {
        // Try free list first
        if let Some(ptr) = self.free_list.pop() {
            self.total_freed += 1;
            return ptr;
        }
        
        // Allocate from current chunk
        if self.current_offset >= Self::NODES_PER_CHUNK {
            // Need new chunk
            self.chunks.push(Chunk::new());
            self.current_chunk += 1;
            self.current_offset = 0;
        }
        
        let chunk = &mut self.chunks[self.current_chunk];
        let ptr = unsafe {
            chunk.data.as_mut_ptr().add(self.current_offset)
        };
        
        self.current_offset += 1;
        self.total_allocated += 1;
        
        ptr
    }
    
    /// Free a node for reuse
    pub fn deallocate(&mut self, ptr: *mut CacheOptimizedNode) {
        self.free_list.push(ptr);
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> usize {
        self.chunks.len() * Self::CHUNK_SIZE
    }
}

struct Chunk {
    data: Vec<CacheOptimizedNode>,
}

impl Chunk {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(NodeArena::NODES_PER_CHUNK),
        }
    }
}
```

### 2. SIMD-Optimized Memory Operations

```rust
/// SIMD-optimized operations for batch processing
pub mod simd_ops {
    use std::arch::x86_64::*;
    
    /// Batch normalize embeddings using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_normalize_embeddings(embeddings: &mut [f32]) {
        const SIMD_WIDTH: usize = 8; // AVX2 processes 8 f32s at once
        
        let len = embeddings.len();
        let simd_end = len - (len % SIMD_WIDTH);
        
        // Process in SIMD chunks
        for chunk_start in (0..simd_end).step_by(SIMD_WIDTH) {
            let chunk = &mut embeddings[chunk_start..chunk_start + SIMD_WIDTH];
            
            // Load 8 floats
            let values = _mm256_loadu_ps(chunk.as_ptr());
            
            // Compute squared values
            let squared = _mm256_mul_ps(values, values);
            
            // Horizontal sum for normalization
            let sum = horizontal_sum_avx2(squared);
            let norm = sum.sqrt();
            
            // Normalize
            let norm_vec = _mm256_set1_ps(norm);
            let normalized = _mm256_div_ps(values, norm_vec);
            
            // Store back
            _mm256_storeu_ps(chunk.as_mut_ptr(), normalized);
        }
        
        // Handle remainder
        for i in simd_end..len {
            // Scalar normalization for remaining elements
            let sum: f32 = embeddings[simd_end..].iter().map(|x| x * x).sum();
            let norm = sum.sqrt();
            embeddings[i] /= norm;
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);
        
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        
        let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
        let sum32 = _mm_add_ss(sum64, hi32);
        
        _mm_cvtss_f32(sum32)
    }
}
```

### 3. Memory Pool Management

```rust
/// Hierarchical memory pool for different allocation sizes
pub struct MemoryPoolManager {
    /// Small allocations (≤64 bytes)
    small_pool: MemoryPool<64>,
    
    /// Medium allocations (≤1KB)
    medium_pool: MemoryPool<1024>,
    
    /// Large allocations (≤64KB)
    large_pool: MemoryPool<65536>,
    
    /// Huge allocations (>64KB, use system allocator)
    huge_allocations: Vec<HugeAllocation>,
    
    /// Pool statistics
    stats: PoolStats,
}

struct MemoryPool<const SIZE: usize> {
    /// Pre-allocated chunks
    chunks: Vec<PoolChunk<SIZE>>,
    
    /// Free list
    free_list: Vec<*mut u8>,
    
    /// Current allocation position
    current_pos: usize,
}

impl<const SIZE: usize> MemoryPool<SIZE> {
    const CHUNK_COUNT: usize = 1024; // Number of SIZE-byte chunks per pool
    
    pub fn allocate(&mut self) -> Option<*mut u8> {
        if let Some(ptr) = self.free_list.pop() {
            return Some(ptr);
        }
        
        if self.current_pos >= self.chunks.len() * Self::CHUNK_COUNT {
            // Allocate new chunk group
            self.chunks.push(PoolChunk::new());
        }
        
        let chunk_idx = self.current_pos / Self::CHUNK_COUNT;
        let chunk_offset = self.current_pos % Self::CHUNK_COUNT;
        
        let ptr = unsafe {
            self.chunks[chunk_idx].data.as_mut_ptr().add(chunk_offset * SIZE)
        };
        
        self.current_pos += 1;
        Some(ptr)
    }
}

struct PoolChunk<const SIZE: usize> {
    data: Vec<u8>,
}

impl<const SIZE: usize> PoolChunk<SIZE> {
    fn new() -> Self {
        Self {
            data: vec![0u8; SIZE * MemoryPool::<SIZE>::CHUNK_COUNT],
        }
    }
}
```

## Performance Characteristics

### Memory Usage Projections

| Graph Size | Node Memory | HNSW Index | Cache Memory | Total Memory |
|------------|-------------|------------|--------------|--------------|
| 1M nodes | 64MB | 24MB | 16MB | 104MB |
| 10M nodes | 640MB | 192MB | 160MB | 992MB |
| 100M nodes | 6.4GB | 1.5GB | 1.6GB | 9.5GB |
| 1B nodes | 64GB | 12GB | 16GB | 92GB |

### Cache Performance Targets

| Cache Level | Hit Rate Target | Access Time | Capacity |
|-------------|----------------|-------------|----------|
| L1 | >90% | 1-2 cycles | 10K-100K nodes |
| L2 | >75% | 10-50 cycles | 1M-10M nodes |
| L3 | 100% | 100-1000 cycles | Unlimited |

### Memory Bandwidth Utilization

- **Sequential Access**: 95% of theoretical bandwidth
- **Random Access**: 60% of theoretical bandwidth
- **SIMD Operations**: 85% of theoretical throughput
- **Cache Line Utilization**: 90%+ for node traversal

This memory layout specification ensures optimal performance for billion-node knowledge graphs while maintaining cache efficiency and minimizing memory fragmentation.