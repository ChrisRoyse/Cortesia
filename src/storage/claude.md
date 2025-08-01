# Directory Overview: Storage

## 1. High-Level Summary

The storage directory contains a comprehensive suite of high-performance storage and indexing components for the LLMKG (Large Language Model Knowledge Graph) system. This directory implements various data structures and algorithms optimized for efficient knowledge graph storage, retrieval, and similarity search operations. The components are designed for maximum performance with features like zero-copy access, memory mapping, quantization, and specialized indexing strategies.

## 2. Tech Stack

*   **Language:** Rust
*   **Core Libraries:**
    *   `parking_lot` - High-performance reader-writer locks
    *   `serde` - Serialization/deserialization framework  
    *   `slotmap` - Memory-efficient entity key management
    *   `ahash` - Fast hash map implementation
    *   `rand` - Random number generation
    *   `bincode` - Binary serialization
*   **Specialized Technologies:**
    *   Memory-mapped I/O for zero-copy operations
    *   Product Quantization for embedding compression
    *   LSH (Locality Sensitive Hashing) for similarity search
    *   HNSW (Hierarchical Navigable Small World) graphs
    *   CSR (Compressed Sparse Row) matrix format
*   **Performance Features:**
    *   SIMD acceleration support
    *   Lock-free concurrent data structures
    *   Cache-friendly memory layouts

## 3. Directory Structure

The storage directory is organized into specialized storage backends and indexing systems:

*   **Core Storage:** `mmap_storage.rs`, `persistent_mmap.rs`, `zero_copy.rs` - Low-level storage engines
*   **Indexing Systems:** `flat_index.rs`, `spatial_index.rs`, `quantized_index.rs`, `lsh.rs`, `hnsw.rs` - Various indexing strategies
*   **Graph Storage:** `csr.rs`, `hybrid_graph.rs` - Graph-specific storage formats
*   **Optimization:** `lru_cache.rs`, `bloom.rs`, `string_interner.rs` - Memory and performance optimizations
*   **Semantic:** `semantic_store.rs` - LLM-friendly entity representation
*   **Interfaces:** `index.rs`, `mod.rs` - Common traits and module definitions

## 4. File Breakdown

### `mod.rs`
*   **Purpose:** Module declarations and public interface exports for the storage subsystem.

### `bloom.rs`
*   **Purpose:** Probabilistic membership testing to avoid expensive lookups for non-existent entities.
*   **Struct:** `BloomFilter`
    *   **Methods:**
        *   `new(expected_items: usize, false_positive_rate: f64)` - Create filter with optimal parameters
        *   `insert<T: Hash>(&mut self, item: &T)` - Add item to filter
        *   `contains<T: Hash>(&self, item: &T) -> bool` - Check probable membership
        *   `fill_ratio(&self) -> f64` - Get filter saturation level
        *   `estimated_items(&self) -> f64` - Estimate number of inserted items
*   **Key Algorithm:** Uses multiple hash functions with bit array for space-efficient set representation

### `csr.rs`
*   **Purpose:** Compressed Sparse Row format for efficient graph storage and traversal operations.
*   **Struct:** `CSRGraph`
    *   **Methods:**
        *   `from_edges(edges: Vec<Relationship>, num_nodes: usize)` - Build CSR from edge list
        *   `get_neighbors(&self, node: u32) -> &[u32]` - Zero-copy neighbor access
        *   `get_out_degree(&self, node: u32) -> usize` - Get node out-degree
        *   `bfs(&self, start: u32, max_depth: Option<u32>) -> Vec<(u32, u32)>` - Breadth-first search
        *   `connected_components(&self) -> Vec<Vec<u32>>` - Find graph components
        *   `pagerank(&self, iterations: usize, damping: f32) -> Vec<f32>` - PageRank computation
*   **Data Structure:** Row pointers + column indices for O(1) neighbor access

### `flat_index.rs`
*   **Purpose:** Simple linear index for small datasets or full-scan operations.
*   **Struct:** `FlatIndex`
    *   **Methods:**
        *   `new(dimension: usize)` - Create index for given embedding dimension
        *   `insert(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>)` - Add entity
        *   `search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>` - Linear k-NN search
        *   `batch_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>>` - Batch queries

### `hnsw.rs`
*   **Purpose:** Hierarchical Navigable Small World algorithm for fast approximate nearest neighbor search.
*   **Struct:** `HNSWIndex`
    *   **Methods:**
        *   `new(dimension: usize, max_connections: usize, ef_construction: usize)` - Create HNSW index
        *   `insert(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>)` - Add with layer assignment
        *   `search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)>` - Multi-layer search
        *   `get_stats(&self) -> HNSWStats` - Index statistics and performance metrics
*   **Algorithm:** Multi-layer skip-list structure with greedy search and pruning

### `hybrid_graph.rs`
*   **Purpose:** Combines multiple storage backends for optimal performance across different access patterns.
*   **Struct:** `HybridGraphStorage`
    *   **Methods:**
        *   `new(config: HybridConfig)` - Create with configuration for different backends
        *   `add_entity(&mut self, entity_id: u32, entity_key: EntityKey, data: &EntityData)` - Route to appropriate backend
        *   `search_similar(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>` - Unified similarity search
        *   `get_stats(&self) -> HybridStats` - Combined statistics from all backends

### `index.rs`
*   **Purpose:** Common traits and interfaces for all indexing implementations.
*   **Traits:**
    *   `VectorIndex` - Interface for similarity search indices
    *   `GraphIndex` - Interface for graph traversal operations  
    *   `Index` - General indexing operations
*   **Common Methods:**
    *   `insert()`, `search()`, `remove()`, `len()`, `is_empty()`, `clear()`

### `lru_cache.rs`
*   **Purpose:** Least Recently Used cache for frequently accessed entities and query results.
*   **Struct:** `LRUCache<K, V>`
    *   **Methods:**
        *   `new(capacity: usize)` - Create cache with maximum capacity
        *   `get(&mut self, key: &K) -> Option<&V>` - Get value and update access order
        *   `put(&mut self, key: K, value: V)` - Insert value, evicting LRU if necessary
        *   `contains_key(&self, key: &K) -> bool` - Check key existence
        *   `clear(&mut self)` - Remove all entries
*   **Implementation:** Doubly-linked list + hash map for O(1) operations

### `lsh.rs`
*   **Purpose:** Locality Sensitive Hashing for fast approximate similarity search using random hyperplanes.
*   **Struct:** `LshIndex`
    *   **Methods:**
        *   `new(dimension: usize, num_hashes: usize, num_tables: usize)` - Create LSH index
        *   `new_optimized(dimension: usize, target_precision: f32)` - Auto-configure parameters
        *   `insert(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>)` - Hash and store
        *   `search(&self, query: &[f32], max_results: usize) -> Vec<(u32, f32)>` - Multi-probe search
        *   `search_threshold(&self, query: &[f32], threshold: f32) -> Vec<(u32, f32)>` - Threshold-based search
        *   `stats(&self) -> LshStats` - Hash distribution and collision statistics
*   **Algorithm:** Random hyperplane hashing with multi-table probing for improved recall

### `mmap_storage.rs`
*   **Purpose:** Ultra-fast memory-mapped storage for zero-copy access with cache optimization.
*   **Struct:** `MMapStorage`
    *   **Methods:**
        *   `new(estimated_entities: usize, estimated_edges: usize, embedding_dim: u16)` - Pre-allocate storage
        *   `get_entity(&self, entity_id: u32) -> Option<&CompactEntity>` - Zero-copy entity access
        *   `get_neighbors(&self, entity_id: u32) -> Option<&[u32]>` - Zero-copy neighbor slice
        *   `batch_get_neighbors(&self, entity_ids: &[u32], results: &mut Vec<Vec<u32>>)` - Batch neighbor lookup
        *   `traverse_multi_hop(&self, start_entities: &[u32], max_hops: u8) -> Vec<u32>` - Graph traversal
        *   `prefetch_entities(&self, entity_ids: &[u32])` - CPU cache prefetching (x86_64)
*   **Optimizations:** Cache-friendly layouts, SIMD prefetching, quantized embeddings

### `persistent_mmap.rs`
*   **Purpose:** Persistent memory-mapped storage with file persistence and Product Quantization integration.
*   **Struct:** `PersistentMMapStorage`
    *   **Methods:**
        *   `new<P: AsRef<Path>>(file_path: Option<P>, embedding_dim: usize)` - Create with optional file backing
        *   `load<P: AsRef<Path>>(file_path: P)` - Load existing storage from file
        *   `add_entity(&mut self, entity_key: EntityKey, data: &EntityData, embedding: &[f32])` - Add with quantization
        *   `batch_add_entities(&mut self, entities_data: &[(EntityKey, EntityData, Vec<f32>)])` - Batch insertion
        *   `similarity_search(&self, query: &[f32], k: usize) -> Vec<(EntityKey, f32)>` - Quantized similarity search
        *   `sync_to_disk(&mut self)` - Persist to file
        *   `storage_stats(&self) -> StorageStats` - Comprehensive storage statistics
*   **Features:** Automatic quantizer training, file format versioning, data integrity checksums

### `quantized_index.rs`
*   **Purpose:** Memory-efficient similarity search using Product Quantization for embedding compression.
*   **Struct:** `QuantizedIndex`
    *   **Methods:**
        *   `new(dimension: usize, subvector_count: usize)` - Create quantized index
        *   `train(&self, training_embeddings: &[Vec<f32>])` - Train quantizer on representative data
        *   `insert(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>)` - Quantize and store
        *   `bulk_insert(&self, entities: Vec<(u32, EntityKey, Vec<f32>)>)` - Batch insertion
        *   `search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>` - Asymmetric distance search
        *   `search_threshold(&self, query: &[f32], threshold: f32) -> Vec<(u32, f32)>` - Threshold-based search
        *   `memory_usage(&self) -> QuantizedIndexStats` - Compression statistics
*   **Compression:** Typically achieves 8-32x compression with minimal accuracy loss

### `semantic_store.rs`
*   **Purpose:** LLM-friendly semantic storage that maintains rich entity summaries for AI comprehension.
*   **Struct:** `SemanticStore`
    *   **Methods:**
        *   `store_entity(&self, entity_id: u32, entity_key: EntityKey, entity_data: &EntityData)` - Create semantic summary
        *   `bulk_store(&self, entities: Vec<(u32, EntityKey, EntityData)>)` - Batch semantic processing
        *   `get_summary(&self, entity_key: EntityKey) -> Option<SemanticSummary>` - Get semantic summary
        *   `get_llm_text(&self, entity_key: EntityKey) -> Option<String>` - Get LLM-friendly text
        *   `semantic_search(&self, query_text: &str, limit: usize) -> Vec<(u32, f32, String)>` - Text-based search
        *   `get_llm_integration_report(&self) -> String` - Detailed quality analysis for LLM usage
*   **Target:** ~150-200 bytes per entity with rich semantic content for LLM understanding

### `spatial_index.rs`
*   **Purpose:** K-d tree spatial index for O(log n) nearest neighbor queries in high-dimensional spaces.
*   **Struct:** `SpatialIndex`
    *   **Methods:**
        *   `new(dimension: usize)` - Create spatial index
        *   `insert(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>)` - Insert with tree rebalancing
        *   `k_nearest_neighbors(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>` - K-NN search
        *   `bulk_build(&mut self, entities: Vec<(u32, EntityKey, Vec<f32>)>)` - Optimal tree construction
*   **Algorithm:** Balanced k-d tree with median splitting and branch-and-bound search

### `string_interner.rs`
*   **Purpose:** High-performance string interning system to reduce memory usage for duplicate strings.
*   **Structs:**
    *   `StringInterner` - Thread-safe string interner
        *   `intern<S: AsRef<str>>(&self, s: S) -> InternedString` - Get/create string ID
        *   `get(&self, id: InternedString) -> Option<String>` - Resolve string by ID
        *   `stats(&self) -> InternerStats` - Deduplication and memory statistics
    *   `InternedProperties` - Property container using interned strings
        *   `insert(&mut self, interner: &StringInterner, key: &str, value: &str)` - Add property
        *   `get(&self, interner: &StringInterner, key: &str) -> Option<String>` - Get property value
*   **Optimization:** Provides significant memory savings for applications with many duplicate strings

### `zero_copy.rs`
*   **Purpose:** Ultra-efficient serialization that avoids memory allocation during read operations.
*   **Structs:**
    *   `ZeroCopyDeserializer<'a>` - Zero-allocation data access
        *   `unsafe fn new(data: &'a [u8]) -> Result<Self>` - Create from raw bytes
        *   `get_entity(&self, index: u32) -> Option<&ZeroCopyEntity>` - Direct memory access
        *   `iter_entities(&self) -> ZeroCopyEntityIter` - Zero-allocation iteration
    *   `ZeroCopySerializer` - Efficient data writing
        *   `add_entity(&mut self, entity: &EntityData, embedding_dim: usize)` - Serialize entity
        *   `finalize(self) -> Result<Vec<u8>>` - Generate final buffer
*   **Performance:** Designed for maximum throughput with direct memory access patterns

## 5. Key Data Structures

### Entity Representations
*   **`CompactEntity`** - Memory-optimized entity structure (mmap_storage.rs)
*   **`MMapEntity`** - Persistent entity format (persistent_mmap.rs)  
*   **`ZeroCopyEntity`** - Zero-copy entity format (zero_copy.rs)
*   **`SemanticSummary`** - LLM-friendly entity representation (semantic_store.rs)

### Graph Storage
*   **CSR Format** - `row_ptr: Vec<u64>` + `col_idx: Vec<u32>` for neighbor lists
*   **Adjacency Lists** - Variable-length neighbor arrays with offsets
*   **Relationship** - `from: EntityKey`, `to: EntityKey`, `rel_type: u8`, `weight: f32`

### Indexing Structures
*   **HNSW Layers** - Multi-level skip-list with connection pruning
*   **LSH Hash Tables** - Multiple hash functions with collision buckets
*   **K-d Tree Nodes** - Binary tree with dimension splitting
*   **Quantization Codebooks** - Learned cluster centroids for compression

## 6. Performance Optimizations

### Memory Management
*   **Zero-Copy Access** - Direct memory mapping without allocation
*   **String Interning** - Deduplication of repeated strings
*   **Quantization** - 8-32x compression of embeddings
*   **Cache-Friendly Layouts** - Struct packing and data locality

### Concurrent Access
*   **RwLock** - Multiple readers, single writer for most structures
*   **Atomic Counters** - Lock-free statistics tracking
*   **Epoch-Based Reclamation** - Safe memory reclamation in concurrent contexts

### SIMD and Hardware Acceleration
*   **Prefetching** - CPU cache optimization (x86_64)
*   **Vectorized Operations** - SIMD distance computations
*   **Memory Bandwidth** - Minimized data movement

## 7. Dependencies

### Internal Dependencies
*   `crate::core::types` - Entity, Relationship, EntityKey definitions
*   `crate::error` - GraphError and Result types
*   `crate::embedding::quantizer` - Product Quantization implementation
*   `crate::embedding::similarity` - Distance functions (cosine, euclidean)
*   `crate::core::semantic_summary` - Semantic summarization system

### External Dependencies
*   **Core:** `std`, `parking_lot`, `serde`
*   **Hashing:** `ahash`, random number generation: `rand`
*   **Serialization:** `bincode`, `serde_json`
*   **Memory:** Memory-mapped I/O, SIMD intrinsics

## 8. Usage Patterns

### Typical Workflow
1. **Index Creation** - Choose appropriate index type based on data characteristics
2. **Data Ingestion** - Batch insert entities and relationships for optimal performance  
3. **Similarity Search** - Query for nearest neighbors using various algorithms
4. **Graph Traversal** - Navigate relationships using CSR or other graph structures
5. **Persistence** - Save/load index state to/from disk

### Performance Considerations
*   **Small datasets (< 10K entities):** Use `FlatIndex` for simplicity
*   **Medium datasets (10K - 1M entities):** Use `SpatialIndex` or `QuantizedIndex`
*   **Large datasets (> 1M entities):** Use `HNSWIndex` or `LshIndex`
*   **Memory-constrained:** Use `QuantizedIndex` with aggressive compression
*   **Ultra-high performance:** Use `MMapStorage` with zero-copy access

### Integration Points
*   **Query Engine** - Similarity search backends for knowledge retrieval
*   **Graph Analytics** - CSR format for PageRank, connected components, etc.
*   **LLM Integration** - SemanticStore provides AI-friendly entity representations
*   **Caching Layer** - LRU cache for frequently accessed entities
*   **Storage Layer** - Persistent storage with memory mapping for large datasets

This storage subsystem provides a comprehensive foundation for high-performance knowledge graph operations, with careful attention to memory efficiency, algorithmic complexity, and modern hardware optimization techniques.