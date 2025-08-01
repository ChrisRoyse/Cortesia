# Directory Overview: Embedding

## 1. High-Level Summary

The `embedding` directory contains a high-performance embedding storage and similarity search system for a large-scale Knowledge Graph (LLMKG). It implements advanced techniques including Product Quantization for compression, SIMD acceleration for fast similarity computations, and optimized batch processing. The system is designed for efficient storage and retrieval of high-dimensional embedding vectors while maintaining search quality.

## 2. Tech Stack

*   **Language:** Rust
*   **Key Features:** 
    *   Product Quantization (PQ) compression
    *   SIMD acceleration (AVX2, SSE4.1)
    *   Memory-efficient storage
    *   High-throughput batch processing
*   **Dependencies:**
    *   `parking_lot` - Fast RwLock implementation
    *   `std::arch::x86_64` - SIMD intrinsics
    *   Custom error types from `crate::error`
    *   Entity types from `crate::core::types`

## 3. Module Structure

*   **`mod.rs`** - Module root with public exports and shared types
*   **`quantizer.rs`** - Product Quantization implementation
*   **`simd_search.rs`** - SIMD-accelerated similarity search
*   **`similarity.rs`** - Distance and similarity metrics
*   **`store.rs`** - Core embedding storage
*   **`store_compat.rs`** - Compatibility layer for performance tests

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module definition, public exports, and shared type definitions
*   **Structs:**
    *   `FilteredEmbedding`
        *   **Description:** Embedding with filtering metadata for optimized queries
        *   **Fields:**
            *   `embedding: Vec<f32>` - The embedding vector
            *   `entity_id: u32` - Entity identifier
            *   `confidence: f32` - Confidence score
            *   `filter_score: f32` - Additional filter score
        *   **Methods:**
            *   `new(embedding, entity_id, confidence, filter_score)`: Constructor
    
    *   `EntityEmbedding`
        *   **Description:** Entity embedding with timestamp metadata
        *   **Fields:**
            *   `entity_id: u32` - Entity identifier
            *   `embedding: Vec<f32>` - The embedding vector
            *   `timestamp: u64` - Unix timestamp
        *   **Methods:**
            *   `new(entity_id, embedding)`: Constructor with auto timestamp
            *   `to_vector()`: Extract embedding vector
    
    *   `ThroughputResult`
        *   **Description:** Performance measurement results
        *   **Fields:**
            *   `operations_per_second: f64`
            *   `total_operations: u64`
            *   `duration_ms: u128`
            *   `memory_usage_mb: f64`
        *   **Methods:**
            *   `new(operations_per_second, total_operations, duration_ms)`: Constructor

*   **Type Aliases:**
    *   `EmbeddingVector = Vec<f32>` - Standard embedding vector type

*   **Helper Functions:**
    *   `entity_embeddings_to_vectors(entity_embeddings)`: Convert Vec<EntityEmbedding> to Vec<EmbeddingVector>
    *   `entity_embeddings_slice_to_vectors(entity_embeddings)`: Convert &[EntityEmbedding] to Vec<EmbeddingVector>

### `quantizer.rs`

*   **Purpose:** Product Quantization implementation for embedding compression
*   **Classes:**
    *   `CompressionStats`
        *   **Description:** Statistics about compression performance
        *   **Fields:** Memory usage, compression ratio, quality metrics
    
    *   `QuantizedEmbeddingStorage`
        *   **Description:** Storage for quantized embeddings
        *   **Methods:**
            *   `new()`: Create new storage
            *   `add_quantized(entity, codes)`: Store quantized codes
            *   `get_quantized(entity)`: Retrieve quantized codes
            *   `build_search_index()`: Build index for faster search
            *   `memory_usage()`: Calculate memory usage
            *   `compression_ratio(original_dimension)`: Calculate compression ratio
    
    *   `ProductQuantizer`
        *   **Description:** Main Product Quantization engine
        *   **Key Properties:**
            *   `subvector_count` - Number of subvectors (M)
            *   `subvector_size` - Dimensions per subvector (D/M)
            *   `cluster_count` - Clusters per subvector (256 for u8)
            *   `codebooks` - Trained centroids
            *   `storage` - Quantized embedding storage
        *   **Methods:**
            *   `new(dimension, subvector_count)`: Create quantizer
            *   `new_optimized(dimension, target_compression)`: Auto-configure quantizer
            *   `train(embeddings, iterations)`: Train codebooks with k-means
            *   `train_adaptive(embeddings)`: Auto-adaptive training
            *   `encode(embedding)`: Quantize embedding to codes
            *   `decode(codes)`: Reconstruct embedding from codes
            *   `asymmetric_distance(query, codes)`: Compute query-to-code distance
            *   `batch_encode(embeddings)`: Batch quantization
            *   `batch_decode(codes)`: Batch reconstruction
            *   `store_quantized(entity, embedding)`: Store with entity key
            *   `quantized_similarity_search(query, k)`: Fast k-NN search
            *   `compute_reconstruction_error(embeddings)`: Measure quality

*   **Helper Functions:**
    *   `euclidean_distance(a, b)`: Scalar euclidean distance
    *   `simd::euclidean_distance_avx2(a, b)`: AVX2-accelerated distance (unsafe)

### `simd_search.rs`

*   **Purpose:** Ultra-fast SIMD-accelerated similarity search
*   **Classes:**
    *   `SIMDSimilaritySearch`
        *   **Description:** SIMD-optimized search engine with precomputed distance tables
        *   **Fields:**
            *   `embedding_dim: usize` - Embedding dimension
            *   `subvector_count: usize` - Number of subvectors
            *   `distance_table: Vec<f32>` - Precomputed distances
        *   **Methods:**
            *   `new(embedding_dim, subvector_count)`: Constructor
            *   `precompute_distances(query, codebooks)`: Build distance lookup table
            *   `batch_asymmetric_distances(codes_batch, results)`: AVX2 batch distances
            *   `batch_asymmetric_distances_scalar(codes_batch, results)`: Fallback scalar
            *   `compute_subvector_distance(a, b)`: Auto-dispatch distance
            *   `euclidean_distance_avx2(a, b)`: AVX2 distance (unsafe)
            *   `top_k_search(codes_batch, entity_ids, k)`: Find k nearest neighbors
    
    *   `BatchProcessor`
        *   **Description:** High-throughput batch processing coordinator
        *   **Fields:**
            *   `simd_search: SIMDSimilaritySearch`
            *   `batch_size: usize`
        *   **Methods:**
            *   `new(embedding_dim, subvector_count, batch_size)`: Constructor
            *   `precompute_distances(query, codebooks)`: Forward to SIMD search
            *   `process_batched_search(all_codes, all_entity_ids, k)`: Process in batches

### `similarity.rs`

*   **Purpose:** Distance and similarity metric implementations with SIMD acceleration
*   **Functions:**
    *   `cosine_similarity(a, b)`: Auto-dispatching cosine similarity
    *   `cosine_similarity_scalar(a, b)`: Scalar fallback implementation
    *   `euclidean_distance(a, b)`: Auto-dispatching euclidean distance
    *   `euclidean_distance_scalar(a, b)`: Scalar fallback implementation
    *   `manhattan_distance(a, b)`: L1 distance
    *   `dot_product(a, b)`: Inner product

*   **SIMD Module (`simd`):**
    *   **Functions:**
        *   `cosine_similarity_avx2(a, b)`: AVX2-optimized cosine similarity
        *   `cosine_similarity_sse41(a, b)`: SSE4.1 fallback
        *   `euclidean_distance_avx2(a, b)`: AVX2-optimized euclidean
        *   `euclidean_distance_sse41(a, b)`: SSE4.1 fallback
        *   `batch_cosine_similarity_avx2(query, embeddings, dimension, results)`: Batch compute
    *   **Helper Functions:**
        *   `horizontal_sum_avx2(vec)`: Efficient horizontal sum for AVX2
        *   `horizontal_sum_sse(vec)`: Efficient horizontal sum for SSE

### `store.rs`

*   **Purpose:** Core embedding storage with quantization
*   **Classes:**
    *   `EmbeddingStore`
        *   **Description:** Main storage interface for embeddings
        *   **Fields:**
            *   `quantizer: Arc<RwLock<ProductQuantizer>>` - Shared quantizer
            *   `quantized_bank: Vec<u8>` - Flat storage for quantized codes
            *   `dimension: usize` - Embedding dimension
            *   `subvector_count: usize` - Quantization parameter
        *   **Methods:**
            *   `new(dimension, subvector_count)`: Create store
            *   `store_embedding(embedding)`: Store and return offset
            *   `get_embedding(offset)`: Retrieve by offset
            *   `asymmetric_distance(query, offset)`: Compute distance
            *   `memory_usage()`: Total memory usage
            *   `dimension()`: Get dimension

### `store_compat.rs`

*   **Purpose:** Compatibility layer for performance testing with enhanced features
*   **Classes:**
    *   `EmbeddingStore` (compat version)
        *   **Description:** Extended storage with both quantized and original embeddings
        *   **Fields:**
            *   `quantizer: Arc<RwLock<ProductQuantizer>>` - Shared quantizer
            *   `embeddings: HashMap<EntityKey, Vec<f32>>` - Original embeddings
            *   `quantized_embeddings: HashMap<EntityKey, Vec<u8>>` - Quantized codes
            *   `dimension: usize` - Embedding dimension
            *   `quantization_enabled: bool` - Auto-quantization flag
            *   `auto_quantize_threshold: usize` - Trigger threshold
        *   **Methods:**
            *   `new(dimension)`: Basic constructor
            *   `new_with_quantization(dimension, target_compression)`: Optimized constructor
            *   `enable_quantization(threshold)`: Enable auto-quantization
            *   `add_embedding(entity, embedding)`: Add by string name
            *   `add_embedding_key(entity, embedding)`: Add by EntityKey
            *   `quantize_all_embeddings()`: Batch quantization
            *   `similarity_search(query, k)`: Hybrid search (quantized + original)
            *   `similarity_search_quantized(query, k)`: Quantized-only search
            *   `memory_stats()`: Detailed memory statistics
            *   `force_quantize()`: Force immediate quantization
    
    *   `MemoryStats`
        *   **Description:** Detailed memory usage breakdown
        *   **Fields:** Memory by component, compression metrics

## 5. Key Algorithms and Logic

### Product Quantization Algorithm
1. **Training Phase:**
   - Split high-dimensional vectors into M subvectors
   - Run k-means clustering on each subvector space independently
   - Store K centroids (codebook) for each subspace

2. **Encoding Phase:**
   - Split input vector into M subvectors
   - Find nearest centroid in each subspace
   - Store M indices (1 byte each for 256 centroids)

3. **Decoding Phase:**
   - Lookup centroids by indices
   - Concatenate to reconstruct approximation

### SIMD Optimization Strategy
1. **AVX2 Processing:** Process 8 float32 values per instruction
2. **SSE4.1 Fallback:** Process 4 float32 values per instruction
3. **Auto-dispatch:** Runtime CPU feature detection
4. **Batch Processing:** Amortize overhead across multiple vectors

### Distance Table Precomputation
1. For query vector q, compute distance to all codebook centroids
2. Store in lookup table indexed by (subspace, cluster_id)
3. Asymmetric distance = sum of table lookups (O(M) instead of O(D))

## 6. Dependencies

*   **Internal:**
    *   `crate::error::{GraphError, Result}` - Error handling types
    *   `crate::core::types::EntityKey` - Entity identification
    *   `crate::core::entity_compat::SimilarityResult` - Search results

*   **External:**
    *   `parking_lot` - High-performance synchronization primitives
    *   `std::arch::x86_64` - SIMD intrinsics for x86-64
    *   `std::collections::HashMap` - Key-value storage
    *   `std::sync::Arc` - Thread-safe reference counting

## 7. Performance Characteristics

*   **Compression:** Up to 32x reduction in memory usage
*   **Search Speed:** 8x throughput with AVX2 vs scalar
*   **Training:** O(n * k * m * iter) complexity
*   **Query:** O(m) with precomputed tables vs O(d) naive
*   **Memory:** O(m * k * d/m) for codebooks + O(n * m) for codes

## 8. Error Handling

All methods return `Result<T>` with potential errors:
*   `GraphError::InvalidEmbeddingDimension` - Dimension mismatch
*   `GraphError::IndexCorruption` - Invalid offset access
*   Training errors for empty datasets

## 9. Thread Safety

*   `ProductQuantizer` wrapped in `Arc<RwLock<>>` for concurrent access
*   Read-heavy workloads benefit from `parking_lot::RwLock`
*   Batch operations minimize lock contention

## 10. Usage Patterns

### Basic Usage:
```rust
// Create and train quantizer
let mut pq = ProductQuantizer::new(128, 8)?;
pq.train(&training_embeddings, 50)?;

// Store embeddings
let mut store = EmbeddingStore::new(128, 8)?;
let offset = store.store_embedding(&embedding)?;

// Search
let results = store.similarity_search(&query, 10);
```

### High-Performance Usage:
```rust
// Batch processing with SIMD
let processor = BatchProcessor::new(128, 8, 1000);
processor.precompute_distances(&query, &codebooks)?;
let results = processor.process_batched_search(&all_codes, &entity_ids, 100)?;
```