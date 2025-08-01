# Directory Overview: GPU Acceleration Module

## 1. High-Level Summary

The `src/gpu` directory contains GPU acceleration capabilities for graph operations within the LLMKG (Lightning-fast Knowledge Graph) system. This module provides a trait-based architecture that enables GPU-accelerated graph computations when available, with a high-performance CPU fallback implementation using parallel processing. The current implementation focuses on three core graph operations: parallel traversal, batch similarity computation, and parallel shortest path finding.

The module is designed with future CUDA support in mind, containing comprehensive documentation and placeholder implementations that outline the requirements for true GPU acceleration using NVIDIA's CUDA platform.

## 2. Tech Stack

- **Language:** Rust (2021 edition)
- **Parallel Processing:** Rayon (1.8) for CPU parallelization
- **Collections:** std::collections (HashMap, HashSet, VecDeque, BinaryHeap)
- **Concurrency:** std::sync (Arc, Mutex) for thread-safe operations
- **Error Handling:** Custom GraphError enum via crate::error module, thiserror crate
- **Feature Gates:** Conditional compilation for CUDA support
- **Future Dependencies (CUDA):** Planned support for cust/rustacuda, cuBLAS

## 3. Directory Structure

```
gpu/
├── mod.rs          # Main module with GpuAccelerator trait and CPU implementation
└── cuda.rs         # CUDA-specific implementation (currently placeholder)
```

## 4. File Breakdown

### `mod.rs`

- **Purpose:** Core module that defines the GPU acceleration interface and provides a high-performance CPU fallback implementation.
- **Key Components:**

#### Trait: `GpuAccelerator`
- **Description:** Defines the interface for GPU-accelerated graph operations
- **Methods:**
  - `parallel_traversal(&self, start_nodes: &[u32], max_depth: u32) -> Result<Vec<u32>, String>`: Performs parallel graph traversal from multiple starting nodes
  - `batch_similarity(&self, embeddings: &[Vec<f32>], query: &[f32]) -> Result<Vec<f32>, String>`: Computes cosine similarity between query and multiple embeddings in parallel
  - `parallel_shortest_paths(&self, sources: &[u32], targets: &[u32]) -> Result<Vec<Option<Vec<u32>>>, String>`: Finds shortest paths between multiple source-target pairs

#### Struct: `CpuGraphProcessor`
- **Description:** CPU-based implementation of GpuAccelerator trait using Rayon for parallelization
- **Properties:**
  - `graph: Option<Arc<HashMap<u32, Vec<u32>>>>`: Optional adjacency list representation for graph operations
- **Methods:**
  - `new() -> Self`: Creates new processor without graph data
  - `with_graph(graph: Arc<HashMap<u32, Vec<u32>>>) -> Self`: Creates processor with specific graph data

#### Helper Functions:
- **`cosine_similarity(a: &[f32], b: &[f32]) -> f32`**: Computes cosine similarity between two vectors with zero-norm handling
- **`dijkstra_shortest_path(graph: &HashMap<u32, Vec<u32>>, source: u32, target: u32) -> Option<Vec<u32>>`**: Implements Dijkstra's algorithm for single-source shortest paths

#### Internal Structures:
- **`State`**: Priority queue state for Dijkstra's algorithm with cost and node information, implementing Ord for min-heap behavior

### `cuda.rs`

- **Purpose:** Provides CUDA GPU acceleration support with comprehensive implementation roadmap and placeholder code.
- **Current State:** Not implemented - contains detailed documentation for future CUDA integration

#### Struct: `CudaGraphProcessor`
- **Description:** CUDA-accelerated graph processor (placeholder implementation)
- **Feature Gates:** 
  - `#[cfg(feature = "cuda")]`: Actual CUDA implementation (returns NotImplemented error)
  - `#[cfg(not(feature = "cuda"))]`: Returns FeatureNotEnabled error
- **Methods:**
  - `new() -> Result<Self>`: Constructor that currently returns appropriate error based on feature availability

#### Implementation Requirements Documentation:
- **Dependencies:** cust/rustacuda, cc build dependencies
- **CUDA Kernels:** Parallel BFS/DFS, matrix operations, parallel shortest paths
- **Memory Management:** Device allocation, host-to-device transfer, CSR format
- **Example Kernel Structure:** Documented parallel BFS kernel signature

## 5. Error Handling

The module integrates with the project's comprehensive error system defined in `src/error.rs`:

- **GraphError Enum:** Comprehensive error types including:
  - `FeatureNotEnabled(String)`: When CUDA feature is disabled
  - `NotImplemented(String)`: For unimplemented CUDA functionality
  - Entity and relationship errors
  - Performance and resource errors
  - Brain-inspired graph specific errors
- **Result Type:** `type Result<T> = std::result::Result<T, GraphError>`

## 6. Key Algorithms and Logic

### Parallel Graph Traversal (CPU Implementation)
- **Algorithm:** Multi-source BFS using Rayon parallel iterators
- **Approach:** Each start node processed in parallel with local visited sets merged into global result
- **Complexity:** O((V + E) / P) where P is the number of CPU cores
- **Thread Safety:** Uses Arc<Mutex<HashSet>> for visited node synchronization

### Batch Similarity Computation
- **Algorithm:** Parallel cosine similarity computation using Rayon
- **Formula:** dot_product / (norm_a * norm_b) with zero-norm protection
- **Optimization:** Vectorized operations for each embedding-query pair

### Parallel Shortest Paths
- **Algorithm:** Dijkstra's algorithm with BinaryHeap-based priority queue
- **Features:** Path reconstruction, early termination, parallel execution across multiple source-target pairs
- **Edge Weights:** Currently assumes unit weights (cost = 1)

## 7. Dependencies

### Internal Dependencies:
- `crate::error`: Custom error types and Result alias
- Standard library: Collections, sync primitives, ordering traits

### External Dependencies:
- `rayon = "1.8"`: Data parallelism for CPU operations
- `thiserror = "1.0"`: Error derivation macros
- `anyhow = "1.0"`: Error context and conversion

### Future Dependencies (CUDA):
- `cust` or `rustacuda`: Rust CUDA bindings
- `cc`: Build script support for CUDA compilation
- NVIDIA CUDA Toolkit: Required for GPU operations

## 8. Feature Gates and Compilation

- **Default Build:** CPU-only implementation with CudaGraphProcessor returning FeatureNotEnabled
- **CUDA Feature:** `--features cuda` enables CUDA-specific code paths (currently returns NotImplemented)
- **Conditional Compilation:** Extensive use of `#[cfg(feature = "cuda")]` for feature separation

## 9. Performance Characteristics

### CPU Implementation:
- **Parallel Traversal:** Scales with CPU core count, efficient memory usage through local visited sets
- **Batch Similarity:** Embarrassingly parallel, optimal for large embedding batches
- **Shortest Paths:** Parallel across source-target pairs, efficient heap-based Dijkstra implementation

### Memory Usage:
- **Graph Storage:** HashMap-based adjacency lists, wrapped in Arc for safe sharing
- **Traversal:** Local HashSets per thread to minimize lock contention
- **Path Finding:** Minimal memory overhead with in-place heap operations

## 10. Future Implementation Roadmap

### CUDA Integration Plan:
1. **Memory Layout:** Convert graph to CSR (Compressed Sparse Row) format for GPU efficiency
2. **Kernel Development:** Implement parallel BFS, similarity computation, and shortest path kernels
3. **Memory Management:** Device memory pools, efficient host-device transfers
4. **Optimization:** Work-efficient algorithms, coalesced memory access patterns
5. **Integration:** Seamless fallback to CPU when GPU unavailable

### Potential Optimizations:
- **SIMD Instructions:** CPU vectorization for similarity computations
- **Cache Optimization:** Graph preprocessing for better memory locality
- **Streaming:** Overlap computation with memory transfers for large datasets

## 11. Usage Patterns

```rust
// Create CPU processor with graph data
let graph = Arc::new(/* adjacency list */);
let processor = CpuGraphProcessor::new().with_graph(graph);

// Parallel graph traversal
let visited = processor.parallel_traversal(&[1, 2, 3], 3)?;

// Batch similarity computation  
let similarities = processor.batch_similarity(&embeddings, &query_vector)?;

// Multiple shortest paths
let paths = processor.parallel_shortest_paths(&sources, &targets)?;
```

This module serves as a critical performance component of the LLMKG system, providing scalable graph operations that can leverage both multi-core CPUs and future GPU acceleration for maximum computational efficiency.