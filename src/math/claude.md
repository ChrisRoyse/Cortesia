# Directory Overview: Math Module

## 1. High-Level Summary

The `math` module provides comprehensive mathematical operations for a distributed knowledge graph system. It implements advanced similarity calculations, graph algorithms, and distributed computation capabilities across federated databases. The module serves as the mathematical foundation for knowledge graph analysis, enabling similarity comparisons, graph traversal, centrality calculations, and distributed processing across multiple database instances.

## 2. Tech Stack

* **Language:** Rust
* **Frameworks:** Tokio (async runtime), Serde (serialization)
* **Libraries:** 
  - `std::collections` (HashMap, HashSet, VecDeque, BinaryHeap)
  - `tokio::sync` (RwLock for async synchronization)
  - `serde::{Deserialize, Serialize}` for data serialization
* **Database:** Federated/distributed database system with custom `DatabaseId` types
* **Dependencies:** 
  - `crate::error` (custom error handling)
  - `crate::federation` (federated entity management)

## 3. Directory Structure

```
math/
├── mod.rs                    # Main coordinator and public API
├── types.rs                  # Comprehensive type definitions
├── similarity.rs             # Similarity metrics and calculations
├── graph_algorithms.rs       # Graph traversal and analysis algorithms
└── distributed_math.rs       # Distributed computation across databases
```

## 4. File Breakdown

### `mod.rs` - Main Mathematical Operations Coordinator

* **Purpose:** Central coordinator for all mathematical operations, provides unified API
* **Main Struct:** `MathEngine`
  * **Description:** Orchestrates similarity, graph, and distributed mathematical operations
  * **Fields:**
    * `similarity_engine: SimilarityEngine` - Handles similarity calculations
    * `graph_algorithm_engine: GraphAlgorithms` - Manages graph algorithms
    * `distributed_engine: DistributedMathEngine` - Coordinates distributed operations
  * **Methods:**
    * `new() -> Result<Self>` - Initialize the mathematical engine
    * `cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32>` - Calculate cosine similarity
    * `euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32>` - Calculate Euclidean distance
    * `jaccard_similarity<T>(&self, set1: &[T], set2: &[T]) -> f32` - Calculate Jaccard similarity for sets
    * `graph_algorithms(&self) -> &GraphAlgorithms` - Access graph algorithms
    * `distributed_engine(&self) -> &DistributedMathEngine` - Access distributed operations

### `types.rs` - Comprehensive Type Definitions

* **Purpose:** Defines all data structures, enums, and result types for mathematical operations
* **Key Types:**

#### Core Result Types
* **`MathematicalResult`** - Wrapper for all mathematical operation results
  * `operation_type: String` - Type of operation performed
  * `execution_time_ms: u64` - Execution time in milliseconds
  * `databases_involved: Vec<DatabaseId>` - Databases used in computation
  * `result_data: MathResultData` - Actual result data
  * `metadata: MathMetadata` - Operation metadata

* **`MathResultData`** - Enum for different result data types
  * `Scalar { value: f64, unit: Option<String> }` - Single numerical result
  * `Vector { values: Vec<f64>, labels: Option<Vec<String>> }` - Vector results
  * `Matrix { values: Vec<Vec<f64>>, row_labels, col_labels }` - Matrix results
  * `Graph { edges, node_properties }` - Graph structure results
  * `Rankings { rankings: Vec<(FederatedEntityKey, f64)>, ranking_type }` - Ranking results
  * `Similarity { pairs: Vec<SimilarityPair>, similarity_type }` - Similarity results

#### Graph Algorithm Results
* **`PageRankResult`** - PageRank calculation results
* **`ShortestPathResult`** - Shortest path computation results  
* **`CentralityResult`** - Centrality measure results
* **`ClusteringResult`** - Clustering analysis results

#### Error Handling
* **`MathError`** - Specific mathematical operation errors
  * `DimensionMismatch` - Vector dimension mismatches
  * `InsufficientData` - Not enough data for computation
  * `ConvergenceFailure` - Algorithm convergence failures
  * `InvalidParameters` - Invalid parameter values
  * `DatabaseUnavailable` - Database connection issues
  * `ComputationTimeout` - Operation timeouts

### `similarity.rs` - Advanced Similarity Calculations

* **Purpose:** Implements comprehensive similarity metrics for knowledge graph analysis
* **Main Struct:** `SimilarityEngine`
  * **Description:** Provides various similarity calculation methods with caching
  * **Fields:**
    * `similarity_cache: HashMap<String, f32>` - Results cache
    * `config: SimilarityConfig` - Configuration parameters
  * **Key Methods:**
    * `cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32>` - Cosine similarity calculation
    * `euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32>` - Euclidean distance
    * `manhattan_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32>` - Manhattan distance
    * `jaccard_similarity<T>(&self, set1: &[T], set2: &[T]) -> f32` - Jaccard index for sets
    * `semantic_similarity(&self, embedding1: &[f32], embedding2: &[f32], text1: &str, text2: &str) -> Result<f32>` - Multi-factor semantic similarity
    * `textual_similarity(&self, text1: &str, text2: &str) -> f32` - Text-based similarity
    * `levenshtein_similarity(&self, s1: &str, s2: &str) -> f32` - Edit distance similarity
    * `structural_similarity(&self, neighbors1: &[u32], neighbors2: &[u32], weights1: Option<&[f32]>, weights2: Option<&[f32]>) -> f32` - Graph structural similarity
    * `graph_similarity(&self, entity1: u32, entity2: u32, graph_neighbors: &HashMap<u32, Vec<u32>>, max_hops: usize) -> f32` - Multi-hop graph similarity

#### Advanced Features
* **Configuration System:** `SimilarityConfig` with configurable weights for different similarity components
* **Caching:** Built-in caching with TTL for expensive computations
* **Multiple Metrics:** Support for 8+ different similarity metrics
* **Graph-Aware:** Considers graph structure and multi-hop relationships
* **Text Processing:** Advanced text similarity with n-grams, word overlap, and edit distance

### `graph_algorithms.rs` - Graph Analysis Algorithms

* **Purpose:** Implements fundamental graph algorithms for knowledge graph analysis
* **Main Struct:** `GraphAlgorithms`
  * **Description:** Provides graph traversal, shortest path, and centrality algorithms
  * **Fields:**
    * `algorithm_cache: HashMap<String, AlgorithmResult>` - Algorithm results cache
  * **Key Methods:**
    * `bfs(&self, graph: &AdjacencyList, start: u32) -> Result<BfsResult>` - Breadth-first search
    * `dfs(&self, graph: &AdjacencyList, start: u32) -> Result<DfsResult>` - Depth-first search
    * `dijkstra(&self, graph: &WeightedAdjacencyList, start: u32) -> Result<DijkstraResult>` - Shortest path algorithm
    * `a_star(&self, graph: &WeightedAdjacencyList, start: u32, goal: u32, heuristic: &dyn Fn(u32, u32) -> f32) -> Result<AStarResult>` - A* pathfinding
    * `pagerank(&self, graph: &AdjacencyList, iterations: usize, damping: f32) -> Result<PageRankResult>` - PageRank algorithm
    * `tarjan_scc(&self, graph: &AdjacencyList) -> Result<TarjanResult>` - Strongly connected components
    * `betweenness_centrality(&self, graph: &AdjacencyList) -> Result<CentralityResult>` - Betweenness centrality
    * `closeness_centrality(&self, graph: &AdjacencyList) -> Result<CentralityResult>` - Closeness centrality
    * `find_cycles(&self, graph: &AdjacencyList) -> Result<CycleResult>` - Cycle detection

#### Graph Representations
* **`AdjacencyList`** - `HashMap<u32, Vec<u32>>` for unweighted graphs
* **`WeightedAdjacencyList`** - `HashMap<u32, Vec<(u32, f32)>>` for weighted graphs

#### Algorithm Results
* **`BfsResult`** - BFS traversal results with distances and visit order
* **`DfsResult`** - DFS traversal with discovery/finish times
* **`DijkstraResult`** - Shortest distances and parent relationships
* **`PageRankResult`** - Node rankings and convergence information
* **`CentralityResult`** - Centrality scores and node rankings

### `distributed_math.rs` - Distributed Mathematical Operations

* **Purpose:** Enables mathematical operations across multiple federated databases
* **Main Struct:** `DistributedMathEngine`
  * **Description:** Coordinates distributed mathematical computations with load balancing
  * **Fields:**
    * `database_connections: Arc<RwLock<HashMap<DatabaseId, DatabaseConnection>>>` - Database connections
    * `node_assignments: Arc<RwLock<HashMap<String, Vec<DatabaseId>>>>` - Computation node assignments
    * `computation_cache: Arc<RwLock<HashMap<String, CachedResult>>>` - Distributed results cache
    * `load_balancer: LoadBalancer` - Load balancing strategy
  * **Key Methods:**
    * `register_database(&self, db_id: DatabaseId, connection: DatabaseConnection) -> Result<()>` - Register database for computation
    * `distributed_similarity(&self, entity_pairs: Vec<(DatabaseId, u32, DatabaseId, u32)>, similarity_method: SimilarityMethod) -> Result<DistributedSimilarityResult>` - Cross-database similarity computation
    * `distributed_graph_analysis(&self, analysis_type: GraphAnalysisType, target_databases: Vec<DatabaseId>) -> Result<DistributedGraphAnalysisResult>` - Distributed graph analysis
    * `distributed_clustering(&self, clustering_params: ClusteringParameters, target_databases: Vec<DatabaseId>) -> Result<DistributedClusteringResult>` - Multi-database clustering
    * `distributed_pagerank(&self, damping_factor: f32, max_iterations: usize, convergence_threshold: f32, target_databases: Vec<DatabaseId>) -> Result<DistributedPageRankResult>` - Distributed PageRank calculation
    * `distributed_centrality(&self, centrality_type: CentralityType, target_databases: Vec<DatabaseId>) -> Result<DistributedCentralityResult>` - Cross-database centrality

#### Distributed Features
* **Load Balancing:** `LoadBalancer` for optimal task distribution
* **Async Operations:** Full async/await support for non-blocking operations
* **Result Aggregation:** Intelligent aggregation of results across databases
* **Fault Tolerance:** Error handling for database unavailability
* **Caching:** Distributed computation results caching with expiry

## 5. Key Variables and Logic

### Similarity Calculations
* **Cosine Similarity:** `dot_product / (norm1 * norm2)` - Measures vector angle similarity
* **Euclidean Distance:** `sqrt(sum((a - b)^2))` - Geometric distance in n-dimensional space
* **Jaccard Index:** `|intersection| / |union|` - Set overlap similarity
* **Levenshtein Distance:** Dynamic programming approach for edit distance

### Graph Algorithms
* **PageRank:** Iterative power method with damping factor (typically 0.85)
* **Betweenness Centrality:** Counts shortest paths passing through each node
* **A* Search:** `f(n) = g(n) + h(n)` where g is cost and h is heuristic

### Distributed Computing
* **Task Distribution:** Groups computations by database to minimize network overhead
* **Result Aggregation:** Combines partial results using appropriate mathematical operations
* **Load Balancing:** Simple round-robin selection with capability awareness

## 6. Dependencies

### Internal Dependencies
* **`crate::error`** - Custom error types (`GraphError`, `Result`)
* **`crate::federation`** - Federated entity management (`DatabaseId`, `FederatedEntityKey`)

### External Dependencies
* **`std::collections`** - HashMap, HashSet, VecDeque, BinaryHeap for data structures
* **`std::sync::Arc`** - Atomic reference counting for shared ownership
* **`tokio::sync::RwLock`** - Async read-write locks for concurrent access
* **`serde`** - Serialization/deserialization for distributed operations

## 7. Performance Considerations

### Caching Strategy
* **Similarity Cache:** Stores computed similarity scores with configurable TTL
* **Algorithm Cache:** Caches expensive graph algorithm results
* **Distributed Cache:** Caches results across database boundaries

### Optimization Features
* **Parallel Processing:** Async operations for concurrent database queries
* **Batch Operations:** Groups similar computations for efficiency
* **Memory Management:** Careful use of references to avoid unnecessary copying
* **Numerical Stability:** Handles edge cases like zero vectors and overflow conditions

## 8. Error Handling

### Custom Error Types
* **`MathError`** - Mathematical operation specific errors
* **`GraphError`** - Graph algorithm and structure errors
* **Distributed Errors** - Database connectivity and timeout handling

### Error Recovery
* **Graceful Degradation:** Operations continue with available databases
* **Retry Logic:** Built-in retry for transient failures
* **Fallback Strategies:** Alternative computation methods when primary fails

## 9. Testing Coverage

The module includes comprehensive test coverage with:
* **Unit Tests:** Each algorithm and similarity metric
* **Property Tests:** Mathematical invariants and properties
* **Edge Case Tests:** Zero vectors, empty graphs, numerical stability
* **Integration Tests:** Cross-component functionality
* **Performance Tests:** Algorithmic complexity validation

## 10. Usage Patterns

### Basic Similarity Calculation
```rust
let engine = MathEngine::new()?;
let similarity = engine.cosine_similarity(&vec1, &vec2)?;
```

### Graph Analysis
```rust
let graph_engine = engine.graph_algorithms();
let pagerank_result = graph_engine.pagerank(&adjacency_list, 50, 0.85)?;
```

### Distributed Operations
```rust
let distributed_engine = engine.distributed_engine();
let result = distributed_engine.distributed_similarity(entity_pairs, SimilarityMethod::Cosine).await?;
```

This math module provides a comprehensive foundation for mathematical operations in distributed knowledge graph systems, with emphasis on performance, scalability, and mathematical correctness.