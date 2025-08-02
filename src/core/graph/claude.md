# Directory Overview: src/core/graph

## 1. High-Level Summary

This directory contains the core knowledge graph implementation for the LLMKG project. It provides a high-performance, modular graph database with embedding-based similarity search capabilities, relationship management, path finding algorithms, and advanced query functionality. The implementation includes multiple indexing strategies, performance optimizations, and a legacy compatibility layer.

## 2. Tech Stack

*   **Languages:** Rust
*   **Key Libraries:** 
    *   `parking_lot` - High-performance synchronization primitives (RwLock)
    *   `ahash` - Fast hashing for HashMaps
    *   `rayon` - Data parallelism for concurrent operations
    *   `serde_json` - JSON serialization for properties
*   **Storage Technologies:** 
    *   CSR (Compressed Sparse Row) graph format
    *   Multiple vector indices (HNSW, LSH, spatial, flat)
    *   Bloom filters for membership testing
    *   Arena allocation for memory management

## 3. Directory Structure

This directory is flat with all files at the same level:
- Core infrastructure files (mod.rs, graph_core.rs)
- Operation-specific implementations (entity_operations.rs, relationship_operations.rs)
- Algorithm implementations (path_finding.rs, similarity_search.rs)
- Advanced features (query_system.rs)
- Backward compatibility (compatibility.rs)

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module definition and public API exports
*   **Key Exports:**
    *   `KnowledgeGraph` - Main graph structure
    *   `MemoryUsage`, `MemoryBreakdown` - Memory statistics
    *   Performance constants (MAX_INSERTION_TIME, MAX_QUERY_TIME, MAX_SIMILARITY_SEARCH_TIME)
    *   Various statistics types (EntityStats, RelationshipStats, PathStats, etc.)

### `graph_core.rs`

*   **Purpose:** Core knowledge graph structure and fundamental operations
*   **Classes:**
    *   `KnowledgeGraph`
        *   **Description:** Main graph database structure with concurrent access support
        *   **Key Fields:**
            *   `arena: RwLock<GraphArena>` - Memory arena for entity storage
            *   `entity_store: RwLock<EntityStore>` - Entity metadata storage
            *   `graph: RwLock<CSRGraph>` - Compressed sparse row graph structure
            *   `embedding_bank: RwLock<Vec<u8>>` - Quantized embedding storage
            *   Multiple index structures (bloom_filter, spatial_index, flat_index, hnsw_index, lsh_index)
            *   `similarity_cache: RwLock<SimilarityCache>` - Query result cache
        *   **Methods:**
            *   `new_internal(embedding_dim: usize)`: Create new graph with specified embedding dimension
            *   `entity_count()`: Get total entity count
            *   `relationship_count()`: Get total relationship count
            *   `memory_usage()`: Get detailed memory usage statistics
            *   `validate_embedding_dimension(embedding: &[f32])`: Validate embedding dimensions
    *   `MemoryUsage`
        *   **Description:** Memory usage statistics container
        *   **Methods:**
            *   `total_bytes()`: Calculate total memory usage
            *   `bytes_per_entity(entity_count: usize)`: Calculate per-entity memory overhead
            *   `usage_breakdown()`: Get percentage breakdown by component
    *   `MemoryBreakdown`
        *   **Description:** Memory usage percentages by component

### `entity_operations.rs`

*   **Purpose:** Entity CRUD operations and batch processing
*   **Key Functions:**
    *   `insert_entity(id: u32, data: EntityData) -> Result<EntityKey>`
        *   **Description:** Insert single entity with validation
        *   **Parameters:** 
            *   `id`: Unique entity identifier
            *   `data`: Entity data including embedding and properties
        *   **Returns:** EntityKey for the inserted entity
    *   `insert_entities_batch(entities: Vec<(u32, EntityData)>) -> Result<Vec<EntityKey>>`
        *   **Description:** Batch insert with parallel validation for large datasets
        *   **Parameters:** Vector of (id, data) tuples
        *   **Returns:** Vector of EntityKeys
    *   `get_entity(key: EntityKey) -> Option<(EntityMeta, EntityData)>`
        *   **Description:** Retrieve entity by key
        *   **Returns:** Entity metadata and data if found
    *   `update_entity(key: EntityKey, data: EntityData) -> Result<()>`
        *   **Description:** Update existing entity data
    *   `remove_entity(key: EntityKey) -> Result<bool>`
        *   **Description:** Remove entity and clean up all references
        *   **Returns:** true if entity was removed
    *   `get_entity_embedding(key: EntityKey) -> Option<Vec<f32>>`
        *   **Description:** Get decoded embedding for entity
*   **Classes:**
    *   `EntityStats`
        *   **Description:** Entity statistics container
        *   **Methods:**
            *   `average_embedding_size()`: Calculate average embedding storage size

### `relationship_operations.rs`

*   **Purpose:** Relationship management and graph connectivity operations
*   **Key Functions:**
    *   `insert_relationship(relationship: Relationship) -> Result<()>`
        *   **Description:** Insert edge between entities with validation
        *   **Parameters:** Relationship struct with from/to keys, type, and weight
    *   `insert_relationships_batch(relationships: Vec<Relationship>) -> Result<()>`
        *   **Description:** Batch relationship insertion
    *   `get_neighbors(entity: EntityKey) -> Vec<EntityKey>`
        *   **Description:** Get all connected entities (bidirectional)
    *   `get_outgoing_neighbors(entity: EntityKey) -> Vec<EntityKey>`
        *   **Description:** Get entities this entity points to
    *   `get_incoming_neighbors(entity: EntityKey) -> Vec<EntityKey>`
        *   **Description:** Get entities pointing to this entity
    *   `get_relationship_weight(from: EntityKey, to: EntityKey) -> Option<f32>`
        *   **Description:** Get edge weight between entities
    *   `remove_relationship(from: EntityKey, to: EntityKey) -> Result<bool>`
        *   **Description:** Remove specific edge
    *   `get_entity_degree(entity: EntityKey) -> usize`
        *   **Description:** Count of connected entities
    *   `is_connected() -> bool`
        *   **Description:** Check if graph is fully connected
*   **Classes:**
    *   `RelationshipStats`
        *   **Description:** Relationship statistics
        *   **Methods:**
            *   `density(entity_count: usize) -> f64`: Calculate graph density
            *   `is_sparse(entity_count: usize) -> bool`: Check if density < 0.1
            *   `is_dense(entity_count: usize) -> bool`: Check if density > 0.7

### `path_finding.rs`

*   **Purpose:** Graph traversal and path finding algorithms
*   **Key Functions:**
    *   `find_path(source: EntityKey, target: EntityKey) -> Option<Vec<EntityKey>>`
        *   **Description:** BFS-based path finding
        *   **Returns:** Path as vector of EntityKeys if exists
    *   `find_all_paths(source: EntityKey, target: EntityKey, max_depth: usize) -> Vec<Vec<EntityKey>>`
        *   **Description:** Find all paths within depth limit
    *   `find_best_path(source: EntityKey, target: EntityKey, max_depth: usize) -> Option<(Vec<EntityKey>, f32)>`
        *   **Description:** Find path with maximum cumulative weight
    *   `find_weakest_path(source: EntityKey, target: EntityKey, max_depth: usize) -> Option<(Vec<EntityKey>, f32)>`
        *   **Description:** Find path with minimum cumulative weight
    *   `find_entities_within_distance(source: EntityKey, max_distance: usize) -> Vec<EntityKey>`
        *   **Description:** Find all reachable entities within hop distance
    *   `compute_diameter() -> Option<usize>`
        *   **Description:** Calculate longest shortest path in graph
    *   `compute_radius() -> Option<usize>`
        *   **Description:** Calculate minimum eccentricity
*   **Classes:**
    *   `PathStats`
        *   **Description:** Path analysis statistics
        *   **Methods:**
            *   `is_direct_connection()`: Check if distance is 1
            *   `is_close_connection()`: Check if distance <= 2
            *   `has_multiple_paths()`: Check if multiple paths exist

### `similarity_search.rs`

*   **Purpose:** Embedding-based similarity search with intelligent index selection
*   **Key Functions:**
    *   `similarity_search(query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>>`
        *   **Description:** Main similarity search with automatic index selection
        *   **Index Selection Logic:**
            *   entity_count < 1000: Use flat index
            *   k <= 10: Use HNSW for speed
            *   k >= entity_count/2: Use LSH for efficiency
            *   Otherwise: Use spatial index
    *   `similarity_search_parallel(query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>>`
        *   **Description:** Parallel search for datasets > 10,000 entities
    *   `similarity_search_threshold(query_embedding: &[f32], threshold: f32) -> Result<Vec<(EntityKey, f32)>>`
        *   **Description:** Find all entities above similarity threshold
    *   `similarity_search_filtered<F>(query_embedding: &[f32], k: usize, filter: F) -> Result<Vec<(EntityKey, f32)>>`
        *   **Description:** Search with custom entity filter
    *   `find_most_similar(query_embedding: &[f32]) -> Result<Option<(EntityKey, f32)>>`
        *   **Description:** Find single most similar entity
    *   `get_similarity_stats(query_embedding: &[f32]) -> Result<SimilarityStats>`
        *   **Description:** Calculate similarity distribution statistics
*   **Classes:**
    *   `SimilarityStats`
        *   **Description:** Similarity distribution statistics
        *   **Methods:**
            *   `is_well_distributed()`: Check if std_dev > 0.1 and range > 0.5
            *   `is_outlier_query()`: Check if max < 0.3 and mean < 0.2
            *   `is_common_query()`: Check if mean > 0.7 and median > 0.7

### `query_system.rs`

*   **Purpose:** Advanced query capabilities with context awareness and filtering
*   **Key Functions:**
    *   `query(query_embedding: &[f32], context_entities: &[ContextEntity], k: usize) -> Result<QueryResult>`
        *   **Description:** Complex query with context weighting
        *   **Scoring:** similarity * 0.7 + context_score * 0.3
    *   `query_filtered<F>(query_embedding: &[f32], k: usize, filter: F) -> Result<QueryResult>`
        *   **Description:** Query with custom entity filter
    *   `query_by_type(query_embedding: &[f32], entity_type: &str, k: usize) -> Result<QueryResult>`
        *   **Description:** Filter by entity type in properties
    *   `query_by_property_range(query_embedding: &[f32], property: &str, min_value: f32, max_value: f32, k: usize) -> Result<QueryResult>`
        *   **Description:** Filter by numeric property range
    *   `multi_step_query(query_embedding: &[f32], steps: usize, k: usize) -> Result<QueryResult>`
        *   **Description:** Expand query through neighbor traversal
    *   `query_connected_entities(query_embedding: &[f32], connected_to: EntityKey, k: usize) -> Result<QueryResult>`
        *   **Description:** Search only entities connected to specific entity
    *   `explain_query(query_embedding: &[f32], k: usize) -> Result<QueryExplanation>`
        *   **Description:** Debug query execution with detailed explanations
*   **Classes:**
    *   `AdvancedQueryResult`
        *   **Description:** Query result with performance metrics
    *   `QueryStats`
        *   **Description:** Query performance statistics
        *   **Methods:**
            *   `is_query_friendly()`: Check if graph is suitable for queries
            *   `graph_density()`: Calculate edge/node ratio
            *   `is_caching_effective()`: Check if cache hit rate > 30%
    *   `QueryExplanation`
        *   **Description:** Detailed query execution information
    *   `EntityExplanation`
        *   **Description:** Per-entity query relevance details

### `compatibility.rs`

*   **Purpose:** Legacy API support for backward compatibility
*   **Key Functions:**
    *   `insert_entity_with_text(id: u32, text: &str, properties: HashMap<String, String>) -> Result<EntityKey>`
        *   **Description:** Insert entity using text-based embedding generation
    *   `generate_text_embedding(text: &str) -> Vec<f32>`
        *   **Description:** Hash-based text to embedding conversion
    *   `similarity_search_by_text(query_text: &str, k: usize) -> Result<Vec<(u32, f32)>>`
        *   **Description:** Text-based similarity search
    *   `get_neighbors_by_id(entity_id: u32) -> Vec<u32>`
        *   **Description:** Get neighbors using numeric IDs
    *   `find_path_by_id(source_id: u32, target_id: u32) -> Option<Vec<u32>>`
        *   **Description:** Path finding with numeric IDs
    *   `export_entities() -> Vec<(u32, HashMap<String, String>, Vec<f32>)>`
        *   **Description:** Export all entities in legacy format
    *   `validate_graph_consistency() -> Vec<String>`
        *   **Description:** Check graph integrity and return issues

## 5. Key Algorithms and Data Structures

### Graph Storage
*   **CSRGraph**: Compressed Sparse Row format for efficient edge storage
*   **GraphArena**: Custom memory arena for entity allocation
*   **EntityStore**: Metadata storage with O(1) lookup

### Indexing Strategies
*   **BloomFilter**: Probabilistic membership testing
*   **FlatVectorIndex**: Brute-force search for small datasets
*   **HnswIndex**: Hierarchical Navigable Small World for fast approximate search
*   **LshIndex**: Locality-Sensitive Hashing for high-dimensional search
*   **SpatialIndex**: Space-partitioning index for geometric queries

### Algorithms
*   **BFS**: Breadth-first search for shortest paths
*   **DFS with backtracking**: All paths enumeration
*   **Cosine similarity**: Primary similarity metric
*   **Product quantization**: Embedding compression

## 6. Performance Considerations

### Timeouts
*   `MAX_INSERTION_TIME`: 10 seconds
*   `MAX_QUERY_TIME`: 100 milliseconds  
*   `MAX_SIMILARITY_SEARCH_TIME`: 50 milliseconds

### Optimizations
*   **Parallel validation**: For batch operations > threshold
*   **Query caching**: LRU cache for similarity searches
*   **Index selection**: Automatic based on data size and query parameters
*   **Read-write locks**: Fine-grained concurrency control

### Memory Management
*   **Arena allocation**: Reduces fragmentation
*   **Quantized embeddings**: 8-bit storage reduces memory
*   **Lazy loading**: Embeddings decoded on demand

## 7. Data Flow

### Entity Insertion Flow
1. Validate embedding dimension and text size
2. Allocate in arena
3. Store metadata in entity store
4. Quantize and store embedding
5. Update all indices (bloom, spatial, HNSW, LSH)
6. Update ID mapping

### Query Flow
1. Check cache for results
2. Select appropriate index
3. Perform similarity search
4. Apply context scoring (if provided)
5. Collect relationships
6. Cache results
7. Return with timing metrics

## 8. Dependencies

### Internal Dependencies
*   `crate::core::entity` - Entity storage types
*   `crate::core::memory` - Arena and epoch management
*   `crate::core::types` - Core type definitions
*   `crate::storage::*` - Various storage implementations
*   `crate::embedding::*` - Embedding operations
*   `crate::error` - Error types

### External Dependencies
*   Thread-safe operations via `parking_lot`
*   Parallel processing via `rayon`
*   Fast hashing via `ahash`

## 9. Error Handling

Common error types:
*   `InvalidEmbeddingDimension`: Embedding size mismatch
*   `EntityNotFound`: Entity ID doesn't exist
*   `EntityKeyNotFound`: Entity key invalid
*   `InvalidRelationshipWeight`: Weight outside [0, 1]
*   `InvalidInput`: General validation failures

## 10. Testing

Each module includes comprehensive test suites covering:
*   Normal operations
*   Edge cases (empty inputs, extreme values)
*   Error conditions
*   Concurrent access patterns
*   Performance characteristics
*   Legacy API compatibility