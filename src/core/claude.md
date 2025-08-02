# Directory Overview: LLMKG Core Module

## 1. High-Level Summary

The `src/core` directory contains the foundational components of LLMKG (Large Language Model Knowledge Graph), a sophisticated knowledge graph system that combines traditional graph data structures with brain-inspired computing patterns. The system is designed for high-performance knowledge storage and retrieval, optimized for LLM interaction with features like sparse distributed representation (SDR), neural-inspired activation patterns, and efficient triple storage.

## 2. Tech Stack

* **Languages:** Rust
* **Key External Libraries:**
  * `serde` - Serialization/deserialization
  * `slotmap` - Efficient entity key management
  * `ahash` - Fast hashing
  * `parking_lot` - High-performance synchronization primitives
  * `bumpalo` - Arena memory allocation
  * `rand` - Random number generation
  * `rayon` - Parallel computation

## 3. Directory Structure

* **Root Level:** Core types, entities, and fundamental components
* **`graph/`:** Traditional knowledge graph implementation with entity management, relationships, and queries
* **`brain_enhanced_graph/`:** Brain-inspired graph enhancements with activation patterns and concept formation

## 4. File Breakdown

### `mod.rs`

* **Purpose:** Module declarations and public API exports
* **Exports:** All sub-modules and key types like `BenchmarkResult`

### `types.rs`

* **Purpose:** Core type definitions used throughout the system
* **Key Types:**
  * `EntityKey`: Unique identifier for entities using slotmap
  * `AttributeValue`: Flexible value type supporting strings, numbers, booleans, arrays, objects, vectors, and null
  * `RelationshipType`: Enum defining directed, undirected, and weighted relationships
  * `TypeId`, `EmbeddingSize`, `EdgeWeight`: Type aliases for clarity

### `entity.rs`

* **Purpose:** Entity storage and management with efficient memory layout
* **Classes:**
  * `EntityStore`
    * **Description:** Manages entity metadata and properties with zero-copy storage
    * **Methods:**
      * `insert(key, data)`: Store new entity with properties
      * `get(key)`: Retrieve entity metadata
      * `get_properties(meta)`: Extract entity properties as string
      * `update_degree(key, delta)`: Update entity connection count
      * `memory_usage()`: Calculate memory consumption

### `entity_compat.rs`

* **Purpose:** Compatibility layer for entity operations across different graph implementations

### `interned_entity.rs`

* **Purpose:** String interning for efficient entity name storage and comparison

### `triple.rs`

* **Purpose:** Triple (Subject-Predicate-Object) storage for knowledge representation
* **Key Types:**
  * `Triple`: Core SPO structure with confidence and source tracking
  * `KnowledgeNode`: Container for triples or chunks with embeddings
  * `NodeType`: Enum (Triple, Chunk, Entity, Relationship)
  * `PredicateVocabulary`: Predicate normalization and consistency
* **Constants:**
  * `MAX_CHUNK_SIZE_BYTES`: 2048 (optimized for ~512 tokens)
  * `MAX_PREDICATE_LENGTH`: 64 (1-3 words optimal)
  * `MAX_ENTITY_NAME_LENGTH`: 128

### `knowledge_engine.rs`

* **Purpose:** Main engine for knowledge storage and retrieval with LLM optimization
* **Classes:**
  * `KnowledgeEngine`
    * **Description:** Ultra-fast triple storage maintaining <60 bytes per entity
    * **Key Components:**
      * Triple indexes (subject, predicate, object) for fast SPO queries
      * Embedding system with batch processing
      * Memory management with eviction policies
      * Pattern tracking for LLM optimization
    * **Methods:**
      * `store_triple(triple, embedding)`: Store with auto-embedding
      * `store_chunk(content, metadata)`: Store larger knowledge chunks
      * `query_triples(query)`: SPO pattern matching
      * `semantic_search(query, limit)`: Embedding-based search

### `knowledge_types.rs`

* **Purpose:** Type definitions for knowledge operations
* **Key Types:**
  * `MemoryStats`: Memory usage tracking
  * `TripleQuery`: Query patterns for SPO matching
  * `KnowledgeResult`: Query result containers
  * `EntityContext`: Context information for entities

### `knowledge_embedding.rs`

* **Purpose:** Embedding generation for semantic similarity
* **Classes:**
  * `EmbeddingGenerator`
    * **Description:** Creates vector embeddings for triples and text
    * **Methods:**
      * `generate_embedding_for_triple(triple)`: Triple-specific embeddings
      * `generate_embedding_for_text(text)`: Text embeddings

### `knowledge_extraction.rs`

* **Purpose:** Extract triples and entities from unstructured text
* **Classes:**
  * `TripleExtractor`
    * **Description:** NLP-based triple extraction
    * **Methods:**
      * `extract_triples(text)`: Extract SPO triples from text
      * `extract_entities(text)`: Identify entities in text

### `semantic_summary.rs`

* **Purpose:** Generate semantic summaries of knowledge subgraphs

### `memory.rs`

* **Purpose:** Memory arena management for efficient allocation
* **Classes:**
  * `GraphArena`
    * **Description:** Arena allocator with entity pooling
    * **Methods:**
      * `allocate_entity(data)`: Allocate new entity
      * `reset_generation()`: Clear arena for reuse
      * `memory_usage()`: Total memory consumption

### `parallel.rs`

* **Purpose:** Parallel processing utilities for graph operations
* **Key Features:**
  * Thread-safe batch processing
  * Parallel query execution
  * Work-stealing algorithms

### SDR (Sparse Distributed Representation) System

#### `sdr_types.rs`
* **Purpose:** Type definitions for SDR system
* **Key Types:**
  * `SDR`: Sparse distributed representation
  * `SDRConfig`: Configuration parameters
  * `SDRPattern`: Pattern recognition types

#### `sdr_core.rs`
* **Purpose:** Core SDR operations
* **Key Functions:**
  * `from_dense_vector(vector, config)`: Convert dense to sparse
  * `overlap(other)`: Calculate SDR overlap
  * `jaccard_similarity(other)`: Jaccard similarity metric
  * `cosine_similarity(other)`: Cosine similarity for binary vectors

#### `sdr_index.rs`
* **Purpose:** Indexing system for fast SDR lookups

#### `sdr_storage.rs`
* **Purpose:** Efficient storage of SDR representations

### Brain-Inspired Activation System

#### `activation_config.rs`
* **Purpose:** Configuration for neural activation patterns

#### `activation_engine.rs`
* **Purpose:** Main activation propagation engine
* **Key Features:**
  * Spreading activation algorithms
  * Attention mechanisms
  * Decay and reinforcement

#### `activation_processors.rs`
* **Purpose:** Specialized processors for different activation types

#### `brain_types.rs`
* **Purpose:** Type definitions for brain-inspired computing

### Zero-Copy System

#### `zero_copy_types.rs`
* **Purpose:** Types for zero-copy operations
* **Exports:** `BenchmarkResult` for performance metrics

#### `zero_copy_engine.rs`
* **Purpose:** Zero-copy data access patterns for performance

## 5. Brain Enhanced Graph Subsystem (`brain_enhanced_graph/`)

### `brain_graph_core.rs`
* **Purpose:** Main brain-enhanced knowledge graph implementation
* **Classes:**
  * `BrainEnhancedKnowledgeGraph`
    * **Description:** Graph with neural processing capabilities
    * **Key Features:**
      * Activation spreading
      * Concept formation
      * Adaptive learning
      * Attention mechanisms

### `brain_entity_manager.rs`
* **Purpose:** Entity management with brain-inspired features
* **Key Features:**
  * Entity activation levels
  * Concept membership
  * Statistical tracking

### `brain_query_engine.rs`
* **Purpose:** Query engine with neural-inspired search
* **Features:**
  * Attention-based queries
  * Activation-guided search
  * Concept-aware retrieval

### `brain_relationship_manager.rs`
* **Purpose:** Relationship management with weight adaptation
* **Key Types:**
  * `RelationshipPattern`: Pattern detection
  * `WeightDistribution`: Weight statistics

### `brain_concept_ops.rs`
* **Purpose:** Concept formation and manipulation
* **Key Types:**
  * `EntityRole`: Role classification in concepts
  * `SplitCriteria`: Criteria for concept splitting

### `brain_analytics.rs`
* **Purpose:** Analytics for brain-enhanced operations
* **Key Types:**
  * `ConceptUsageStats`: Concept utilization metrics
  * `GraphPatternAnalysis`: Pattern detection results

## 6. Traditional Graph Subsystem (`graph/`)

### `graph_core.rs`
* **Purpose:** Core knowledge graph implementation
* **Classes:**
  * `KnowledgeGraph`
    * **Description:** Traditional graph with optimized performance
    * **Performance Constraints:**
      * `MAX_INSERTION_TIME`: Performance bounds
      * `MAX_QUERY_TIME`: Query performance limits
      * `MAX_SIMILARITY_SEARCH_TIME`: Search time limits

### `entity_operations.rs`
* **Purpose:** CRUD operations for entities
* **Key Types:**
  * `EntityStats`: Entity statistics

### `relationship_operations.rs`
* **Purpose:** Relationship management
* **Key Types:**
  * `RelationshipStats`: Relationship statistics

### `path_finding.rs`
* **Purpose:** Graph traversal and pathfinding algorithms
* **Features:**
  * Shortest path algorithms
  * Subgraph extraction
  * Path statistics

### `similarity_search.rs`
* **Purpose:** Vector similarity search
* **Key Types:**
  * `SimilarityStats`: Search performance metrics

### `query_system.rs`
* **Purpose:** Advanced query capabilities
* **Key Types:**
  * `QueryStats`: Query performance tracking
  * `QueryExplanation`: Query execution details
  * `AdvancedQueryResult`: Rich query results

## 7. Key Algorithms and Logic

* **Triple Storage:** Optimized for <60 bytes per entity with dynamic indexing
* **SDR Generation:** Top-k sparse encoding from dense vectors
* **Activation Spreading:** Neural-inspired information propagation
* **Memory Management:** Arena allocation with generation-based cleanup
* **Pattern Recognition:** Frequency tracking for LLM optimization
* **Concept Formation:** Clustering based on activation patterns

## 8. Dependencies

* **Internal:** 
  * `crate::error` - Error handling
  * `crate::embedding` - Embedding generation and search
* **External:**
  * Thread-safe data structures (`parking_lot`)
  * High-performance collections (`ahash`)
  * Memory efficiency (`bumpalo`, `slotmap`)
  * Parallel processing (`rayon`)

## 9. Performance Considerations

* Zero-copy operations for string data
* Arena allocation for reduced fragmentation
* Parallel processing for batch operations
* Sparse representations for memory efficiency
* Index-based lookups for O(1) access patterns
* Configurable eviction policies for bounded memory

## 10. Usage Patterns

The core module provides low-level primitives that higher-level modules build upon:

1. **Knowledge Storage:** Use `KnowledgeEngine` for triple/chunk storage
2. **Graph Operations:** Use `KnowledgeGraph` for traditional graph algorithms
3. **Brain Processing:** Use `BrainEnhancedKnowledgeGraph` for neural-inspired operations
4. **Memory Management:** `GraphArena` provides efficient allocation
5. **Parallel Processing:** Utilities in `parallel.rs` for batch operations

The system is designed for high concurrency with read-write locks and atomic operations throughout.