# Directory Overview: LLMKG Query Module

## 1. High-Level Summary

The `src/query` directory contains the advanced query processing subsystem for LLMKG (Large Language Model Knowledge Graph). This module implements sophisticated query strategies including Graph RAG (Retrieval-Augmented Generation), hierarchical clustering, query optimization, and a two-tier query system that combines global and local search strategies. The system is designed to provide efficient knowledge retrieval for LLM interactions while maintaining sub-millisecond response times for critical operations.

## 2. Tech Stack

* **Languages:** Rust
* **Key External Libraries:**
  * `tokio` - Asynchronous runtime and concurrency
  * `chrono` - Date and time handling for caching
  * `std::collections` - HashMap, HashSet for data structures
  * `std::sync::Arc` - Atomic reference counting for shared data
  * `parking_lot::RwLock` - Read-write locks for concurrent access
* **Internal Dependencies:**
  * `crate::core::graph::KnowledgeGraph` - Core graph implementation
  * `crate::core::types` - Entity keys, relationships, and core types
  * `crate::error` - Error handling types
  * `crate::text::TextCompressor` - Text compression utilities

## 3. Directory Structure

The directory is flat with six main files, each serving a distinct query processing function:

* **Query Management:** `mod.rs` - Core types and module exports
* **Clustering:** `clustering.rs` - Hierarchical community detection
* **Optimization:** `optimizer.rs` - Performance monitoring and optimization
* **RAG System:** `rag.rs` - Graph RAG for context retrieval
* **Summarization:** `summarization.rs` - Fast community summarization
* **Two-Tier System:** `two_tier.rs` - Global/local query coordination

## 4. File Breakdown

### `mod.rs`

* **Purpose:** Module declarations and core query data structures
* **Key Structures:**
  * `RealtimeQuery`: Real-time query configuration with similarity thresholds
  * `PartitionedGraphRAG`: Distributed querying across graph partitions
* **Enums:**
  * `PartitionStrategy`: ByNodeId, ByEntityType, ByEmbeddingCluster, Custom
  * `MergeStrategy`: UnionAll, IntersectAll, RankByScore, Custom
* **Key Methods:**
  * `RealtimeQuery::new(query_text)`: Create query with default parameters
  * `RealtimeQuery::with_initial_nodes(nodes)`: Set starting nodes
  * `PartitionedGraphRAG::add_partition(graph)`: Add graph partition

### `clustering.rs`

* **Purpose:** Hierarchical clustering implementation using the Leiden algorithm for community detection
* **Classes:**
  * `HierarchicalClusterer`
    * **Description:** Multi-level clustering with configurable resolution levels
    * **Key Fields:**
      * `leiden_algorithm`: Core clustering implementation
      * `max_levels`: Maximum hierarchy depth
      * `resolution_levels`: [0.1, 0.5, 1.0, 2.0, 5.0] for different granularities
    * **Methods:**
      * `cluster_graph(graph)`: Perform hierarchical clustering asynchronously
      * `extract_entities(graph)`: Get all entity IDs from graph
      * `build_adjacency_matrix(entities, graph)`: Create adjacency matrix
      * `build_hierarchy_relationships(hierarchy)`: Link parent-child communities
  * `LeidenClustering`
    * **Description:** Core Leiden algorithm implementation
    * **Configuration:**
      * `max_iterations`: 100 (convergence limit)
      * `min_improvement`: 0.001 (convergence threshold)
    * **Methods:**
      * `cluster(adjacency, resolution, min_size)`: Main clustering algorithm
      * `calculate_modularity_gain(...)`: Compute community assignment benefit
      * `move_node(...)`: Transfer node between communities
  * `AdjacencyMatrix`
    * **Description:** Efficient matrix representation for clustering
    * **Methods:**
      * `set_edge(from, to, weight)`: Add undirected edge
      * `degree(node)`: Calculate node degree
      * `edges_between_node_and_community(...)`: Connection strength calculation
* **Data Structures:**
  * `Community`: Contains entity set, internal/external edge counts
  * `ClusterLevel`: Communities at specific resolution level
  * `ClusterHierarchy`: Complete hierarchical clustering result
  * `ClusteringStatistics`: Performance and quality metrics

### `optimizer.rs`

* **Purpose:** Query performance monitoring, optimization suggestions, and adaptive parameter tuning
* **Classes:**
  * `QueryOptimizer`
    * **Description:** Tracks query performance and suggests optimizations
    * **Key Fields:**
      * `query_history`: Rolling window of query statistics
      * `optimization_settings`: Caching and optimization configuration
    * **Methods:**
      * `record_query(stats)`: Log query execution statistics
      * `suggest_optimizations()`: Generate performance improvement recommendations
      * `optimize_query_parameters(max_entities, max_depth)`: Adaptive parameter tuning
      * `performance_report()`: Comprehensive performance analysis
* **Configuration Types:**
  * `OptimizationSettings`
    * **Default Values:**
      * `cache_ttl`: 300 seconds (5 minutes)
      * `max_cache_size`: 1000 entries
      * `enable_caching`: true
      * `enable_query_rewriting`: true
      * `enable_result_prefetching`: false
* **Optimization Types:**
  * `OptimizationType`: ReduceQueryScope, IncreaseCacheSize, EnablePrefetching, OptimizeEmbeddings, RebuildIndex
  * `OptimizationImpact`: Low, Medium, High severity levels
* **Performance Thresholds:**
  * **Slow Query:** >100ms triggers scope reduction suggestions
  * **Low Cache Hit Rate:** <50% triggers cache optimization
  * **Repeated Patterns:** >3 occurrences triggers prefetching recommendation

### `rag.rs`

* **Purpose:** Graph RAG (Retrieval-Augmented Generation) engine combining vector similarity with graph traversal
* **Classes:**
  * `GraphRAGEngine`
    * **Description:** Main RAG system for comprehensive context retrieval
    * **Key Fields:**
      * `graph`: Core knowledge graph
      * `cache`: LRU cache with access counting
      * `cache_size_limit`: 1000 entries maximum
    * **Methods:**
      * `retrieve_context(query_embedding, max_entities, max_depth)`: Main retrieval pipeline
      * `expand_entity_context(...)`: Multi-hop graph exploration
      * `find_bridge_entities(...)`: Identify connecting entities between clusters
      * `rank_context_entities(...)`: Multi-factor entity ranking
      * `generate_relationship_explanations(...)`: Create human-readable explanations
* **Retrieval Strategies:**
  1. **Vector Similarity Search:** Find relevant entities using embeddings
  2. **Direct Neighbor Expansion:** Include immediate neighbors of similar entities
  3. **Multi-hop Exploration:** Deep traversal for highly relevant entities (max 5)
  4. **Bridge Entity Detection:** Find entities connecting multiple similar entities
* **Scoring Algorithm:**
  * **Entity Score:** 0.6 × similarity + 0.3 × connectivity + 0.1 × recency
  * **Connectivity Score:** log₁₀(neighbor_count) capped at 1.0
  * **Distance Decay:** Weight = 1.0 / (depth + 1.0)
* **Data Structures:**
  * `GraphRAGContext`: Complete retrieval result with entities, relationships, explanations
  * `RelationshipExplanation`: Human-readable relationship descriptions
  * `QueryMetadata`: Performance and strategy tracking
  * `CacheStats`: Cache utilization metrics

### `summarization.rs`

* **Purpose:** Sub-millisecond community summarization without external LLM usage
* **Classes:**
  * `CommunitySummarizer`
    * **Description:** Fast, cache-enabled community summary generation
    * **Key Features:**
      * **Performance Target:** Sub-millisecond summarization
      * **No LLM Dependency:** Rule-based summarization for speed
      * **Async Caching:** Concurrent-safe result caching
    * **Methods:**
      * `summarize_community(community, graph)`: Main summarization pipeline
      * `extract_entity_information(...)`: Gather entity connectivity data
      * `build_fast_summary(...)`: Generate summary without LLM
      * `identify_key_entities(...)`: Score entities by centrality and hub metrics
      * `calculate_confidence_score(...)`: Community cohesion measurement
    * **Cache Management:**
      * `invalidate_cache(community_id)`: Remove specific cached summary
      * `clear_cache()`: Full cache reset
* **Summarization Algorithm:**
  1. **Entity Analysis:** Extract properties, internal/external connections
  2. **Connectivity Assessment:** Compare internal vs external link ratios
  3. **Property Extraction:** Identify common terms (top 5)
  4. **Key Entity Scoring:** Centrality + hub score combination
  5. **Confidence Calculation:** Internal connection density
* **Data Structures:**
  * `EntityInfo`: Entity metadata with connection statistics
  * `CommunitySummary`: Final summary with key entities and confidence
  * **Summary Format:** 50-100 words maximum for LLM efficiency

### `two_tier.rs`

* **Purpose:** Sophisticated two-tier query system combining global and local search strategies
* **Classes:**
  * `TwoTierQueryEngine`
    * **Description:** Orchestrates global/local query strategies with intelligent routing
    * **Key Components:**
      * `graph`: Shared knowledge graph reference
      * `hierarchy`: Async-safe cluster hierarchy
      * `summarizer`: Community summarization engine
      * `rag_engine`: Graph RAG for local searches
      * `query_cache`: 5-minute TTL query result cache
    * **Methods:**
      * `query(query)`: Main query entry point with intelligent routing
      * `global_search(...)`: Broad knowledge retrieval using communities
      * `local_search(...)`: Entity-focused neighborhood exploration
      * `hybrid_search(...)`: Combined global-local strategy
* **Query Types:**
  * `GraphRAGQuery::GlobalSearch`: Question-based search with community summaries
  * `GraphRAGQuery::LocalSearch`: Entity-centered exploration with hop limits
  * `GraphRAGQuery::HybridSearch`: Combined approach for comprehensive results
* **Search Strategies:**
  * **Community-Based Global:** Uses hierarchical clustering and summaries
  * **Traditional Global:** Direct RAG without community preprocessing
  * **Local Exploration:** Entity-focused with configurable hop distance
  * **Hybrid Coordination:** Sequential global-then-local with result merging
* **Performance Features:**
  * **Query Caching:** 5-minute TTL with automatic expiration
  * **Result Merging:** Deduplication and similarity-based ranking
  * **Concurrent Processing:** Async operations with RwLock coordination
* **Data Structures:**
  * `GraphRAGResult`: Unified result format with confidence scoring
  * `CachedQueryResult`: TTL-enabled cache entries
  * `QueryType`: Global, Local, or Hybrid search classification

## 5. Key Algorithms and Logic

### Leiden Clustering Algorithm
* **Purpose:** Community detection in knowledge graphs
* **Steps:**
  1. Initialize each node as separate community
  2. Local moving phase: Optimize modularity by moving nodes
  3. Iterative improvement until convergence
  4. Multi-resolution analysis across different granularities
* **Complexity:** O(n log n) for sparse graphs
* **Modularity Calculation:** Δmod = (edges_gain / total_edges) - resolution × degree_penalty

### Graph RAG Retrieval Pipeline
* **Purpose:** Comprehensive context generation for LLM queries
* **Steps:**
  1. **Vector Search:** Find semantically similar entities (2× max_entities)
  2. **Neighbor Expansion:** Include direct neighbors of top entities
  3. **Multi-hop Exploration:** Deep traversal for top 5 entities
  4. **Bridge Detection:** Find connecting entities between clusters
  5. **Ranking & Filtering:** Multi-factor scoring and result truncation
* **Performance:** Sub-100ms for typical queries

### Query Optimization Heuristics
* **Performance Thresholds:**
  * Query time >100ms → Reduce scope (entities × 0.8, depth - 1)
  * Cache hit rate <50% → Increase cache size/TTL
  * Repeated patterns >3× → Enable prefetching
* **Adaptive Parameters:** Dynamic adjustment based on performance history

### Two-Tier Search Strategy
* **Global Search:** Community-level analysis for broad context
* **Local Search:** Entity-focused exploration for specific details
* **Hybrid Coordination:** Sequential execution with intelligent result merging

## 6. Performance Characteristics

### Time Complexity
* **Clustering:** O(n log n) for hierarchical Leiden
* **RAG Retrieval:** O(k + d×n) where k=entities, d=depth, n=neighbors
* **Summarization:** O(m) where m=community size (sub-millisecond target)
* **Query Optimization:** O(h) where h=history size

### Memory Usage
* **Query Cache:** Configurable limit (default 1000 entries)
* **Clustering:** O(n²) adjacency matrix during clustering
* **RAG Cache:** LRU eviction with access-based prioritization
* **Summary Cache:** Concurrent HashMap with RwLock protection

### Scalability Limits
* **Max Entities per Query:** Configurable (default 50-100)
* **Max Traversal Depth:** Configurable (default 2-3 hops)
* **Cache Size:** 1000 entries (memory-bounded)
* **Community Size:** No hard limit (limited by available memory)

## 7. Dependencies

### Internal Dependencies
* **`crate::core::graph::KnowledgeGraph`:** Core graph data structure and operations
* **`crate::core::types`:** 
  * `EntityKey`: Unique entity identifiers
  * `ContextEntity`: Entity with similarity and neighbor information
  * `Relationship`: Graph edge with type and weight
  * `QueryResult`: Standard query result format
* **`crate::error::Result`:** Error handling with `GraphError` types
* **`crate::text::TextCompressor`:** Text compression for summary storage

### External Dependencies
* **`tokio`:** Async runtime for concurrent operations
* **`chrono`:** DateTime handling for cache TTL management
* **`parking_lot::RwLock`:** High-performance read-write locks
* **`std::sync::Arc`:** Atomic reference counting for shared data
* **`std::collections`:** HashMap, HashSet, VecDeque for data structures

## 8. Error Handling

### Error Types (from `crate::error::GraphError`)
* **`EntityNotFound`:** Missing entity in graph operations
* **`EntityKeyNotFound`:** Invalid entity key reference
* **`InvalidInput`:** Malformed query parameters
* **`QueryTimeout`:** Operation exceeded time limits
* **`OutOfMemory`:** Memory allocation failures

### Error Recovery Strategies
* **Graceful Degradation:** Reduce query scope on timeout/memory errors
* **Cache Fallback:** Return cached results on fresh query failures  
* **Default Values:** Use reasonable defaults for missing parameters
* **Async Error Propagation:** Proper error handling in async contexts

## 9. Configuration Options

### Query Optimization Settings
```rust
OptimizationSettings {
    enable_caching: true,
    cache_ttl: Duration::from_secs(300),  // 5 minutes
    max_cache_size: 1000,
    enable_query_rewriting: true,
    enable_result_prefetching: false,
}
```

### Clustering Configuration
```rust
HierarchicalClusterer {
    max_levels: 5,
    min_cluster_size: 2,
    resolution_levels: [0.1, 0.5, 1.0, 2.0, 5.0],
}
```

### RAG Engine Defaults
```rust
RealtimeQuery {
    max_depth: 3,
    similarity_threshold: 0.7,
    max_results: 10,
}
```

## 10. Usage Patterns

### Typical Query Flow
1. **Query Reception:** Parse and validate query parameters
2. **Strategy Selection:** Choose global, local, or hybrid approach
3. **Cache Check:** Verify if results are already cached
4. **Execution:** Run appropriate search strategy
5. **Result Processing:** Merge, rank, and format results
6. **Caching:** Store results for future queries
7. **Response:** Return formatted results to caller

### Integration with LLM Systems
* **Context Generation:** `GraphRAGContext::to_llm_context()` formats results
* **Summary Integration:** Community summaries provide high-level context
* **Explanation Generation:** Relationship explanations aid LLM understanding
* **Performance Optimization:** Sub-millisecond summarization for real-time use

### Performance Monitoring
* **Query Statistics:** Track execution time, cache hits, result counts
* **Optimization Suggestions:** Automated performance recommendations
* **Resource Usage:** Memory and cache utilization monitoring
* **Error Tracking:** Failure rate and error type analysis

## 11. Thread Safety and Concurrency

### Synchronization Primitives
* **`Arc<RwLock<T>>`:** Shared mutable state with reader-writer locks
* **`tokio::sync::RwLock`:** Async-compatible read-write locks
* **`Arc<T>`:** Immutable shared references for graph data

### Concurrent Operations
* **Cache Access:** Multiple readers, single writer for cache updates
* **Query Processing:** Parallel query execution across partitions
* **Summary Generation:** Concurrent community processing
* **Result Merging:** Thread-safe result aggregation

### Performance Considerations
* **Lock Granularity:** Fine-grained locks for specific data structures
* **Async Operations:** Non-blocking I/O and computation
* **Memory Sharing:** Minimize data copying through Arc references
* **Contention Avoidance:** Separate locks for independent operations

This query module provides a sophisticated, high-performance foundation for knowledge graph querying that scales from simple entity lookups to complex multi-strategy searches suitable for LLM integration.