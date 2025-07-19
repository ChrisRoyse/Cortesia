# LLMKG Query Module Analysis Report

## Project Context
- **Project Name:** LLMKG (Lightning-fast Knowledge Graph)
- **Project Goal:** A high-performance knowledge graph optimized for LLM integration
- **Programming Languages & Frameworks:** Rust, Tokio (async runtime), Serde (serialization), and various optimization libraries
- **Directory Under Analysis:** ./src/query/

---

## File Analysis: mod.rs

### 1. Purpose and Functionality

**Primary Role:** Module declaration and core query types definition

**Summary:** This file serves as the module root for the query system, defining fundamental types for real-time queries and partitioned graph RAG operations. It establishes the public API surface for the query module and provides core data structures for distributed querying.

**Key Components:**

- **RealtimeQuery**: Core query structure that encapsulates query parameters including text, initial nodes, traversal depth, similarity threshold, and result limits. Provides a builder pattern with methods like `with_initial_nodes()` and `with_max_depth()` for flexible query construction.

- **PartitionedGraphRAG**: Manages distributed graph querying across multiple partitions. Contains partition collections, strategies for partitioning (by node ID, entity type, or embedding cluster), and merge strategies for combining results from different partitions.

- **PartitionStrategy**: Enum defining how the graph should be partitioned - supports ByNodeId, ByEntityType, ByEmbeddingCluster, and Custom strategies.

- **MergeStrategy**: Enum for result combination strategies - UnionAll, IntersectAll, RankByScore, and Custom options.

### 2. Project Relevance and Dependencies

**Architectural Role:** Acts as the central entry point and type definition hub for the query system. Other query modules depend on these types for consistent query representation across the system.

**Dependencies:**
- **Imports:** 
  - `crate::core::graph::KnowledgeGraph` - Core graph structure for storing knowledge
  - `std::sync::Arc` - For thread-safe reference counting of graph partitions
- **Exports:** All submodules (rag, optimizer, clustering, summarization, two_tier) and the core types (RealtimeQuery, PartitionedGraphRAG, PartitionStrategy, MergeStrategy)

### 3. Testing Strategy

**Overall Approach:** Focus on unit testing the builder patterns and type conversions, with integration tests verifying proper module loading.

**Unit Testing Suggestions:**
- **RealtimeQuery Construction:**
  - Happy Path: Create query with valid text and verify default values
  - Edge Cases: Empty query text, extreme depth values (0, u32::MAX)
  - Error Handling: Test builder methods with invalid combinations

- **PartitionedGraphRAG Operations:**
  - Happy Path: Add partitions and verify they're stored correctly
  - Edge Cases: Empty partition list, duplicate partitions
  - Error Handling: Test conversion from single graph to partitioned

**Integration Testing Suggestions:**
- Verify all submodules are properly exposed through mod.rs
- Test that types can be passed between different query subsystems

---

## File Analysis: clustering.rs

### 1. Purpose and Functionality

**Primary Role:** Graph clustering implementation using the Leiden algorithm

**Summary:** Implements hierarchical clustering for knowledge graphs using the Leiden community detection algorithm. Creates multi-level community structures that can be used for efficient graph summarization and query optimization. The implementation supports different resolution levels for varying granularities of clustering.

**Key Components:**

- **HierarchicalClusterer**: Main clustering orchestrator that manages the Leiden algorithm execution across multiple resolution levels. Contains configuration for max levels, minimum cluster size, and resolution parameters.

- **LeidenClustering**: Core algorithm implementation that performs community detection through iterative node movement and modularity optimization. Uses local moving phases to optimize community assignments.

- **AdjacencyMatrix**: Efficient matrix representation for graph edges, supporting operations like degree calculation, neighbor retrieval, and edge counting between nodes and communities.

- **Community**: Represents a detected community with entity membership, internal/external edge counts, and total degree information.

- **ClusterHierarchy**: Complete hierarchical result containing multiple clustering levels and parent-child relationships between communities across levels.

### 2. Project Relevance and Dependencies

**Architectural Role:** Provides graph partitioning capabilities essential for the two-tier query system and community summarization. Enables efficient global search by organizing entities into meaningful communities.

**Dependencies:**
- **Imports:**
  - `crate::core::graph::KnowledgeGraph` - Source graph for clustering
  - `crate::core::types::EntityKey` - Entity identification
  - `crate::error::{GraphError, Result}` - Error handling
  - Standard collections (HashMap, HashSet) for community management
- **Exports:** HierarchicalClusterer, Community, ClusterLevel, ClusterHierarchy, ClusteringStatistics

### 3. Testing Strategy

**Overall Approach:** Heavy unit testing required due to complex algorithmic logic. Focus on correctness of community detection and performance at scale.

**Unit Testing Suggestions:**
- **Leiden Algorithm:**
  - Happy Path: Small graph with known community structure
  - Edge Cases: Single node graph, disconnected components, complete graph
  - Error Handling: Empty graph, invalid resolution parameters

- **Hierarchy Building:**
  - Happy Path: Multi-level clustering with proper parent-child relationships
  - Edge Cases: Single level only, communities smaller than min_size
  - Error Handling: Conflicting hierarchy relationships

**Integration Testing Suggestions:**
- Test with real knowledge graph data to verify community quality
- Performance testing with graphs of various sizes (100 to 1M nodes)
- Verify integration with two_tier.rs for query optimization

---

## File Analysis: optimizer.rs

### 1. Purpose and Functionality

**Primary Role:** Query performance optimization and monitoring

**Summary:** Provides query performance tracking, optimization suggestions, and parameter tuning based on historical query patterns. Maintains query statistics and suggests improvements to enhance system performance.

**Key Components:**

- **QueryOptimizer**: Main optimization engine that tracks query history and provides optimization suggestions. Manages settings for caching, query rewriting, and result prefetching.

- **QueryStats**: Captures execution metrics including query hash, execution time, result count, cache hit status, and timestamp.

- **OptimizationSettings**: Configuration for optimization features including cache settings (TTL, size), query rewriting, and prefetching options.

- **OptimizationSuggestion**: Structured recommendations with type, description, impact level, and estimated improvement percentage.

- **PerformanceReport**: Comprehensive performance metrics including total queries, average execution time, cache hit rate, QPS, and optimization suggestions.

### 2. Project Relevance and Dependencies

**Architectural Role:** Acts as a performance monitoring layer that can be integrated with any query execution path to provide optimization insights and automatic parameter tuning.

**Dependencies:**
- **Imports:**
  - `std::time::Duration` - Time measurement
  - `std::collections::HashMap` - For pattern detection
- **Exports:** QueryOptimizer, QueryStats, OptimizationSettings, OptimizationSuggestion, PerformanceReport

### 3. Testing Strategy

**Overall Approach:** Focus on statistical calculations and suggestion logic. Mock time-based operations for deterministic testing.

**Unit Testing Suggestions:**
- **Statistics Calculation:**
  - Happy Path: Calculate average time, cache hit rate with normal data
  - Edge Cases: Empty history, single query, all cache hits/misses
  - Error Handling: Division by zero in rate calculations

- **Optimization Suggestions:**
  - Happy Path: Generate appropriate suggestions for slow queries
  - Edge Cases: Perfect performance (no suggestions needed)
  - Error Handling: Conflicting optimization requirements

**Integration Testing Suggestions:**
- Test with actual query execution to verify timing accuracy
- Validate that parameter adjustments improve performance
- Long-running test to verify query history retention logic

---

## File Analysis: rag.rs

### 1. Purpose and Functionality

**Primary Role:** Graph RAG (Retrieval-Augmented Generation) engine implementation

**Summary:** Implements a sophisticated RAG system that combines vector similarity search with graph traversal to retrieve comprehensive context for LLM queries. Uses multiple strategies including direct neighbor expansion, multi-hop exploration, and bridge entity detection.

**Key Components:**

- **GraphRAGEngine**: Core engine that orchestrates context retrieval using hybrid search strategies. Manages caching and combines vector search with graph topology analysis.

- **GraphRAGContext**: Complete query result containing entities, relationships, explanations, and metadata. Provides `to_llm_context()` method for LLM-friendly formatting.

- **CachedResult**: LRU cache implementation for query results with timestamp and access counting.

- **RelationshipExplanation**: Structured explanations for entity relationships including confidence scores and supporting evidence.

- **Retrieval Strategies**: Three main approaches - similarity-based entity selection, multi-hop neighbor expansion, and bridge entity identification for connecting disparate graph regions.

### 2. Project Relevance and Dependencies

**Architectural Role:** Central retrieval mechanism that powers the local search functionality in the two-tier system. Provides the core RAG capabilities for knowledge graph querying.

**Dependencies:**
- **Imports:**
  - `crate::core::graph::KnowledgeGraph` - Graph data source
  - `crate::core::types::{ContextEntity, EntityKey, QueryResult, Relationship}` - Core type definitions
  - Standard collections for caching and visited tracking
- **Exports:** GraphRAGEngine, GraphRAGContext, RelationshipExplanation, CacheStats

### 3. Testing Strategy

**Overall Approach:** Combination of unit tests for individual strategies and integration tests for complete retrieval flows. Mock the knowledge graph for controlled testing.

**Unit Testing Suggestions:**
- **Vector Similarity Search:**
  - Happy Path: Find similar entities with known embeddings
  - Edge Cases: All entities equally similar, no similar entities
  - Error Handling: Invalid embedding dimensions

- **Graph Traversal:**
  - Happy Path: Expand from seed entities to specified depth
  - Edge Cases: Isolated nodes, cyclic references, depth = 0
  - Error Handling: Non-existent entity IDs

**Integration Testing Suggestions:**
- End-to-end retrieval with different query types
- Cache effectiveness testing with repeated queries
- Performance benchmarking for various graph sizes and query complexities

---

## File Analysis: summarization.rs

### 1. Purpose and Functionality

**Primary Role:** Fast community summarization for MCP (Model Context Protocol) responses

**Summary:** Provides sub-millisecond community summarization without using LLMs internally. Designed to quickly generate concise summaries of graph communities for external LLM consumption, using statistical analysis and text compression.

**Key Components:**

- **CommunitySummarizer**: Main summarization engine with caching support. Analyzes community structure and generates compressed summaries suitable for LLM context windows.

- **CommunitySummary**: Structured summary containing community ID, entity count, compressed summary text, key entities, and confidence score.

- **EntityInfo**: Internal structure tracking entity properties and connectivity metrics (internal vs external connections).

- **Fast Summary Generation**: Statistical approach using connectivity analysis, property extraction, and key entity identification based on centrality metrics.

### 2. Project Relevance and Dependencies

**Architectural Role:** Critical component for the global search functionality in the two-tier system. Enables efficient summarization of large graph regions without expensive LLM calls.

**Dependencies:**
- **Imports:**
  - `crate::core::graph::KnowledgeGraph` - Entity property access
  - `crate::query::clustering::Community` - Community structures to summarize
  - `crate::text::TextCompressor` - Text compression utilities
  - `tokio::sync::RwLock` - Async cache management
- **Exports:** CommunitySummarizer, CommunitySummary

### 3. Testing Strategy

**Overall Approach:** Focus on performance testing to ensure sub-millisecond operation. Test summary quality and cache behavior.

**Unit Testing Suggestions:**
- **Summary Generation:**
  - Happy Path: Summarize typical community with mixed connectivity
  - Edge Cases: Empty community, single entity, fully connected clique
  - Error Handling: Missing entity data in graph

- **Performance:**
  - Happy Path: Verify sub-millisecond execution (as shown in existing test)
  - Edge Cases: Large communities (1000+ entities)
  - Error Handling: Cache overflow scenarios

**Integration Testing Suggestions:**
- Test with real clustering output from clustering.rs
- Verify summary quality is sufficient for LLM understanding
- Concurrent access testing for cache thread safety

---

## File Analysis: two_tier.rs

### 1. Purpose and Functionality

**Primary Role:** Two-tier query orchestration system combining global and local search strategies

**Summary:** Implements a sophisticated query routing system that determines optimal search strategies based on query type. Supports global searches using community summaries, local entity-focused searches, and hybrid approaches that combine both strategies.

**Key Components:**

- **TwoTierQueryEngine**: Main orchestrator that routes queries to appropriate search strategies. Manages cluster hierarchy, coordinates with RAG engine and summarizer, and maintains query result cache.

- **GraphRAGQuery**: Enum defining three query types - GlobalSearch (broad knowledge retrieval), LocalSearch (entity-focused exploration), and HybridSearch (combination of both).

- **GraphRAGResult**: Comprehensive result structure containing entities, relationships, community summaries, global context, and performance metadata.

- **Search Strategies**: 
  - Global: Uses community summaries for broad topic understanding
  - Local: Focuses on specific entities and their neighborhoods  
  - Hybrid: Combines global context with targeted local exploration

- **CachedQueryResult**: Time-based cache implementation with TTL support for query results.

### 2. Project Relevance and Dependencies

**Architectural Role:** Top-level query interface that integrates all other query components. Serves as the main entry point for external systems querying the knowledge graph.

**Dependencies:**
- **Imports:**
  - `crate::query::clustering::ClusterHierarchy` - Community structure for global search
  - `crate::query::summarization::CommunitySummarizer` - Community summarization
  - `crate::query::rag::GraphRAGEngine` - Core RAG functionality
  - Various async primitives for concurrent operations
- **Exports:** TwoTierQueryEngine, GraphRAGQuery, GraphRAGResult, QueryType

### 3. Testing Strategy

**Overall Approach:** Integration testing is critical as this module orchestrates multiple subsystems. Mock individual components for unit tests.

**Unit Testing Suggestions:**
- **Query Routing:**
  - Happy Path: Each query type routes to correct handler
  - Edge Cases: Empty queries, missing cluster hierarchy
  - Error Handling: Component failures (summarizer, RAG engine)

- **Result Merging:**
  - Happy Path: Merge global and local results in hybrid search
  - Edge Cases: Overlapping entities, conflicting scores
  - Error Handling: Incomplete results from subsystems

**Integration Testing Suggestions:**
- Full pipeline testing with real graph data
- Performance comparison between query strategies
- Cache effectiveness across different query patterns
- Concurrent query handling and result consistency

---

## Directory Summary: ./src/query/

### Overall Purpose and Role

The query directory implements a sophisticated multi-strategy knowledge graph querying system optimized for LLM integration. It provides high-performance retrieval mechanisms that combine traditional graph algorithms (clustering, traversal) with modern RAG techniques (vector similarity, context expansion). The system is designed to handle both broad exploratory queries and focused entity-specific searches efficiently.

### Core Files

1. **two_tier.rs** - Most critical file as it orchestrates all query operations and provides the external API
2. **rag.rs** - Foundational retrieval engine that powers the core graph exploration capabilities  
3. **clustering.rs** - Essential for enabling scalable global search through hierarchical community detection

### Interaction Patterns

- External systems primarily interact through `TwoTierQueryEngine::query()` method
- The two-tier engine routes to either RAG engine (local search) or clustering+summarization (global search)
- Query optimizer can wrap any query execution to provide performance insights
- All components designed for async operation with Tokio runtime
- Results formatted for direct LLM consumption via `to_llm_context()` methods

### Directory-Wide Testing Strategy

**Shared Infrastructure Needs:**
- Mock `KnowledgeGraph` implementation for consistent test data
- Shared test utilities for creating sample communities and entities
- Performance benchmarking harness for sub-millisecond operation verification

**Integration Test Scenarios:**
1. **End-to-End Query Flow**: Create graph → cluster → query (all types) → verify results
2. **Performance Regression**: Ensure query latency remains under target thresholds
3. **Concurrent Operations**: Multiple queries accessing shared caches and resources
4. **Failure Cascades**: Verify graceful degradation when subsystems fail

**Quality Assurance Approach:**
- Maintain benchmark suite testing performance at different scale points (1K, 10K, 100K entities)
- Property-based testing for clustering algorithm correctness
- Fuzzing for query parser robustness
- Load testing for cache effectiveness under pressure