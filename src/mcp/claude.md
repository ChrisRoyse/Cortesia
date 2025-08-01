# Directory Overview: Model Context Protocol (MCP) Server Implementation

## 1. High-Level Summary

The `src/mcp` directory contains a comprehensive Model Context Protocol (MCP) server implementation for the LLMKG (Large Language Model Knowledge Graph) system. This module provides multiple specialized MCP servers that enable LLMs to interact with knowledge graphs through standardized tools and APIs. The implementation includes production-ready servers, brain-inspired cognitive capabilities, federated multi-database operations, and a complete suite of LLM-friendly tools for knowledge storage, retrieval, and analysis.

## 2. Tech Stack

* **Languages:** Rust
* **Frameworks:** Tokio (async runtime), Serde (serialization)
* **Libraries:** 
  - `tokio::sync::RwLock` for concurrent access
  - `serde_json` for JSON handling
  - `ahash::AHashMap` for high-performance hashing
  - `uuid` for unique identifier generation
  - `chrono` for time handling
  - `lazy_static` for global singletons
* **Architecture:** Async/await pattern, modular handler design
* **Communication:** JSON-RPC style request/response patterns

## 3. Directory Structure

```
src/mcp/
├── mod.rs                          # Main module with core server and global singletons
├── shared_types.rs                 # Common data structures across all MCP servers
├── production_server.rs            # Production-ready wrapper with monitoring and health checks
├── brain_inspired_server.rs        # Cognitive capabilities with temporal graph integration
├── federated_server.rs            # Multi-database federation and cross-system operations
└── llm_friendly_server/           # Primary LLM-optimized MCP server
    ├── mod.rs                     # Main LLM-friendly server implementation
    ├── types.rs                   # Type definitions and usage statistics
    ├── tools.rs                   # Complete tool definitions with examples
    ├── handlers/                  # Request handlers by category
    │   ├── mod.rs                # Handler module exports
    │   ├── storage.rs            # Fact and knowledge storage handlers
    │   ├── query.rs              # Search and retrieval handlers
    │   ├── advanced.rs           # Advanced analysis and validation
    │   ├── cognitive.rs          # Cognitive reasoning and divergent thinking
    │   ├── graph_analysis.rs     # Graph structure analysis
    │   ├── temporal.rs           # Time-travel queries and branching
    │   └── stats.rs              # Statistics and health reporting
    ├── validation.rs             # Input validation and data integrity
    ├── utils.rs                  # Utility functions and statistics updates
    ├── migration.rs              # Tool migration and deprecation handling
    ├── temporal_tracking.rs      # Temporal indexing and change tracking
    ├── database_branching.rs     # Database versioning and branch management
    ├── query_generation.rs       # Query optimization and generation
    ├── query_generation_enhanced.rs  # AI-enhanced query processing
    ├── query_generation_native.rs    # Native query implementations
    ├── divergent_graph_traversal.rs  # Creative exploration algorithms
    ├── search_fusion.rs          # Multi-mode search combination
    └── reasoning_engine.rs       # Logical reasoning chain generation
```

## 4. File Breakdown

### `mod.rs` (Main Module)
* **Purpose:** Central module that exports all MCP servers and provides global resource management
* **Key Components:**
  * `LLMKGMCPServer` struct: Legacy MCP server implementation
  * Global singletons via `lazy_static!`:
    - `MODEL_MANAGER`: AI model resource management (4GB limit, 3 concurrent models)
    - `PROCESSING_CONFIG`: Knowledge processing configuration for entity/relationship extraction
  * Core server initialization with GraphRAG engine integration
* **Functions:**
  * `new(embedding_dim: usize)`: Creates server with specified embedding dimensions
  * `initialize_mmap_storage()`: Sets up memory-mapped storage for performance
  * `get_tools()`: Returns available MCP tools (knowledge_search, entity_lookup, etc.)
  * `handle_request()`: Routes incoming MCP requests to appropriate handlers

### `shared_types.rs` (Common Data Structures)
* **Purpose:** Defines shared data structures used across all MCP server implementations
* **Key Types:**
  * `MCPTool`: Basic tool definition with name, description, and JSON schema
  * `LLMMCPTool`: Enhanced tool with examples and usage tips for LLM consumption
  * `MCPRequest/MCPResponse`: Standard request/response structures
  * `LLMMCPRequest/LLMMCPResponse`: Enhanced structures with performance metrics
  * `PerformanceInfo`: Execution metrics (time, memory, cache hits, complexity scores)

### `production_server.rs` (Production-Ready Server)
* **Purpose:** Production wrapper providing comprehensive monitoring, health checks, and graceful shutdown
* **Key Components:**
  * `ProductionMCPServer`: Wraps LLMFriendlyMCPServer with production features
  * Integration with production system for error recovery, rate limiting, metrics
* **Functions:**
  * `new()`: Creates production server with optional configuration
  * `handle_request()`: Protected request handling with timeout and error recovery
  * `get_system_status()`: Comprehensive system health and resource metrics
  * `get_health_report()`: Structured health assessment with component status
  * `get_prometheus_metrics()`: Metrics export in Prometheus format
  * `shutdown()`: Graceful shutdown with cleanup report

### `brain_inspired_server.rs` (Cognitive Server)
* **Purpose:** Provides brain-inspired cognitive capabilities with temporal graph integration
* **Key Components:**
  * `BrainInspiredMCPServer`: Server with optional cognitive orchestrator
  * Integration with `TemporalKnowledgeGraph` and `CognitiveOrchestrator`
* **Functions:**
  * `new_with_cognitive_capabilities()`: Initialize with full cognitive features
  * `handle_store_knowledge()`: Graph-based knowledge storage with activation states
  * `handle_query()`: Supports exact matching and cognitive pattern queries
  * `handle_cognitive_reasoning_tool_call()`: Advanced reasoning with multiple cognitive patterns

### `federated_server.rs` (Federation Server)
* **Purpose:** Multi-database operations with mathematical computations and version management
* **Key Components:**
  * `FederatedMCPServer`: Manages multiple knowledge graph databases
  * `FederatedUsageStats`: Tracks cross-database operation statistics
* **Functions:**
  * `get_federated_tools()`: Provides cross-database similarity search and version control tools

## 5. LLM-Friendly Server Deep Dive

### Core Server (`llm_friendly_server/mod.rs`)
* **Purpose:** Primary MCP server optimized for LLM interaction with comprehensive tool suite
* **Key Features:**
  * Enhanced storage configuration with AI model integration
  * Timeout protection (5-second default) for all operations
  * Automatic request migration for deprecated tools
  * Performance monitoring and usage statistics
* **Functions:**
  * `new_with_enhanced_config()`: Customizable server creation
  * `handle_request()`: Main request dispatcher with timeout protection
  * `get_health()`: Health check including model manager statistics

### Tool Definitions (`tools.rs`)
* **Purpose:** Comprehensive tool catalog with 20+ specialized tools for knowledge graph operations
* **Tool Categories:**
  1. **Core Storage & Retrieval:**
     - `store_fact`: Simple subject-predicate-object triples
     - `store_knowledge`: Complex text with AI-powered processing
     - `find_facts`: Enhanced retrieval with semantic similarity
     - `ask_question`: Natural language Q&A with evidence synthesis
  
  2. **Advanced Analysis:**
     - `hybrid_search`: Multi-mode search (semantic, structural, keyword)
     - `analyze_graph`: Centrality, clustering, prediction analysis
     - `validate_knowledge`: Consistency and quality checking
  
  3. **Specialized Cognitive Tools:**
     - `divergent_thinking_engine`: Creative exploration and ideation
     - `cognitive_reasoning_chains`: Logical reasoning (deductive, inductive, abductive)
     - `time_travel_query`: Temporal database queries and evolution tracking
  
  4. **Branching & Versioning:**
     - `create_branch`: Git-like database branching
     - `list_branches`: Branch management and discovery
     - `compare_branches`: Difference analysis between branches
     - `merge_branches`: Controlled merging with conflict resolution

### Request Handlers (`handlers/`)

#### Storage Handlers (`storage.rs`)
* **Functions:**
  * `handle_store_fact()`: Validates and stores simple triples with confidence scores
  * `handle_store_knowledge()`: Processes complex text through AI extraction pipeline
* **Validation:** Length limits (subject/object: 128 chars, predicate: 64 chars), confidence bounds

#### Query Handlers (`query.rs`)
* **Functions:**
  * `handle_find_facts()`: Triple pattern matching with enhanced retrieval fallback
  * `handle_ask_question()`: Natural language processing with multi-hop reasoning
* **Features:** Semantic similarity, evidence synthesis, confidence scoring

#### Advanced Handlers (`advanced.rs`)
* **Functions:**
  * `handle_hybrid_search()`: SIMD/LSH acceleration, multiple search modes
  * `handle_validate_knowledge()`: Consistency checking, conflict detection
  * `handle_cognitive_reasoning_chains()`: Multi-step logical reasoning

## 6. Database Integration

### Knowledge Engine Integration
* **Primary Database:** `KnowledgeEngine` with configurable embedding dimensions
* **Storage Types:**
  - Memory-mapped storage (`MMapStorage`) for performance
  - Triple-based storage for facts
  - Chunk-based storage for complex knowledge
* **Caching:** Multi-level caching with embedding cache and result cache

### Temporal Capabilities
* **Time Travel Queries:** Point-in-time snapshots, evolution tracking
* **Versioning:** Database branching similar to Git workflows
* **Change Detection:** Temporal indexing for modification tracking

## 7. API Endpoints (MCP Tools)

### Core Knowledge Operations
* **`store_fact`**: POST-style operation for storing S-P-O triples
  - Input: `{subject, predicate, object, confidence?}`
  - Response: Confirmation with triple ID and metadata
* **`find_facts`**: GET-style operation for triple retrieval
  - Input: `{query: {subject?, predicate?, object?}, limit?}`
  - Response: Array of matching triples with confidence scores

### Advanced Operations
* **`hybrid_search`**: Multi-modal search with performance optimization
  - Input: `{query, search_type, performance_mode, filters?, config?}`
  - Response: Ranked results with scoring details and performance metrics
* **`analyze_graph`**: Graph analysis suite
  - Input: `{analysis_type, config}`
  - Response: Analysis results (centrality scores, clusters, predictions)

## 8. Key Variables and Logic

### Global Configuration
* **`MODEL_MANAGER`**: Singleton managing AI models with memory limits and concurrency control
* **`PROCESSING_CONFIG`**: Entity extraction settings (confidence thresholds, chunk sizes)

### Performance Parameters
* **Request Timeout**: 5-second default with configurable limits
* **Memory Limits**: 4GB for model manager, 2GB for enhanced storage
* **Concurrency**: 3 concurrent models maximum, 8 SIMD workers for search
* **Cache Configuration**: LRU caching with configurable hit rate targets

### Enhanced Storage Features
* **AI Processing Pipeline**: Entity extraction → relationship mapping → semantic chunking
* **Fallback Strategy**: Automatic fallback to basic processing on AI failure
* **Quality Metrics**: Confidence scoring, importance ranking, coherence assessment

## 9. Dependencies

### Internal Dependencies
* **Core Modules:**
  - `crate::core::knowledge_engine::KnowledgeEngine`
  - `crate::core::triple::Triple`
  - `crate::query::rag::GraphRAGEngine`
  - `crate::embedding::simd_search::BatchProcessor`

* **Cognitive Modules:**
  - `crate::cognitive::CognitiveOrchestrator`
  - `crate::versioning::temporal_graph::TemporalKnowledgeGraph`

* **Production Features:**
  - `crate::production::ProductionSystem`
  - `crate::production::graceful_shutdown`

### External Dependencies
* **Core Rust:**
  - `std::sync::Arc`, `std::collections::HashMap`
  - `tokio::sync::RwLock` for async concurrency
* **Serialization:**
  - `serde`, `serde_json` for data serialization
* **Utilities:**
  - `uuid` for unique identifiers
  - `chrono` for temporal operations
  - `lazy_static` for global state management

## 10. Performance Optimizations

### Search Acceleration
* **SIMD Operations**: Vectorized similarity search with 10x performance gains
* **LSH (Locality-Sensitive Hashing)**: Approximate search with 8.5x speedup
* **Memory Mapping**: Direct file system integration for large datasets

### Caching Strategy
* **Multi-Level Cache**: Embedding cache, result cache, model cache
* **Cache Metrics**: Hit rate monitoring and optimization
* **Memory Management**: Automatic cleanup with configurable thresholds

### Concurrent Processing
* **Async Design**: Non-blocking operations throughout
* **Resource Pooling**: Shared model instances and batch processors
* **Rate Limiting**: Protection against resource exhaustion

## 11. Error Handling and Recovery

### Production Features
* **Timeout Protection**: Automatic request timeout with cleanup
* **Graceful Degradation**: Fallback to basic processing on enhanced feature failures
* **Health Monitoring**: Continuous system health assessment
* **Error Recovery**: Automatic retry logic with exponential backoff

### Validation and Safety
* **Input Validation**: Comprehensive parameter checking and sanitization
* **Resource Limits**: Memory and processing time constraints
* **Conflict Resolution**: Automated conflict detection and resolution strategies

This MCP implementation represents a sophisticated knowledge graph interface designed specifically for LLM consumption, providing both high-level convenience tools and advanced analytical capabilities while maintaining production-grade reliability and performance.