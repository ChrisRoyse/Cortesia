# Directory Overview: WebAssembly Interface (`src/wasm`)

## 1. High-Level Summary

This directory contains WebAssembly (WASM) bindings for the LLMKG (LLM Knowledge Graph) system, providing JavaScript/web browser access to ultra-fast knowledge graph operations. The module exposes two main interfaces: a standard comprehensive API and a high-performance interface optimized specifically for LLM integration scenarios.

**Primary Purpose**: Enable web applications and JavaScript environments to leverage the high-performance Rust-based knowledge graph through WebAssembly bindings.

**Target Use Cases**:
- Web-based knowledge graph applications
- Browser-based LLM applications requiring fast context retrieval
- JavaScript/TypeScript applications needing vector similarity search
- Real-time knowledge graph RAG (Retrieval Augmented Generation) systems

## 2. Tech Stack

* **Languages:** Rust (compiled to WebAssembly)
* **Frameworks:** 
  - `wasm-bindgen` - Rust/WebAssembly bindings
  - `js-sys` - JavaScript API bindings
  - `web-sys` - Web API bindings
  - `serde_json` - JSON serialization
* **Libraries:**
  - `console_error_panic_hook` - Better error reporting in browser
  - `parking_lot` - High-performance synchronization primitives
* **JavaScript Interop:** Float32Array, Uint32Array, Uint8Array for zero-copy data transfer
* **Performance:** SIMD-optimized vector operations, memory-mapped storage

## 3. Directory Structure

```
src/wasm/
├── mod.rs              # Standard WASM API with comprehensive functionality
└── fast_interface.rs   # High-performance interface optimized for LLM workflows
```

## 4. File Breakdown

### `mod.rs` - Standard WebAssembly Interface

* **Purpose:** Provides comprehensive WebAssembly bindings for the knowledge graph with full API coverage and detailed error handling.

#### **Classes:**

* **`KnowledgeGraphWasm`**
  * **Description:** Main WebAssembly wrapper around the Rust KnowledgeGraph core
  * **Properties:**
    * `inner: KnowledgeGraph` - The underlying Rust knowledge graph instance
  * **Methods:**
    * `new(embedding_dimension: usize)` → `Result<KnowledgeGraphWasm, JsValue>`: Creates new knowledge graph instance
    * `insert_entity(id: u32, type_id: u16, properties: &str, embedding: &[f32])` → `Result<String, JsValue>`: Inserts entity with embedding vector
    * `insert_relationship(from_id: u32, to_id: u32, relationship_type: u8, weight: f32)` → `Result<String, JsValue>`: Creates weighted relationship between entities
    * `semantic_search(query_embedding: &[f32], max_results: usize)` → `Result<String, JsValue>`: Vector similarity search returning JSON results
    * `get_neighbors(entity_id: u32)` → `Result<Vec<u32>, JsValue>`: Gets directly connected entities
    * `find_path(from_id: u32, to_id: u32, max_depth: u8)` → `Result<Option<Vec<u32>>, JsValue>`: Finds shortest path between entities
    * `get_context(query_embedding: &[f32], max_entities: usize, max_depth: u8)` → `Result<String, JsValue>`: Comprehensive context retrieval for LLMs
    * `get_stats()` → `String`: Returns system statistics as JSON
    * `get_api_capabilities()` → `String`: Returns comprehensive API documentation as JSON
    * `explain_relationship(entity_a_id: u32, entity_b_id: u32)` → `Result<String, JsValue>`: Provides relationship analysis with supporting evidence

* **`PerformanceTimer`**
  * **Description:** Web-based performance measurement utility
  * **Methods:**
    * `new()` → `PerformanceTimer`: Creates timer starting from current time
    * `elapsed_ms()` → `f64`: Returns elapsed milliseconds since creation

#### **Functions:**
* `main()`: Sets up panic hook for better error reporting
* `set_panic_hook()`: Configures console error reporting
* `log(s: &str)`: Console logging helper

### `fast_interface.rs` - High-Performance LLM-Optimized Interface

* **Purpose:** Ultra-fast interface designed for high-throughput LLM workflows with minimal latency and maximum throughput.

#### **Classes:**

* **`FastKnowledgeGraph`**
  * **Description:** High-performance knowledge graph interface with advanced caching and batch processing
  * **Properties:**
    * `graph: Arc<RwLock<KnowledgeGraph>>` - Thread-safe graph core
    * `rag_engine: Arc<RwLock<GraphRAGEngine>>` - RAG retrieval engine
    * `batch_processor: Arc<RwLock<BatchProcessor>>` - SIMD batch processor
    * `embedding_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>` - Embedding cache
    * `performance_stats: Arc<RwLock<PerformanceStats>>` - Performance metrics
  * **Methods:**
    * `new(embedding_dim: usize)` → `Result<FastKnowledgeGraph, JsValue>`: Creates optimized graph instance
    * `embed(text: &str)` → `Float32Array`: Ultra-fast embedding computation with caching
    * `nearest(embedding: &Float32Array, k: usize)` → `Uint32Array`: High-speed vector similarity search
    * `neighbors(entity_id: u32, max_hops: u8)` → `Uint32Array`: Zero-copy neighbor lookup
    * `relate(entity_a: u32, entity_b: u32, max_hops: u8)` → `bool`: Fast path existence check
    * `explain(entity_ids: &Uint32Array)` → `String`: Compact context generation for LLM prompting
    * `graph_rag_search(query_embedding: &Float32Array, max_entities: usize, max_depth: u8)` → `String`: Comprehensive RAG retrieval
    * `stats()` → `JsValue`: Real-time performance statistics
    * `batch_nearest(embeddings: &Float32Array, k: usize)` → `Uint32Array`: Batch similarity search
    * `optimize_memory()`: Memory optimization and cache management
    * `llm_search(text: &str, max_entities: Option<usize>, max_depth: Option<u8>)` → `String`: One-shot LLM-optimized search
    * `get_context(query: &str)` → `String`: Simple context retrieval for LLM prompting

* **`PerformanceStats`**
  * **Description:** Performance tracking structure
  * **Properties:**
    * `total_queries: u32` - Total number of queries processed
    * `avg_query_time_ms: f64` - Average query latency
    * `cache_hit_rate: f64` - Embedding cache hit rate
    * `memory_usage_mb: f64` - Current memory usage

## 5. Core Dependencies and Relationships

### **Internal Dependencies:**
* `crate::core::graph::KnowledgeGraph` - Core knowledge graph implementation
* `crate::core::types::{EntityData, Relationship}` - Core data structures
* `crate::error::GraphError` - Error handling system
* `crate::query::rag::GraphRAGEngine` - RAG retrieval system
* `crate::embedding::simd_search::BatchProcessor` - SIMD-optimized batch processing

### **External Dependencies:**
* `wasm-bindgen` - Rust/WebAssembly/JavaScript bindings
* `js-sys` - JavaScript standard library bindings
* `web-sys` - Web APIs (console, performance, window)
* `serde_json` - JSON serialization/deserialization
* `parking_lot` - High-performance synchronization primitives

## 6. API Capabilities and Performance Targets

### **Performance Specifications:**
* **Query Latency:** < 1ms for single entity operations
* **Similarity Search:** < 5ms for vector searches
* **Context Retrieval:** < 10ms for comprehensive context
* **Memory Efficiency:** < 70 bytes per entity
* **Embedding Dimension:** Up to 4,096 dimensions supported
* **Scale:** Up to 100 million entities supported

### **Key Features:**
* **Zero-Copy Operations:** Uses typed arrays for efficient data transfer
* **Batch Processing:** SIMD-optimized batch operations
* **Memory Optimization:** Automatic cache management and memory compaction
* **Real-Time Statistics:** Live performance monitoring
* **Error Recovery:** Comprehensive error handling with detailed messages

## 7. Usage Patterns and Integration

### **Graph RAG Pattern:**
```javascript
// Initialize knowledge graph
const kg = new FastKnowledgeGraph(384); // 384-dimensional embeddings

// Retrieve context for LLM
const context = kg.get_context("What is machine learning?");
// Use context in LLM prompt
```

### **Entity Discovery Pattern:**
```javascript
// Find similar entities
const embedding = kg.embed("artificial intelligence");
const similar = kg.nearest(embedding, 10);
```

### **Relationship Exploration:**
```javascript
// Check if entities are connected
const connected = kg.relate(entity1_id, entity2_id, 3);
const explanation = kg.explain(new Uint32Array([entity1_id, entity2_id]));
```

### **Batch Processing Pattern:**
```javascript
// Process multiple queries efficiently
const embeddings = new Float32Array([...multiple_embeddings]);
const results = kg.batch_nearest(embeddings, 5);
```

## 8. Memory Management and Optimization

* **Embedding Cache:** Automatic caching of computed embeddings with LRU eviction
* **Memory Compaction:** `optimize_memory()` triggers cache cleanup and storage optimization
* **Zero-Copy Transfer:** Uses JavaScript typed arrays to avoid memory copying
* **Arc/RwLock Pattern:** Thread-safe shared ownership with reader-writer locks
* **SIMD Optimization:** Vectorized operations for batch processing

## 9. Error Handling and Debugging

* **Panic Hook:** `console_error_panic_hook` provides detailed error information in browser console
* **Result Types:** All fallible operations return `Result<T, JsValue>` for proper error handling
* **Performance Monitoring:** Built-in timing and statistics for performance debugging
* **Detailed Error Messages:** Context-rich error messages for debugging

## 10. JavaScript/TypeScript Integration

### **Type Definitions:**
* `SearchParams` - TypeScript interface for search parameters
* `SearchResult` - TypeScript interface for search results with entities and relationships
* Automatic TypeScript bindings generation through `wasm-bindgen`

### **Data Formats:**
* **Embeddings:** Float32Array for vector data
* **Entity IDs:** Uint32Array for entity collections
* **JSON Responses:** Structured JSON for complex data (search results, statistics, context)
* **Properties:** String-based JSON for entity properties

## 11. Development and Extension Notes

* **Adding New Methods:** Use `#[wasm_bindgen]` attribute for new public methods
* **Performance Critical Code:** Use zero-copy operations with typed arrays
* **Error Handling:** Always return `Result<T, JsValue>` for fallible operations
* **Memory Safety:** Leverage Rust's ownership system even in WASM context
* **Testing:** Use browser-based testing for WASM functionality verification