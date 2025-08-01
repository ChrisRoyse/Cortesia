# Directory Overview: API

## 1. High-Level Summary

The `api` directory implements a RESTful API server for the LLMKG (Large Language Model Knowledge Graph) system. It provides HTTP endpoints for storing, querying, and managing knowledge triples, entities, and text chunks with semantic search capabilities. The API includes monitoring integration with real-time metrics dashboards.

## 2. Tech Stack

*   **Languages:** Rust
*   **Frameworks:** 
    *   Warp (async web framework)
    *   Tokio (async runtime)
*   **Libraries:** 
    *   serde/serde_json (serialization)
    *   parking_lot (efficient synchronization)
*   **Architecture:** RESTful API with shared state pattern

## 3. Directory Structure

The API module is organized as a flat structure with clear separation of concerns:
*   `mod.rs` - Module declarations
*   `models.rs` - Request/response data structures
*   `handlers.rs` - Business logic for each endpoint
*   `routes.rs` - Route definitions and middleware
*   `server.rs` - Server configuration and startup
*   `tests.rs` - Integration tests

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module declaration file that exposes the API components.
*   **Exports:**
    *   `server` - Main server implementation
    *   `routes` - Route definitions
    *   `handlers` - Request handlers
    *   `models` - Data models
    *   `tests` (conditional for testing)

### `models.rs`

*   **Purpose:** Defines all request and response data structures for the API.
*   **Request Models:**
    *   `StoreTripleRequest`
        *   Fields: `subject`, `predicate`, `object`, `confidence` (optional), `metadata` (optional)
        *   Purpose: Store knowledge triples (subject-predicate-object relationships)
    *   `StoreChunkRequest`
        *   Fields: `text`, `embedding` (optional Vec<f32>)
        *   Purpose: Store text chunks with optional embeddings
    *   `StoreEntityRequest`
        *   Fields: `name`, `entity_type`, `description`, `properties` (HashMap)
        *   Purpose: Store entities with properties
    *   `QueryTriplesRequest`
        *   Fields: `subject`, `predicate`, `object` (all optional), `limit` (optional)
        *   Purpose: Query triples with optional filters
    *   `SemanticSearchRequest`
        *   Fields: `query`, `limit`
        *   Purpose: Semantic search for similar content
    *   `EntityRelationshipsRequest`
        *   Fields: `entity_name`, `max_hops` (optional)
        *   Purpose: Get relationships for an entity
*   **Response Models:**
    *   `ApiResponse<T>`
        *   Generic wrapper with `status`, `data`, `error` fields
        *   Methods: `success(data)`, `error(message)`
    *   `StoreTripleResponse` - Returns `node_id`
    *   `MetricsResponse` - System metrics including entity count, memory stats
    *   `QueryResponse` - Returns triples, chunks, and query time
    *   `ApiDiscoveryResponse` - API documentation with endpoints

### `handlers.rs`

*   **Purpose:** Implements the business logic for each API endpoint.
*   **Type Alias:**
    *   `SharedEngine = Arc<RwLock<KnowledgeEngine>>` - Thread-safe knowledge engine
*   **Handler Functions:**
    *   `get_api_discovery()`: Returns API documentation with all endpoints
    *   `store_triple(req, engine)`: Stores a knowledge triple
    *   `store_chunk(req, engine)`: Stores a text chunk with optional embedding
    *   `store_entity(req, engine)`: Stores an entity with properties
    *   `query_triples(req, engine)`: Queries triples based on filters
    *   `semantic_search(req, engine)`: Performs semantic similarity search
    *   `get_entity_relationships(req, engine)`: Gets relationships for an entity
    *   `get_metrics(engine)`: Returns system metrics and statistics
    *   `get_entity_types(engine)`: Returns all entity types in the system
    *   `suggest_predicates(query, engine)`: Suggests predicates based on context

### `routes.rs`

*   **Purpose:** Defines API routes and applies middleware.
*   **Main Function:**
    *   `api_routes(engine: SharedEngine)`: Creates all API routes
*   **Endpoints:**
    *   `GET /api/v1/discovery` - API documentation
    *   `POST /api/v1/triple` - Store triple
    *   `POST /api/v1/chunk` - Store text chunk
    *   `POST /api/v1/entity` - Store entity
    *   `POST /api/v1/query` - Query triples
    *   `POST /api/v1/search` - Semantic search
    *   `POST /api/v1/relationships` - Get entity relationships
    *   `GET /api/v1/metrics` - System metrics
    *   `GET /api/v1/entity-types` - List entity types
    *   `GET /api/v1/suggest-predicates?context=...` - Suggest predicates
*   **Middleware:**
    *   CORS configuration allowing any origin
    *   JSON body parsing
    *   Engine injection via `with_engine` filter

### `server.rs`

*   **Purpose:** Main server implementation with monitoring integration.
*   **Classes:**
    *   `LLMKGApiServer`
        *   **Fields:**
            *   `knowledge_engine`: Arc<RwLock<KnowledgeEngine>>
            *   `metric_registry`: Arc<MetricRegistry>
            *   `config`: ApiServerConfig
        *   **Methods:**
            *   `new(config)`: Creates server instance
            *   `run()`: Starts server with API and monitoring
    *   `ApiServerConfig`
        *   **Fields:**
            *   `api_port` (default: 3001)
            *   `dashboard_http_port` (default: 8090)
            *   `dashboard_websocket_port` (default: 8081)
            *   `embedding_dim` (default: 384)
            *   `max_nodes` (default: 1000000)
*   **Features:**
    *   Integrated monitoring dashboard
    *   Multiple metrics collectors (system, application, knowledge engine)
    *   Additional monitoring endpoint: `GET /api/v1/monitoring/metrics`

### `tests.rs`

*   **Purpose:** Integration tests for API endpoints.
*   **Test Functions:**
    *   `test_store_triple_endpoint()`: Tests triple storage
    *   `test_query_triples_endpoint()`: Tests triple querying
    *   `test_get_metrics_endpoint()`: Tests metrics retrieval
    *   `test_semantic_search_endpoint()`: Tests semantic search
*   **Test Approach:**
    *   Uses real KnowledgeEngine instances
    *   Tests full request/response cycle
    *   Validates response structure and status codes

## 5. Database Interaction

The API doesn't directly interact with a database. Instead, it uses the `KnowledgeEngine` from the core module, which manages in-memory storage of knowledge graphs. The engine handles:
*   Triple storage and retrieval
*   Entity management
*   Semantic embeddings
*   Query optimization

## 6. API Endpoints

### Storage Endpoints

*   **`POST /api/v1/triple`**
    *   **Description:** Store a knowledge triple (subject-predicate-object)
    *   **Request:** `{ "subject": "string", "predicate": "string", "object": "string", "confidence": number, "metadata": {} }`
    *   **Response:** `{ "status": "success", "data": { "node_id": "string" } }`

*   **`POST /api/v1/chunk`**
    *   **Description:** Store a text chunk with optional embedding
    *   **Request:** `{ "text": "string", "embedding": [numbers] }`
    *   **Response:** `{ "status": "success", "data": { "node_id": "string" } }`

*   **`POST /api/v1/entity`**
    *   **Description:** Store an entity with properties
    *   **Request:** `{ "name": "string", "entity_type": "string", "description": "string", "properties": {} }`
    *   **Response:** `{ "status": "success", "data": { "node_id": "string" } }`

### Query Endpoints

*   **`POST /api/v1/query`**
    *   **Description:** Query for triples matching criteria
    *   **Request:** `{ "subject": "string", "predicate": "string", "object": "string", "limit": number }`
    *   **Response:** `{ "status": "success", "data": { "triples": [], "chunks": [], "query_time_ms": number } }`

*   **`POST /api/v1/search`**
    *   **Description:** Semantic search for similar content
    *   **Request:** `{ "query": "string", "limit": number }`
    *   **Response:** `{ "status": "success", "data": { "results": [], "query_time_ms": number } }`

*   **`POST /api/v1/relationships`**
    *   **Description:** Get relationships for an entity
    *   **Request:** `{ "entity_name": "string", "max_hops": number }`
    *   **Response:** `{ "status": "success", "data": { "entity": "string", "relationships": [] } }`

### Utility Endpoints

*   **`GET /api/v1/discovery`**
    *   **Description:** Get API documentation
    *   **Response:** `{ "version": "1.0.0", "endpoints": [...] }`

*   **`GET /api/v1/metrics`**
    *   **Description:** Get system metrics and statistics
    *   **Response:** `{ "status": "success", "data": { "entity_count": number, "memory_stats": {}, "entity_types": {} } }`

*   **`GET /api/v1/entity-types`**
    *   **Description:** Get all entity types
    *   **Response:** `{ "status": "success", "data": {} }`

*   **`GET /api/v1/suggest-predicates?context=...`**
    *   **Description:** Suggest predicates based on context
    *   **Response:** `{ "status": "success", "data": [...] }`

## 7. Key Variables and Logic

### Thread Safety
*   All handlers use `Arc<RwLock<KnowledgeEngine>>` for thread-safe access
*   Read operations use `.read()` locks, write operations use `.write()` locks

### Error Handling
*   All handlers return `Result<impl Reply, Rejection>`
*   Errors are wrapped in `ApiResponse::error()` with descriptive messages
*   No panics - all errors are gracefully handled

### Performance Considerations
*   Query responses include `query_time_ms` for performance monitoring
*   Default query limit of 100 items to prevent excessive memory usage
*   Configurable `max_nodes` limit (default: 1,000,000)

## 8. Dependencies

### Internal Dependencies
*   `crate::core::knowledge_engine::KnowledgeEngine` - Main knowledge graph engine
*   `crate::core::triple::Triple` - Triple data structure
*   `crate::core::knowledge_types::TripleQuery` - Query structure
*   `crate::monitoring::dashboard` - Monitoring dashboard server
*   `crate::monitoring::metrics` - Metrics registry
*   `crate::monitoring::collectors` - Various metrics collectors

### External Dependencies
*   `warp` - Web framework for building the API
*   `tokio` - Async runtime
*   `serde`/`serde_json` - JSON serialization
*   `parking_lot` - Efficient read-write locks
*   `std::sync::Arc` - Atomic reference counting for shared state

## 9. Usage Example

To start the API server:

```rust
use api::server::{LLMKGApiServer, ApiServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ApiServerConfig::default();
    let server = LLMKGApiServer::new(config)?;
    server.run().await?;
    Ok(())
}
```

The server will start on:
*   API: `http://localhost:3001/api/v1`
*   Dashboard: `http://localhost:8090`
*   WebSocket: `ws://localhost:8081`