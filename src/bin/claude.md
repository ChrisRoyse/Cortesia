# Directory Overview: src/bin/

## 1. High-Level Summary

This directory contains six executable binaries for the LLMKG (Lightning-fast Knowledge Graph optimized for LLM integration) system. These binaries provide various server implementations including API servers, MCP (Model Context Protocol) servers, monitoring dashboards, and test utilities. The binaries collectively form the server infrastructure for deploying and monitoring the LLMKG knowledge graph system.

## 2. Tech Stack

* **Languages:** Rust
* **Frameworks:** 
  - Tokio (async runtime)
  - Warp (web framework)
  - Tokio-tungstenite (WebSocket)
* **Libraries:** 
  - Serde/serde_json (serialization)
  - Clap (CLI parsing)
  - Tracing/log (logging)
  - Chrono (time handling)
  - Rand (random number generation)
* **Database:** None directly (uses in-memory knowledge graph structures)

## 3. Directory Structure

All files are at the root level of `/src/bin/` with no subdirectories.

## 4. File Breakdown

### `llmkg_api_server.rs`

* **Purpose:** Main API server for LLMKG that exposes REST endpoints and provides a web dashboard interface.
* **Main Function:**
  * `main()`: Initializes logging, configures server with ports (API: 3001, Dashboard: 8090, WebSocket: 8081), creates and runs the API server.
* **Key Configuration:**
  * `ApiServerConfig`: Defines ports and knowledge graph parameters
    - `api_port`: 3001
    - `dashboard_http_port`: 8090
    - `dashboard_websocket_port`: 8081
    - `embedding_dim`: 384
    - `max_nodes`: 1,000,000
* **Dependencies:**
  * Internal: `llmkg::api::server::{LLMKGApiServer, ApiServerConfig}`, `llmkg::enhanced_knowledge_storage::logging`
  * External: tokio, tracing

### `llmkg_brain_server.rs`

* **Purpose:** Brain-enhanced dashboard server that provides real-time monitoring of the brain-inspired knowledge graph with neural-like activation patterns and synaptic weights.
* **Main Function:**
  * `main()`: Initializes brain-enhanced knowledge graph, populates with demo data, sets up monitoring collectors, and runs dashboard server.
* **Key Features:**
  * Creates 20 demo entities with random embeddings and activation levels
  * Establishes 15 relationships with synaptic weights
  * Runs continuous simulation that adds new entities and updates activations every 10 seconds
  * Provides comprehensive monitoring including system metrics, application metrics, and brain-specific metrics
* **Configuration:**
  * `DashboardConfig`:
    - `http_port`: 8082
    - `websocket_port`: 8083
    - `refresh_rate_ms`: 1000
* **Dependencies:**
  * Internal: `llmkg::monitoring::*`, `llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph`
  * External: tokio, rand, chrono, sysinfo

### `llmkg_mcp_server.rs`

* **Purpose:** Model Context Protocol (MCP) server that enables LLMs like Claude to interact with the LLMKG knowledge graph via JSON-RPC over stdio.
* **Main Function:**
  * `main()`: Initializes knowledge engine, creates MCP server, and handles JSON-RPC communication over stdin/stdout.
* **Functions:**
  * `handle_mcp_request(server, request)`: Routes MCP protocol requests to appropriate handlers
    - Handles "initialize", "tools/list", and "tools/call" methods
    - Returns protocol version, available tools, and tool execution results
* **Protocol Details:**
  * Uses MCP protocol version "2025-06-18"
  * Communicates via line-delimited JSON-RPC 2.0 over stdio
  * Provides tool discovery and invocation capabilities
* **Dependencies:**
  * Internal: `llmkg::mcp::llm_friendly_server::LLMFriendlyMCPServer`, `llmkg::core::knowledge_engine::KnowledgeEngine`
  * External: tokio, serde_json, clap, env_logger/tracing

### `llmkg_mcp_server_test.rs`

* **Purpose:** Test version of the MCP server with identical functionality to `llmkg_mcp_server.rs` but using simpler logging configuration (env_logger only).
* **Main Function:** Identical to `llmkg_mcp_server.rs`
* **Functions:** Same as `llmkg_mcp_server.rs`
* **Key Differences:**
  * Uses only env_logger instead of the enhanced logging system
  * Otherwise functionally identical to the main MCP server
* **Dependencies:** Same as `llmkg_mcp_server.rs` but without enhanced logging

### `llmkg_server.rs`

* **Purpose:** Basic monitoring dashboard server that provides system and application metrics visualization without brain-enhanced features.
* **Main Function:**
  * `main()`: Sets up metric collection for system and application metrics, then starts the dashboard server.
* **Configuration:**
  * `DashboardConfig`:
    - `http_port`: 8080
    - `websocket_port`: 8081
* **Collectors:**
  * `SystemMetricsCollector`: CPU, memory, disk, network, load metrics
  * `ApplicationMetricsCollector`: Performance, operations, errors, resources
* **Dependencies:**
  * Internal: `llmkg::monitoring::dashboard::*`, `llmkg::monitoring::metrics::*`, `llmkg::monitoring::collectors::*`
  * External: tokio

### `test_brain_metrics.rs`

* **Purpose:** Test utility for validating brain metrics collection functionality.
* **Main Function:**
  * `main()`: Creates test knowledge graph with entities and relationships, collects metrics, and verifies dynamic updates.
* **Test Workflow:**
  1. Creates 20 test entities with varying activation levels
  2. Establishes 30 relationships with different synaptic weights
  3. Collects brain metrics using BrainMetricsCollector
  4. Displays all brain_* metrics
  5. Updates some entity activations
  6. Re-collects metrics to verify changes
* **Metrics Validated:**
  * Entity and relationship counts
  * Average and maximum activation levels
  * Other brain-specific metrics
* **Dependencies:**
  * Internal: `llmkg::core::brain_enhanced_graph::*`, `llmkg::monitoring::*`
  * External: tokio

## 5. Database Interaction

The binaries do not directly interact with traditional databases. Instead, they work with in-memory knowledge graph structures (`BrainEnhancedKnowledgeGraph` and `KnowledgeEngine`) that persist data through internal mechanisms.

## 6. API Endpoints

### llmkg_api_server.rs
* **HTTP API:** Port 3001
* **Dashboard:** Port 8090
* **WebSocket:** Port 8081

### llmkg_brain_server.rs
* **HTTP Dashboard:** Port 8082
* **WebSocket Metrics:** Port 8083
* **React Dashboard:** Port 3001 (reference)

### llmkg_server.rs
* **HTTP Dashboard:** Port 8080
* **WebSocket Metrics:** Port 8081

### MCP Servers (stdio-based)
* **Protocol:** JSON-RPC 2.0 over stdin/stdout
* **Methods:**
  - `initialize`: Protocol handshake
  - `tools/list`: List available tools
  - `tools/call`: Execute tool with parameters

## 7. Key Variables and Logic

### Brain-Enhanced Features
* **Activation Levels:** Neural-inspired activation values (0.0-1.0) for entities
* **Synaptic Weights:** Connection strengths between entities
* **Embedding Dimension:** 384-dimensional vectors for entity representations
* **Max Nodes:** Default limit of 1,000,000 nodes

### MCP Protocol
* **Request Format:** `{ "jsonrpc": "2.0", "method": "...", "params": {...}, "id": ... }`
* **Response Format:** `{ "jsonrpc": "2.0", "result": {...}, "id": ... }` or error
* **Tool Invocation:** Converts MCP tool calls to internal LLMKG requests

### Monitoring Architecture
* **MetricRegistry:** Central storage for all metrics
* **Collectors:** Modular components that gather specific metric types
* **Real-time Updates:** WebSocket connections for live metric streaming

## 8. Dependencies

### Internal Dependencies
* **Core Modules:**
  - `llmkg::core::brain_enhanced_graph`
  - `llmkg::core::knowledge_engine`
  - `llmkg::core::types`
* **API/Server Modules:**
  - `llmkg::api::server`
  - `llmkg::mcp::llm_friendly_server`
  - `llmkg::mcp::shared_types`
* **Monitoring Modules:**
  - `llmkg::monitoring::dashboard`
  - `llmkg::monitoring::metrics`
  - `llmkg::monitoring::collectors`
  - `llmkg::monitoring::BrainMetricsCollector`
* **Utility Modules:**
  - `llmkg::enhanced_knowledge_storage::logging`
  - `llmkg::cli::Args`
  - `llmkg::error`

### External Dependencies (Key)
* **Async Runtime:** tokio (with full features)
* **Serialization:** serde, serde_json
* **CLI:** clap
* **Logging:** tracing, tracing-subscriber, env_logger, log
* **Web:** warp, tokio-tungstenite
* **Utilities:** chrono, uuid, rand
* **System:** sysinfo
* **Concurrency:** parking_lot, crossbeam, rayon, dashmap

### Binary Relationships
1. **API Server** (`llmkg_api_server.rs`) - Main user-facing API
2. **Brain Server** (`llmkg_brain_server.rs`) - Enhanced monitoring with neural features
3. **MCP Servers** (`llmkg_mcp_server.rs`, `llmkg_mcp_server_test.rs`) - LLM integration points
4. **Basic Server** (`llmkg_server.rs`) - Simple monitoring dashboard
5. **Test Utility** (`test_brain_metrics.rs`) - Validation tool for brain metrics

These binaries can be run independently but may complement each other in a full deployment (e.g., API server for data operations + Brain server for monitoring).