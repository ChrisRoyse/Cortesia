# MicroPhase 1: MCP Server Foundation Setup (15-20 Micro-Tasks)

**Total Duration**: 5-6 hours (15-20 micro-tasks × 15-20 minutes each)  
**Priority**: Critical - Blocking for all other phases  
**Prerequisites**: None

## Overview

Establish the core MCP server infrastructure through atomic micro-tasks, each delivering ONE concrete file in 15-20 minutes. Each task follows the established pattern with exact code specifications and clear verification criteria.

## Micro-Task Breakdown

---

## Micro-Task 1.1.1: Create Core MCP Server Imports and Structure
**Duration**: 15 minutes  
**Dependencies**: None  
**Input**: MCP server foundation requirements  
**Output**: Basic server struct with core imports  

### Task Prompt for AI
```
Create the core MCP server structure with essential imports:

```rust
use mcp_server::{Server, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error};

pub struct CortexKGMCPServer {
    // Core state - will be expanded in subsequent tasks
}

impl CortexKGMCPServer {
    pub async fn new() -> Result<Self> {
        info!("Initializing CortexKG MCP Server");
        Ok(Self {})
    }
}
```

Write ONE unit test verifying the server can be created.
```

**Expected Deliverable**: `src/mcp/cortex_kg_server.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.1.2: Add MCP Dependencies to Cargo.toml
**Duration**: 15 minutes  
**Dependencies**: Task 1.1.1  
**Input**: MCP server requirements  
**Output**: Core MCP dependencies added  

### Task Prompt for AI
```
Add MCP-specific dependencies to Cargo.toml:

```toml
[dependencies]
# MCP Server Dependencies
mcp-server = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
```

Verify the project compiles successfully with these dependencies.
```

**Expected Deliverable**: Updated `Cargo.toml`  
**Verification**: Compiles successfully  

---

## Micro-Task 1.1.3: Add Logging Dependencies to Cargo.toml
**Duration**: 15 minutes  
**Dependencies**: Task 1.1.2  
**Input**: Logging requirements  
**Output**: Logging dependencies added  

### Task Prompt for AI
```
Add logging and tracing dependencies to Cargo.toml:

```toml
# Logging Dependencies  
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
async-trait = "0.1"
```

Verify all dependencies resolve without conflicts.
```

**Expected Deliverable**: Updated `Cargo.toml`  
**Verification**: Compiles + no dependency conflicts  

---

## Micro-Task 1.2.1: Create Basic MCP Error Types
**Duration**: 18 minutes  
**Dependencies**: Task 1.1.3  
**Input**: Error handling requirements  
**Output**: Core error types structure  

### Task Prompt for AI
```
Create basic MCP error types with thiserror:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MCPServerError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Tool execution failed: {0}")]
    ToolExecutionError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
}

pub type MCPResult<T> = Result<T, MCPServerError>;
```

Write ONE unit test verifying error message formatting.
```

**Expected Deliverable**: `src/mcp/errors.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.2.2: Add Authentication and Validation Errors
**Duration**: 16 minutes  
**Dependencies**: Task 1.2.1  
**Input**: Security error requirements  
**Output**: Extended error types  

### Task Prompt for AI
```
Extend MCPServerError with authentication and validation errors:

```rust
#[derive(Error, Debug)]
pub enum MCPServerError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Tool execution failed: {0}")]
    ToolExecutionError(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    #[error("Validation failed: {0}")]
    ValidationError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Internal server error: {0}")]
    InternalError(String),
}
```

Write ONE test verifying each error variant can be created and displayed.
```

**Expected Deliverable**: Updated `src/mcp/errors.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.3.1: Create Server Configuration Structure
**Duration**: 17 minutes  
**Dependencies**: Task 1.2.2  
**Input**: Server configuration requirements  
**Output**: ServerConfig struct with defaults  

### Task Prompt for AI
```
Create the ServerConfig structure:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 1000,
            request_timeout_ms: 30000,
        }
    }
}
```

Write ONE test verifying serialization and default values.
```

**Expected Deliverable**: `src/mcp/config.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.3.2: Create Neuromorphic Configuration Structure
**Duration**: 18 minutes  
**Dependencies**: Task 1.3.1  
**Input**: Neuromorphic processing requirements  
**Output**: NeuromorphicConfig struct  

### Task Prompt for AI
```
Add the NeuromorphicConfig structure to config.rs:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct NeuromorphicConfig {
    pub ttfs_precision_ms: f32,
    pub cortical_columns: usize,
    pub network_pool_size: usize,
    pub stdp_learning_rate: f32,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            ttfs_precision_ms: 0.1,
            cortical_columns: 4,
            network_pool_size: 1024,
            stdp_learning_rate: 0.01,
        }
    }
}
```

Write ONE test verifying neuromorphic configuration defaults.
```

**Expected Deliverable**: Updated `src/mcp/config.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.3.3: Create Performance Configuration Structure
**Duration**: 16 minutes  
**Dependencies**: Task 1.3.2  
**Input**: Performance optimization requirements  
**Output**: PerformanceConfig struct  

### Task Prompt for AI
```
Add the PerformanceConfig structure to config.rs:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_simd: bool,
    pub batch_size: usize,
    pub connection_pool_size: usize,
    pub cache_size_mb: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            batch_size: 32,
            connection_pool_size: 100,
            cache_size_mb: 512,
        }
    }
}
```

Write ONE test verifying performance configuration values are sensible.
```

**Expected Deliverable**: Updated `src/mcp/config.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.3.4: Create Security Configuration Structure
**Duration**: 19 minutes  
**Dependencies**: Task 1.3.3  
**Input**: Security and authentication requirements  
**Output**: SecurityConfig and main MCPServerConfig  

### Task Prompt for AI
```
Add SecurityConfig and complete MCPServerConfig:

```rust
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_oauth: bool,
    pub jwt_secret_path: PathBuf,
    pub session_timeout_minutes: u64,
    pub rate_limit_per_minute: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_oauth: true,
            jwt_secret_path: PathBuf::from("secrets/jwt.key"),
            session_timeout_minutes: 60,
            rate_limit_per_minute: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub server: ServerConfig,
    pub neuromorphic: NeuromorphicConfig,
    pub performance: PerformanceConfig,
    pub security: SecurityConfig,
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            neuromorphic: NeuromorphicConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}
```

Write ONE test verifying complete configuration can be serialized to JSON.
```

**Expected Deliverable**: Updated `src/mcp/config.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.4.1: Create Basic Logging Setup
**Duration**: 15 minutes  
**Dependencies**: Task 1.3.4  
**Input**: Structured logging requirements  
**Output**: Logging initialization module  

### Task Prompt for AI
```
Create the logging initialization module:

```rust
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub fn init_logging() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    info!("Logging initialized");
    Ok(())
}
```

Write ONE test verifying logging can be initialized without errors.
```

**Expected Deliverable**: `src/mcp/logging.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.4.2: Add Server Lifecycle Methods to Core Server
**Duration**: 18 minutes  
**Dependencies**: Tasks 1.1.1, 1.3.4, 1.4.1  
**Input**: Server structure, config, and logging  
**Output**: Start and shutdown methods  

### Task Prompt for AI
```
Add lifecycle methods to CortexKGMCPServer:

```rust
use crate::mcp::config::MCPServerConfig;
use crate::mcp::logging;

impl CortexKGMCPServer {
    pub async fn new() -> Result<Self> {
        logging::init_logging()?;
        info!("Initializing CortexKG MCP Server");
        Ok(Self {})
    }
    
    pub async fn start(&self, port: u16) -> Result<()> {
        info!("Starting MCP server on port {}", port);
        // Server startup logic placeholder
        Ok(())
    }
    
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP server gracefully");
        // Graceful shutdown logic placeholder
        Ok(())
    }
}
```

Write ONE test verifying server lifecycle (new -> start -> shutdown).
```

**Expected Deliverable**: Updated `src/mcp/cortex_kg_server.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.5.1: Create MCP Module Declaration Structure
**Duration**: 15 minutes  
**Dependencies**: All previous tasks  
**Input**: All MCP module files created  
**Output**: Module declarations and exports  

### Task Prompt for AI
```
Create the main MCP module file with proper declarations:

```rust
pub mod cortex_kg_server;
pub mod config;
pub mod errors;
pub mod logging;

// Re-export main types for convenience
pub use cortex_kg_server::CortexKGMCPServer;
pub use config::MCPServerConfig;
pub use errors::{MCPServerError, MCPResult};
```

Write ONE test verifying all exports are accessible.
```

**Expected Deliverable**: `src/mcp/mod.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 1.5.2: Create Basic Integration Test Structure
**Duration**: 17 minutes  
**Dependencies**: Task 1.5.1  
**Input**: Complete MCP foundation modules  
**Output**: Integration test framework  

### Task Prompt for AI
```
Create basic integration tests for MCP foundation:

```rust
use cortex_kg::mcp::CortexKGMCPServer;

#[tokio::test]
async fn test_server_creation() {
    let server = CortexKGMCPServer::new().await;
    assert!(server.is_ok());
}

#[tokio::test] 
async fn test_server_basic_lifecycle() {
    let server = CortexKGMCPServer::new().await.unwrap();
    
    // Test that we can start without errors
    let start_result = server.start(8081).await;
    assert!(start_result.is_ok());
    
    // Test graceful shutdown
    let shutdown_result = server.shutdown().await;
    assert!(shutdown_result.is_ok());
}
```

Verify both tests pass and server lifecycle works correctly.
```

**Expected Deliverable**: `tests/mcp_foundation_test.rs`  
**Verification**: Compiles + 2 tests pass  

---

## Micro-Task 1.6.1: Create Configuration Integration Test
**Duration**: 16 minutes  
**Dependencies**: Task 1.5.2  
**Input**: Configuration modules and test framework  
**Output**: Configuration validation tests  

### Task Prompt for AI
```
Add configuration integration tests:

```rust
use cortex_kg::mcp::config::MCPServerConfig;

#[test]
fn test_default_configuration_validity() {
    let config = MCPServerConfig::default();
    
    // Verify server config is sensible
    assert!(!config.server.host.is_empty());
    assert!(config.server.port > 0);
    assert!(config.server.max_connections > 0);
    
    // Verify neuromorphic config
    assert!(config.neuromorphic.ttfs_precision_ms > 0.0);
    assert!(config.neuromorphic.cortical_columns > 0);
    
    // Verify performance config
    assert!(config.performance.batch_size > 0);
    assert!(config.performance.cache_size_mb > 0);
}

#[test]
fn test_configuration_serialization() {
    let config = MCPServerConfig::default();
    
    // Test JSON serialization roundtrip
    let json = serde_json::to_string(&config).expect("Should serialize");
    let deserialized: MCPServerConfig = serde_json::from_str(&json).expect("Should deserialize");
    
    assert_eq!(config.server.port, deserialized.server.port);
}
```

Ensure configuration can be loaded, validated, and serialized.
```

**Expected Deliverable**: Updated `tests/mcp_foundation_test.rs`  
**Verification**: Compiles + 2 additional tests pass  

---

## Micro-Task 1.6.2: Create Error Handling Integration Test
**Duration**: 18 minutes  
**Dependencies**: Task 1.6.1  
**Input**: Error types and test framework  
**Output**: Error handling validation tests  

### Task Prompt for AI
```
Add error handling integration tests:

```rust
use cortex_kg::mcp::errors::{MCPServerError, MCPResult};

#[test]
fn test_error_types_and_messages() {
    let init_error = MCPServerError::InitializationError("test failure".to_string());
    assert!(init_error.to_string().contains("Initialization failed"));
    assert!(init_error.to_string().contains("test failure"));
    
    let tool_error = MCPServerError::ToolExecutionError("tool failed".to_string());
    assert!(tool_error.to_string().contains("Tool execution failed"));
    
    let auth_error = MCPServerError::AuthenticationError("bad token".to_string());
    assert!(auth_error.to_string().contains("Authentication failed"));
}

#[test] 
fn test_error_result_type() {
    fn failing_function() -> MCPResult<String> {
        Err(MCPServerError::ValidationError("invalid input".to_string()))
    }
    
    let result = failing_function();
    assert!(result.is_err());
    
    match result {
        Err(MCPServerError::ValidationError(msg)) => {
            assert!(msg.contains("invalid input"));
        }
        _ => panic!("Expected ValidationError"),
    }
}
```

Verify all error types work correctly and provide useful messages.
```

**Expected Deliverable**: Updated `tests/mcp_foundation_test.rs`  
**Verification**: Compiles + 2 additional tests pass  

---

## Micro-Task 1.7.1: Create Foundation Performance Test
**Duration**: 19 minutes  
**Dependencies**: Task 1.6.2  
**Input**: Complete foundation with all tests  
**Output**: Performance verification test  

### Task Prompt for AI
```
Add performance tests for foundation components:

```rust
use std::time::Instant;
use cortex_kg::mcp::{CortexKGMCPServer, config::MCPServerConfig};

#[tokio::test]
async fn test_server_creation_performance() {
    let start = Instant::now();
    
    for _ in 0..100 {
        let _server = CortexKGMCPServer::new().await.unwrap();
    }
    
    let elapsed = start.elapsed();
    let per_creation = elapsed.as_millis() / 100;
    
    // Server creation should be < 10ms per instance
    assert!(per_creation < 10, "Server creation took {}ms, expected <10ms", per_creation);
}

#[test]
fn test_configuration_serialization_performance() {
    let config = MCPServerConfig::default();
    let start = Instant::now();
    
    for _ in 0..1000 {
        let _json = serde_json::to_string(&config).unwrap();
    }
    
    let elapsed = start.elapsed();
    let per_serialization = elapsed.as_nanos() / 1000;
    
    // Config serialization should be < 100μs per operation  
    assert!(per_serialization < 100_000, "Config serialization took {}ns, expected <100μs", per_serialization);
}
```

Verify foundation components meet performance requirements.
```

**Expected Deliverable**: Updated `tests/mcp_foundation_test.rs`  
**Verification**: Compiles + 2 performance tests pass  

---

## Micro-Task 1.7.2: Create Foundation Documentation Test
**Duration**: 16 minutes  
**Dependencies**: Task 1.7.1  
**Input**: Complete foundation implementation  
**Output**: Documentation verification  

### Task Prompt for AI
```
Add documentation tests to verify API documentation:

```rust
/// Test that demonstrates basic MCP server usage
/// 
/// # Example
/// 
/// ```
/// use cortex_kg::mcp::CortexKGMCPServer;
/// 
/// # tokio_test::block_on(async {
/// let server = CortexKGMCPServer::new().await.unwrap();
/// server.start(8080).await.unwrap();
/// server.shutdown().await.unwrap();
/// # });
/// ```
#[test]
fn test_basic_usage_documentation() {
    // This test verifies the documentation example compiles
    // The actual test is in the doc comment above
}

/// Test configuration usage documentation
/// 
/// # Example
/// 
/// ```
/// use cortex_kg::mcp::config::MCPServerConfig;
/// 
/// let config = MCPServerConfig::default();
/// assert!(config.server.port > 0);
/// 
/// let json = serde_json::to_string(&config).unwrap();
/// let parsed: MCPServerConfig = serde_json::from_str(&json).unwrap();
/// assert_eq!(config.server.port, parsed.server.port);
/// ```
#[test] 
fn test_configuration_usage_documentation() {
    // This test verifies the configuration documentation example compiles
}
```

Run `cargo test --doc` to verify all documentation examples work.
```

**Expected Deliverable**: Updated `tests/mcp_foundation_test.rs`  
**Verification**: Compiles + doc tests pass + 2 regular tests pass

## Summary

**Total Micro-Tasks**: 18 tasks  
**Total Time**: ~5-6 hours (18 × 15-20 minutes)  
**Deliverables**:
- `src/mcp/cortex_kg_server.rs` (core server structure)
- `src/mcp/config.rs` (complete configuration system)  
- `src/mcp/errors.rs` (comprehensive error handling)
- `src/mcp/logging.rs` (structured logging setup)
- `src/mcp/mod.rs` (module organization)
- `tests/mcp_foundation_test.rs` (integration tests)
- Updated `Cargo.toml` (all dependencies)

**Key Benefits**:
- Each micro-task has ONE concrete deliverable
- Each micro-task can be completed in 15-20 minutes
- Clear verification criteria for each task
- Incremental progress with working code at each step
- AI can complete each task independently
- Follows established Phase 0/1/2 micro-task patterns

## Validation Checklist

**Foundation Components**:
- [ ] All 18 micro-tasks completed
- [ ] All tests passing (20+ individual unit/integration tests)
- [ ] Performance benchmarks <10ms server creation, <100μs config serialization
- [ ] All Rust files compile without warnings
- [ ] Dependencies resolve correctly (MCP, logging, async)
- [ ] Configuration can be loaded, validated, and serialized
- [ ] Error handling comprehensive with proper message formatting
- [ ] Structured logging outputs correctly with configurable levels
- [ ] Module organization follows Rust conventions
- [ ] Documentation examples work and compile

**Quality Gates**:
- [ ] Server lifecycle (new -> start -> shutdown) verified
- [ ] Configuration serialization roundtrip tested
- [ ] All error types can be created and displayed correctly
- [ ] Performance meets <10ms server creation requirement
- [ ] Integration with existing codebase verified
- [ ] Ready for MicroPhase 2 neuromorphic core integration

## Next Phase Dependencies

This micro-phase provides the critical foundation for:
- **MicroPhase 2**: Neuromorphic core integration (depends on server structure)
- **MicroPhase 3**: Tool schema definitions (depends on error handling)
- **MicroPhase 5**: Authentication implementation (depends on security config)
- **MicroPhase 6**: Performance optimization (depends on performance config)
- **MicroPhase 7**: Testing framework (depends on integration test structure)

**Critical Path**: This phase is blocking for ALL subsequent phases. No other MCP-related micro-phases can begin until this foundation is complete and all 18 micro-tasks pass their verification criteria.

This micro-task breakdown transforms 6 macro-tasks (10-30 minutes each, 90 minutes total) into 18 focused micro-tasks (15-20 minutes each, 300+ minutes total) with much more granular control, better verification, and true AI-completability following the established patterns from Phase 0, Phase 1, and Phase 2.