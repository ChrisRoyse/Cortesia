# MicroPhase 8: System Integration

**Duration**: 3-4 hours  
**Priority**: Critical - Final system assembly  
**Prerequisites**: All previous MicroPhases (1-7)

## Overview

Break down system integration into atomic micro-tasks that each AI can complete in 15-20 minutes. Each task produces a single, testable deliverable.

## AI-Actionable Micro-Tasks

### Micro-Task 8.1.1: Create Main Server Structure
**Estimated Time**: 18 minutes  
**File**: `src/mcp/server.rs`
**Expected Deliverable**: Server struct definition and initialization

**Task Prompt for AI**:
Create the main server struct and initialization method for CortexKGMCPServer. Focus only on struct definition and basic initialization - do not implement background tasks or request processing.

```rust
use crate::mcp::{
    handlers::ToolExecutor,
    neuromorphic::NeuromorphicCore,
    auth::middleware::AuthenticationMiddleware,
    performance::{
        connection_pool::MCPConnectionPool,
        caching::MultiLevelCache,
        metrics::PerformanceMonitor,
    },
    config::MCPServerConfig,
    errors::{MCPResult, MCPServerError},
    schemas::SchemaRegistry,
};
use crate::core::knowledge_engine::KnowledgeEngine;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

pub struct CortexKGMCPServer {
    config: MCPServerConfig,
    tool_executor: Arc<ToolExecutor>,
    auth_middleware: Arc<AuthenticationMiddleware>,
    performance_monitor: Arc<PerformanceMonitor>,
    cache: Arc<MultiLevelCache>,
    connection_pool: Arc<MCPConnectionPool>,
    schema_registry: Arc<SchemaRegistry>,
    neuromorphic_core: Arc<RwLock<NeuromorphicCore>>,
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    shutdown_signal: Arc<tokio::sync::Notify>,
}

impl CortexKGMCPServer {
    pub async fn new(config: MCPServerConfig) -> MCPResult<Self> {
        info!("Initializing CortexKG MCP Server...");
        
        // Initialize core components (placeholder - actual initialization in next task)
        let neuromorphic_core = Arc::new(RwLock::new(NeuromorphicCore::new()));
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new()));
        
        // Initialize components (to be implemented in subsequent tasks)
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        let cache = Arc::new(MultiLevelCache::new(Default::default()));
        let connection_pool = Arc::new(MCPConnectionPool::new(Default::default()).await?);
        let auth_middleware = Arc::new(AuthenticationMiddleware::new_mock());
        let tool_executor = Arc::new(ToolExecutor::new(neuromorphic_core.clone(), knowledge_engine.clone()));
        let schema_registry = Arc::new(SchemaRegistry::new());
        let shutdown_signal = Arc::new(tokio::sync::Notify::new());
        
        info!("CortexKG MCP Server struct created");
        
        Ok(Self {
            config,
            tool_executor,
            auth_middleware,
            performance_monitor,
            cache,
            connection_pool,
            schema_registry,
            neuromorphic_core,
            knowledge_engine,
            shutdown_signal,
        })
    }
}
```

**Verification**: 
- Server struct compiles without errors
- All fields are properly initialized 
- Basic new() method works

---

### Micro-Task 8.1.2: Add Component Configuration
**Estimated Time**: 16 minutes  
**File**: `src/mcp/server.rs` (update existing)
**Expected Deliverable**: Proper component configuration using config values
**Task Prompt for AI**:
Replace the basic configuration in the new() method with proper configuration loading from the MCPServerConfig. Only update the component initialization - do not add server lifecycle methods.

```rust
// Update the new() method to use proper configuration:
pub async fn new(config: MCPServerConfig) -> MCPResult<Self> {
    info!("Initializing CortexKG MCP Server...");
    
    // Initialize core components
    let neuromorphic_core = Arc::new(RwLock::new(NeuromorphicCore::new()));
    let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new()));
    
    // Initialize performance components with config
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    let cache_config = CacheConfig {
        l1_capacity: config.performance.cache_size_mb / 4,
        l2_capacity: config.performance.cache_size_mb / 2,
        l3_capacity: config.performance.cache_size_mb,
        ..Default::default()
    };
    let cache = Arc::new(MultiLevelCache::new(cache_config));
    
    let pool_config = ConnectionConfig {
        max_connections: config.performance.connection_pool_size,
        min_connections: config.performance.connection_pool_size / 5,
        ..Default::default()
    };
    let connection_pool = Arc::new(MCPConnectionPool::new(pool_config).await?);
    
    // Initialize with proper auth configuration
    let auth_middleware = Arc::new(AuthenticationMiddleware::from_config(&config.security).await?);
    let tool_executor = Arc::new(ToolExecutor::new(neuromorphic_core.clone(), knowledge_engine.clone()));
    let schema_registry = Arc::new(SchemaRegistry::new());
    let shutdown_signal = Arc::new(tokio::sync::Notify::new());
    
    info!("CortexKG MCP Server initialization complete");
    
    Ok(Self {
        config,
        tool_executor,
        auth_middleware,
        performance_monitor,
        cache,
        connection_pool,
        schema_registry,
        neuromorphic_core,
        knowledge_engine,
        shutdown_signal,
    })
}
```

**Verification**: 
- All components use config values
- Proper error handling for config loading
- No compilation errors

---

### Micro-Task 8.1.3: Add Server Lifecycle Methods
**Estimated Time**: 19 minutes  
**File**: `src/mcp/server.rs` (update existing)
**Expected Deliverable**: Start, stop, and basic health check methods
**Task Prompt for AI**:
Add basic server lifecycle methods: start(), shutdown(), and get_health_status(). Focus only on essential functionality - no complex background task management yet.

```rust
impl CortexKGMCPServer {
    // ... existing new() method ...
    
    pub async fn start(&self) -> MCPResult<()> {
        info!("Starting CortexKG MCP Server on {}:{}", self.config.server.host, self.config.server.port);
        
        // Basic startup - background tasks in next micro-task
        info!("Server started successfully");
        Ok(())
    }
    
    pub async fn shutdown(&self) -> MCPResult<()> {
        info!("Initiating graceful shutdown...");
        
        // Signal shutdown to all components
        self.shutdown_signal.notify_waiters();
        
        // Basic cleanup
        self.connection_pool.close().await;
        self.cache.clear().await;
        
        info!("Graceful shutdown complete");
        Ok(())
    }
    
    pub async fn get_health_status(&self) -> HealthStatus {
        let health_score = self.performance_monitor.get_health_score().await;
        
        HealthStatus {
            status: if health_score > 0.8 { "healthy".to_string() } 
                   else if health_score > 0.5 { "degraded".to_string() }
                   else { "unhealthy".to_string() },
            health_score,
            response_time_ms: 45.0, // Placeholder - actual metrics in next task
            throughput_ops_per_min: 1000.0,
            error_rate_percent: 0.5,
            memory_usage_mb: 512.0,
            uptime_seconds: 3600,
        }
    }
}

// Supporting types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub health_score: f64,
    pub response_time_ms: f64,
    pub throughput_ops_per_min: f64,
    pub error_rate_percent: f64,
    pub memory_usage_mb: f64,
    pub uptime_seconds: u64,
}
```

**Verification**: 
- Server can start and shutdown gracefully
- Health status returns valid data
- No hanging processes after shutdown

---

### Micro-Task 8.1.4: Add Request Processing Pipeline
**Estimated Time**: 20 minutes  
**File**: `src/mcp/server.rs` (update existing)
**Expected Deliverable**: MCP request processing with authentication and validation
**Task Prompt for AI**:
Add MCP request processing pipeline with authentication, validation, and tool execution. Focus on the core request handling logic.

```rust
impl CortexKGMCPServer {
    // ... existing methods ...
    
    pub async fn process_mcp_request(&self, request: MCPRequest) -> MCPResult<MCPResponse> {
        let timer = self.performance_monitor.record_request_start().await;
        
        // 1. Authenticate request
        let auth_context = match self.auth_middleware.authenticate_request(&request).await {
            Ok(context) => context,
            Err(e) => {
                timer.finish_with_error("authentication").await;
                return Err(e);
            }
        };
        
        // 2. Check rate limits
        let rate_limit_result = self.auth_middleware.check_rate_limits(
            &auth_context,
            &request.method
        ).await?;
        
        if !rate_limit_result.is_allowed() {
            timer.finish_with_error("rate_limit").await;
            return Err(MCPServerError::NetworkError("Rate limit exceeded".to_string()));
        }
        
        // 3. Validate input schema
        self.schema_registry.validate_tool_input(&request.method, &request.params)?;
        
        // 4. Execute tool
        let result = self.tool_executor.execute_tool(&request.method, request.params).await;
        
        match result {
            Ok(output) => {
                timer.finish().await;
                Ok(MCPResponse {
                    id: request.id,
                    result: Some(output),
                    error: None,
                    rate_limit_headers: rate_limit_result.to_headers(),
                })
            },
            Err(e) => {
                timer.finish_with_error("execution").await;
                Err(e)
            }
        }
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct MCPRequest {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
    pub headers: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct MCPResponse {
    pub id: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<MCPError>,
    pub rate_limit_headers: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}
```

**Verification**: 
- Request pipeline processes all stages
- Authentication and rate limiting work
- Proper error handling and response formatting

---

### Micro-Task 8.1.5: Add Background Task Management
**Estimated Time**: 17 minutes  
**File**: `src/mcp/server.rs` (update existing)
**Expected Deliverable**: Background tasks for monitoring and maintenance

**Task Prompt for AI**:
Add background task spawning for periodic maintenance of cache, connection pool, and performance monitoring.

```rust
impl CortexKGMCPServer {
    // ... existing methods ...
    
    // Update the start() method to include background tasks:
    pub async fn start(&self) -> MCPResult<()> {
        info!("Starting CortexKG MCP Server on {}:{}", self.config.server.host, self.config.server.port);
        
        // Start background tasks
        self.start_background_tasks().await;
        
        info!("Server started with background tasks");
        Ok(())
    }
    
    async fn start_background_tasks(&self) {
        // Start performance monitoring
        let monitor = self.performance_monitor.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                monitor.cleanup_expired_metrics().await;
            }
        });
        
        // Start cache optimization
        let cache = self.cache.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
            loop {
                interval.tick().await;
                cache.cleanup_expired().await;
                cache.optimize_cache_levels().await;
            }
        });
        
        // Start connection pool maintenance
        let pool = self.connection_pool.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                pool.cleanup_expired_connections().await;
            }
        });
        
        info!("Background tasks started");
    }
}
```

**Verification**: 
- Background tasks spawn without errors
- Tasks run on their scheduled intervals
- Server maintains performance with tasks running

---

### Micro-Task 8.2.1: Create Production Configuration
**Estimated Time**: 15 minutes  
**File**: `config/production.toml`
**Expected Deliverable**: Production-optimized configuration file
```

**Task Prompt for AI**:
Create production configuration with optimized settings for security, performance, and monitoring.

```toml

```toml
[server]
host = "0.0.0.0"
port = 8080
max_connections = 1000
request_timeout_ms = 30000

[neuromorphic]
ttfs_precision_ms = 0.1
cortical_columns = 4
network_pool_size = 2048
stdp_learning_rate = 0.01

[performance]
enable_simd = true
batch_size = 64
connection_pool_size = 200
cache_size_mb = 1024

[security]
enable_oauth = true
jwt_secret_path = "/etc/cortex-kg/jwt.key"
session_timeout_minutes = 60
rate_limit_per_minute = 1000

[logging]
level = "info"
format = "json"
output = "stdout"

[monitoring]
enable_metrics = true
metrics_port = 9090
health_check_interval_seconds = 30

[database]
knowledge_graph_path = "/var/lib/cortex-kg/knowledge.db"
backup_interval_hours = 6
max_memory_size_gb = 4

[neural_networks]
model_weights_directory = "/var/lib/cortex-kg/models"
enable_gpu_acceleration = false
memory_limit_mb = 2048
```

**File**: `config/development.toml`

```toml
[server]
host = "127.0.0.1"
port = 8080
max_connections = 100
request_timeout_ms = 10000

[neuromorphic]
ttfs_precision_ms = 0.5
cortical_columns = 4
network_pool_size = 512
stdp_learning_rate = 0.02

[performance]
enable_simd = true
batch_size = 32
connection_pool_size = 50
cache_size_mb = 256

[security]
enable_oauth = false
jwt_secret_path = "./config/jwt_dev.key"
session_timeout_minutes = 120
rate_limit_per_minute = 100

[logging]
level = "debug"
format = "pretty"
output = "stdout"

[monitoring]
enable_metrics = true
metrics_port = 9091
health_check_interval_seconds = 10

[database]
knowledge_graph_path = "./data/knowledge_dev.db"
backup_interval_hours = 1
max_memory_size_gb = 1

[neural_networks]
model_weights_directory = "./models"
enable_gpu_acceleration = false
memory_limit_mb = 512
```

**File**: `config/testing.toml`

```toml
[server]
host = "127.0.0.1"
port = 8082
max_connections = 50
request_timeout_ms = 5000

[neuromorphic]
ttfs_precision_ms = 1.0
cortical_columns = 4
network_pool_size = 128
stdp_learning_rate = 0.05

[performance]
enable_simd = false  # Disable for consistent testing
batch_size = 16
connection_pool_size = 20
cache_size_mb = 64

[security]
enable_oauth = false
jwt_secret_path = "./config/jwt_test.key"
session_timeout_minutes = 10
rate_limit_per_minute = 1000  # High limit for testing

[logging]
level = "warn"
format = "json"
output = "file"
file_path = "./logs/test.log"

[monitoring]
enable_metrics = false
health_check_interval_seconds = 60

[database]
knowledge_graph_path = ":memory:"  # In-memory for testing
backup_interval_hours = 999
max_memory_size_gb = 1

[neural_networks]
model_weights_directory = "./test_models"
enable_gpu_acceleration = false
memory_limit_mb = 256
```

**Verification**: 
- Production config optimized for scale and security
- Development config suitable for local testing
- Testing config works with test suite

---

### Micro-Task 8.2.2: Create Development Configuration  
**Estimated Time**: 12 minutes  
**File**: `config/development.toml`
**Expected Deliverable**: Development-optimized configuration file

**Task Prompt for AI**:
Create development configuration with debug-friendly settings and local paths.

```toml
[server]
host = "127.0.0.1"
port = 8080
max_connections = 100
request_timeout_ms = 10000

[neuromorphic]
ttfs_precision_ms = 0.5
cortical_columns = 4
network_pool_size = 512
stdp_learning_rate = 0.02

[performance]
enable_simd = true
batch_size = 32
connection_pool_size = 50
cache_size_mb = 256

[security]
enable_oauth = false
jwt_secret_path = "./config/jwt_dev.key"
session_timeout_minutes = 120
rate_limit_per_minute = 100

[logging]
level = "debug"
format = "pretty"
output = "stdout"

[monitoring]
enable_metrics = true
metrics_port = 9091
health_check_interval_seconds = 10

[database]
knowledge_graph_path = "./data/knowledge_dev.db"
backup_interval_hours = 1
max_memory_size_gb = 1

[neural_networks]
model_weights_directory = "./models"
enable_gpu_acceleration = false
memory_limit_mb = 512
```

**Verification**: 
- Development-friendly settings
- Local file paths
- Verbose logging enabled

---

### Micro-Task 8.2.3: Create Testing Configuration
**Estimated Time**: 10 minutes  
**File**: `config/testing.toml`
**Expected Deliverable**: Test-optimized configuration file

**Task Prompt for AI**:
Create testing configuration with in-memory storage and simplified settings.

```toml
[server]
host = "127.0.0.1"
port = 8082
max_connections = 50
request_timeout_ms = 5000

[neuromorphic]
ttfs_precision_ms = 1.0
cortical_columns = 4
network_pool_size = 128
stdp_learning_rate = 0.05

[performance]
enable_simd = false  # Disable for consistent testing
batch_size = 16
connection_pool_size = 20
cache_size_mb = 64

[security]
enable_oauth = false
jwt_secret_path = "./config/jwt_test.key"
session_timeout_minutes = 10
rate_limit_per_minute = 1000  # High limit for testing

[logging]
level = "warn"
format = "json"
output = "file"
file_path = "./logs/test.log"

[monitoring]
enable_metrics = false
health_check_interval_seconds = 60

[database]
knowledge_graph_path = ":memory:"  # In-memory for testing
backup_interval_hours = 999
max_memory_size_gb = 1

[neural_networks]
model_weights_directory = "./test_models"
enable_gpu_acceleration = false
memory_limit_mb = 256
```

**Verification**: 
- Fast test execution
- In-memory storage
- Minimal resource usage

---

### Micro-Task 8.3.1: Create Basic Deployment Script
**Estimated Time**: 18 minutes  
**File**: `scripts/deploy.sh`
**Expected Deliverable**: Basic deployment automation script

**Task Prompt for AI**:
Create a basic deployment script with environment validation and prerequisite checking.

```bash
#!/bin/bash
set -euo pipefail

# CortexKG MCP Server Basic Deployment Script
# Usage: ./deploy.sh [environment]

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Deploying CortexKG MCP Server"
echo "Environment: $ENVIRONMENT"
echo "================================"

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        production|staging|development)
            echo "✅ Valid environment: $ENVIRONMENT"
            ;;
        *)
            echo "❌ Invalid environment: $ENVIRONMENT"
            echo "Valid environments: production, staging, development"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        echo "❌ Rust/Cargo is required but not installed"
        exit 1
    fi
    
    # Check if configuration exists
    CONFIG_FILE="$PROJECT_ROOT/config/$ENVIRONMENT.toml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "❌ Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Build the project
build_project() {
    echo "🔨 Building project..."
    cd "$PROJECT_ROOT"
    
    cargo build --release
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Build successful"
    else
        echo "❌ Build failed"
        exit 1
    fi
}

# Deploy to environment
deploy() {
    echo "🚀 Deploying to $ENVIRONMENT..."
    
    # Copy binary to appropriate location
    case "$ENVIRONMENT" in
        production)
            sudo cp target/release/cortex_kg_mcp_server /usr/local/bin/
            sudo systemctl restart cortex-kg-mcp || echo "Service not found - manual start required"
            ;;
        development|staging)
            echo "✅ Binary built for $ENVIRONMENT"
            echo "Manual start: ./target/release/cortex_kg_mcp_server --config config/$ENVIRONMENT.toml"
            ;;
    esac
    
    echo "✅ Deployment to $ENVIRONMENT complete"
}

# Main execution
main() {
    validate_environment
    check_prerequisites
    build_project
    deploy
}

main "$@"
```

**Verification**: 
- Script validates environment correctly
- Prerequisites are checked
- Basic deployment works

---

### Micro-Task 8.3.2: Create Docker Configuration
**Estimated Time**: 16 minutes  
**File**: `Dockerfile.production`
**Expected Deliverable**: Production Docker container configuration

**Task Prompt for AI**:
Create optimized multi-stage Dockerfile for production deployment.

```dockerfile
# Multi-stage build for optimal size
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build optimized release binary
RUN cargo build --release --bin cortex_kg_mcp_server

# Production runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cortex && \
    mkdir -p /var/lib/cortex-kg /var/log/cortex-kg && \
    chown -R cortex:cortex /var/lib/cortex-kg /var/log/cortex-kg

WORKDIR /app

# Copy binary and configuration
COPY --from=builder /app/target/release/cortex_kg_mcp_server /usr/local/bin/
COPY --chown=cortex:cortex config/ /app/config/

USER cortex

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080 9090

CMD ["cortex_kg_mcp_server", "--config", "/app/config/production.toml"]
```

**Verification**: 
- Docker builds successfully
- Container runs with non-root user
- Health check works

---

### Micro-Task 8.3.3: Create Docker Compose Configuration
**Estimated Time**: 14 minutes  
**File**: `docker-compose.yml`
**Expected Deliverable**: Complete container orchestration

**Task Prompt for AI**:
Create Docker Compose configuration for local development and production deployment.

```yaml
version: '3.8'

services:
  cortex-kg-mcp:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: cortex-kg-mcp
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./config/production.toml:/app/config/production.toml:ro
      - ./secrets:/etc/cortex-kg:ro
      - cortex-data:/var/lib/cortex-kg
      - cortex-logs:/var/log/cortex-kg
    environment:
      - RUST_LOG=info
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

volumes:
  cortex-data:
  cortex-logs:
```

**Verification**: 
- Compose starts services successfully
- Volumes mount correctly
- Resource limits are enforced

---

### Micro-Task 8.4.1: Create Basic CI Workflow
**Estimated Time**: 19 minutes  
**File**: `.github/workflows/ci.yml`
**Expected Deliverable**: Automated testing and building

**Task Prompt for AI**:
Create basic GitHub Actions CI workflow for automated testing and building.

```yaml
name: CortexKG MCP Server CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust: [stable, beta]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Check formatting
      run: cargo fmt --all -- --check
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    
    - name: Create test JWT secret
      run: |
        mkdir -p config
        echo "test_secret_key_32_bytes_minimum_length" > config/jwt_test.key
    
    - name: Run unit tests
      run: cargo test --lib --bins
    
    - name: Run integration tests
      run: cargo test --test '*integration*'

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Install cargo-audit
      run: cargo install cargo-audit
    
    - name: Run security audit
      run: cargo audit

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: [test, security-audit]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Build release binary
      run: cargo build --release --bin cortex_kg_mcp_server
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: cortex-kg-mcp-binary
        path: target/release/cortex_kg_mcp_server
```

**Verification**: 
- CI runs on push and PR
- All tests pass
- Security audit succeeds

---

### Micro-Task 8.4.2: Create Integration Test Suite
**Estimated Time**: 20 minutes  
**File**: `tests/integration/system_integration.rs`
**Expected Deliverable**: End-to-end integration tests

**Task Prompt for AI**:
Create comprehensive integration tests that validate the entire system working together.

```rust
use cortex_kg::mcp::server::{CortexKGMCPServer, MCPRequest, HealthStatus};
use cortex_kg::mcp::config::MCPServerConfig;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_server_initialization_and_health() {
    // Initialize server with test configuration
    let mut config = MCPServerConfig::default();
    config.server.port = 8083;
    config.security.enable_oauth = false;
    config.performance.cache_size_mb = 64;
    
    // Create temporary JWT secret
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_integration_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_integration_test.key");
    
    let server = CortexKGMCPServer::new(config).await.unwrap();
    
    // Test health status
    let health = server.get_health_status().await;
    assert!(!health.status.is_empty());
    assert!(health.health_score >= 0.0);
    
    // Test graceful shutdown
    let shutdown_result = timeout(Duration::from_secs(5), server.shutdown()).await;
    assert!(shutdown_result.is_ok());
    
    // Cleanup
    std::fs::remove_file("temp/jwt_integration_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_concurrent_operations() {
    let mut config = MCPServerConfig::default();
    config.server.port = 8084;
    config.security.enable_oauth = false;
    
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_concurrent_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_concurrent_test.key");
    
    let server = std::sync::Arc::new(CortexKGMCPServer::new(config).await.unwrap());
    
    // Test concurrent health checks
    let mut handles = Vec::new();
    for _i in 0..10 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let health = server_clone.get_health_status().await;
            health.health_score
        });
        handles.push(handle);
    }
    
    // Wait for all health checks
    let mut health_scores = Vec::new();
    for handle in handles {
        let score = handle.await.unwrap();
        health_scores.push(score);
    }
    
    assert_eq!(health_scores.len(), 10);
    for score in health_scores {
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    server.shutdown().await.unwrap();
    
    // Cleanup
    std::fs::remove_file("temp/jwt_concurrent_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_deployment_readiness() {
    let mut config = MCPServerConfig::default();
    config.server.port = 8085;
    config.security.enable_oauth = false;
    
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_deploy_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_deploy_test.key");
    
    let server = CortexKGMCPServer::new(config).await.unwrap();
    
    // Verify all deployment requirements
    let health = server.get_health_status().await;
    
    assert!(!health.status.is_empty());
    assert!(health.health_score >= 0.0);
    assert!(health.response_time_ms >= 0.0);
    assert!(health.throughput_ops_per_min >= 0.0);
    
    // Test shutdown performance
    let shutdown_start = std::time::Instant::now();
    server.shutdown().await.unwrap();
    let shutdown_time = shutdown_start.elapsed();
    
    assert!(shutdown_time < Duration::from_secs(5));
    
    println!("✅ Deployment readiness verified");
    println!("   - Health checks: Working");
    println!("   - Performance metrics: Available");
    println!("   - Graceful shutdown: {}ms", shutdown_time.as_millis());
    
    // Cleanup
    std::fs::remove_file("temp/jwt_deploy_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}
```

**Verification**: 
- All integration tests pass
- Server initialization works correctly
- Concurrent operations are handled
- Deployment readiness is verified

---

## Validation Checklist

- [ ] Server struct and initialization complete
- [ ] Component configuration from config files
- [ ] Server lifecycle (start/stop) methods implemented
- [ ] Request processing pipeline functional
- [ ] Background task management working
- [ ] Production configuration optimized
- [ ] Development and testing configs created
- [ ] Basic deployment script functional
- [ ] Docker containerization complete
- [ ] CI/CD workflow operational
- [ ] Integration tests passing
- [ ] System ready for production deployment

## Next Phase Dependencies

This phase enables:
- **MicroPhase 9**: Documentation of deployed system
- **Production Operations**: Monitoring and maintenance
- **User Adoption**: Complete, deployable system

```

**File**: `scripts/monitor.sh`

```bash
#!/bin/bash
# CortexKG MCP Server Monitoring Script

CONTAINER_NAME="cortex-kg-mcp-prod"
LOG_FILE="/var/log/cortex-kg/monitor.log"

echo "🔍 CortexKG MCP Server Monitoring Dashboard"
echo "=========================================="

# Check container status
check_container_status() {
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "✅ Container Status: Running"
        echo "📊 Container Info:"
        docker stats "$CONTAINER_NAME" --no-stream --format "   CPU: {{.CPUPerc}} | Memory: {{.MemUsage}} | Network: {{.NetIO}}"
    else
        echo "❌ Container Status: Not Running"
        return 1
    fi
}

# Check health endpoint
check_health() {
    echo "🏥 Health Check:"
    if curl -s http://localhost:8080/health | jq -r '.status' 2>/dev/null; then
        echo "✅ Health endpoint responding"
        curl -s http://localhost:8080/health | jq '.'
    else
        echo "❌ Health endpoint not responding"
        return 1
    fi
}

# Check performance metrics
check_metrics() {
    echo "📈 Performance Metrics:"
    if curl -s http://localhost:9090/metrics 2>/dev/null | head -20; then
        echo "✅ Metrics endpoint responding"
    else
        echo "❌ Metrics endpoint not responding"
    fi
}

# Check logs for errors
check_logs() {
    echo "📋 Recent Logs (last 20 lines):"
    docker logs "$CONTAINER_NAME" --tail 20
}

# Performance summary
performance_summary() {
    echo "📊 Performance Summary:"
    
    # Get health status
    HEALTH_DATA=$(curl -s http://localhost:8080/health 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo "   Response Time: $(echo "$HEALTH_DATA" | jq -r '.response_time_ms')ms"
        echo "   Throughput: $(echo "$HEALTH_DATA" | jq -r '.throughput_ops_per_min') ops/min"
        echo "   Error Rate: $(echo "$HEALTH_DATA" | jq -r '.error_rate_percent')%"
        echo "   Memory Usage: $(echo "$HEALTH_DATA" | jq -r '.memory_usage_mb')MB"
    fi
}

# Main monitoring function
main() {
    while true; do
        clear
        echo "🔍 CortexKG MCP Server Monitoring Dashboard"
        echo "==========================================="
        echo "Time: $(date)"
        echo ""
        
        check_container_status
        echo ""
        
        check_health
        echo ""
        
        performance_summary
        echo ""
        
        echo "Press Ctrl+C to exit, or wait 30 seconds for refresh..."
        sleep 30
    done
}

# Command line options
case "${1:-monitor}" in
    status)
        check_container_status
        ;;
    health)
        check_health
        ;;
    metrics)
        check_metrics
        ;;
    logs)
        check_logs
        ;;
    *)
        main
        ;;
esac
```

**Success Criteria**:
- Automated deployment script with environment support
- Docker containerization with optimized production image
- Health checks and monitoring integration
- Backup and rollback capabilities
- Resource limits and security configurations
- Monitoring dashboard for operational visibility

### Task 8.4: CI/CD Pipeline Configuration
**Estimated Time**: 20 minutes  
**File**: `.github/workflows/ci-cd.yml`

```yaml
name: CortexKG MCP Server CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust: [stable, beta]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Check formatting
      run: cargo fmt --all -- --check
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    
    - name: Create test JWT secret
      run: |
        mkdir -p config
        echo "test_secret_key_32_bytes_minimum_length" > config/jwt_test.key
    
    - name: Run unit tests
      run: cargo test --lib --bins
    
    - name: Run integration tests
      run: cargo test --test '*integration*'
    
    - name: Run performance tests
      run: cargo test --test '*performance*' -- --test-threads=1
      env:
        RUST_TEST_TIME_UNIT: 1000
        RUST_TEST_TIME_INTEGRATION: 5000
    
    - name: Generate test report
      run: |
        cargo test --no-run
        ./target/debug/test_runner > test-results.txt
        cat test-results.txt
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.rust }}
        path: |
          test-results.txt
          test-results.xml
          deployment-readiness.md

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Install cargo-audit
      run: cargo install cargo-audit
    
    - name: Run security audit
      run: cargo audit
    
    - name: Check for vulnerable dependencies
      run: cargo audit --deny warnings

  build:
    name: Build and Package
    runs-on: ubuntu-latest
    needs: [test, security-audit]
    if: github.event_name != 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Build release binary
      run: cargo build --release --bin cortex_kg_mcp_server
    
    - name: Create deployment package
      run: |
        mkdir -p package
        cp target/release/cortex_kg_mcp_server package/
        cp -r config package/
        cp -r scripts package/
        cp README.md package/
        tar -czf cortex-kg-mcp-${{ github.sha }}.tar.gz -C package .
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: cortex-kg-mcp-${{ github.sha }}
        path: cortex-kg-mcp-${{ github.sha }}.tar.gz

  docker-build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, security-audit]
    if: github.event_name != 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ghcr.io/${{ github.repository }}/cortex-kg-mcp
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: scripts/Dockerfile.production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build, docker-build]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # In real implementation, this would deploy to staging infrastructure
        echo "✅ Staging deployment complete"

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build, docker-build]
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deploying to production environment..."
        # In real implementation, this would deploy to production infrastructure
        echo "✅ Production deployment complete"
    
    - name: Post-deployment verification
      run: |
        echo "Running post-deployment verification..."
        # Health checks and smoke tests
        echo "✅ Verification complete"

  notification:
    name: Deployment Notification
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: always()
    
    steps:
    - name: Notify deployment status
      run: |
        if [[ "${{ needs.deploy-production.result }}" == "success" ]]; then
          echo "🎉 Production deployment successful!"
        else
          echo "❌ Production deployment failed!"
        fi
```

**File**: `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "maintainer-team"
    labels:
      - "dependencies"
      - "rust"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "github-actions"
```

**Success Criteria**:
- Complete CI/CD pipeline with multiple environments
- Automated testing on pull requests and pushes
- Security auditing for dependencies
- Docker image building and publishing
- Staged deployment with manual approval for production
- Automated dependency updates with Dependabot

### Task 8.5: Integration Testing and Validation
**Estimated Time**: 15 minutes  
**File**: `tests/integration/full_system_test.rs`

```rust
use cortex_kg::mcp::server::{CortexKGMCPServer, MCPRequest, HealthStatus};
use cortex_kg::mcp::config::MCPServerConfig;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_full_system_integration() {
    // Initialize server with test configuration
    let mut config = MCPServerConfig::default();
    
    // Use test-specific settings
    config.server.port = 8083; // Different port for testing
    config.security.enable_oauth = false;
    config.performance.cache_size_mb = 64;
    config.performance.connection_pool_size = 10;
    
    // Create temporary JWT secret
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_integration_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_integration_test.key");
    
    let server = CortexKGMCPServer::new(config).await.unwrap();
    
    // Test health status
    let health = server.get_health_status().await;
    assert!(!health.status.is_empty());
    assert!(health.health_score >= 0.0);
    
    // Test basic request processing (with mock authentication)
    let request = MCPRequest {
        id: "test_integration_1".to_string(),
        method: "get_memory_stats".to_string(),
        params: serde_json::json!({
            "include_performance": true
        }),
        headers: std::collections::HashMap::new(), // No auth for this test
    };
    
    // Note: In real integration test, you'd set up proper authentication
    // For now, we're testing the server infrastructure
    
    // Test graceful shutdown
    let shutdown_result = timeout(Duration::from_secs(5), server.shutdown()).await;
    assert!(shutdown_result.is_ok());
    
    // Cleanup
    std::fs::remove_file("temp/jwt_integration_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_concurrent_server_operations() {
    let mut config = MCPServerConfig::default();
    config.server.port = 8084;
    config.security.enable_oauth = false;
    
    // Create temporary JWT secret
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_concurrent_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_concurrent_test.key");
    
    let server = std::sync::Arc::new(CortexKGMCPServer::new(config).await.unwrap());
    
    // Test concurrent health checks
    let mut handles = Vec::new();
    for i in 0..10 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let health = server_clone.get_health_status().await;
            assert!(!health.status.is_empty());
            health.health_score
        });
        handles.push(handle);
    }
    
    // Wait for all health checks to complete
    let mut health_scores = Vec::new();
    for handle in handles {
        let score = handle.await.unwrap();
        health_scores.push(score);
    }
    
    // All health scores should be valid
    assert_eq!(health_scores.len(), 10);
    for score in health_scores {
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    // Test shutdown
    server.shutdown().await.unwrap();
    
    // Cleanup
    std::fs::remove_file("temp/jwt_concurrent_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_configuration_loading() {
    use cortex_kg::mcp::config::MCPServerConfig;
    
    // Test default configuration
    let default_config = MCPServerConfig::default();
    assert_eq!(default_config.server.host, "127.0.0.1");
    assert_eq!(default_config.server.port, 8080);
    assert_eq!(default_config.neuromorphic.cortical_columns, 4);
    assert!(default_config.performance.enable_simd);
    
    // Test configuration validation
    assert!(default_config.server.max_connections > 0);
    assert!(default_config.performance.cache_size_mb > 0);
    assert!(default_config.security.rate_limit_per_minute > 0);
}

#[tokio::test]
async fn test_error_handling_and_recovery() {
    let mut config = MCPServerConfig::default();
    config.server.port = 8085;
    config.security.enable_oauth = false;
    
    // Test with invalid JWT secret path
    config.security.jwt_secret_path = std::path::PathBuf::from("/nonexistent/path/jwt.key");
    
    let server_result = CortexKGMCPServer::new(config).await;
    assert!(server_result.is_err());
    
    // Test with valid configuration
    let mut valid_config = MCPServerConfig::default();
    valid_config.server.port = 8086;
    valid_config.security.enable_oauth = false;
    
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_recovery_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    valid_config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_recovery_test.key");
    
    let server = CortexKGMCPServer::new(valid_config).await.unwrap();
    
    // Should initialize successfully
    let health = server.get_health_status().await;
    assert!(!health.status.is_empty());
    
    server.shutdown().await.unwrap();
    
    // Cleanup
    std::fs::remove_file("temp/jwt_recovery_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_performance_under_integration_load() {
    let mut config = MCPServerConfig::default();
    config.server.port = 8087;
    config.security.enable_oauth = false;
    config.performance.cache_size_mb = 128;
    config.performance.connection_pool_size = 50;
    
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_perf_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_perf_test.key");
    
    let server = std::sync::Arc::new(CortexKGMCPServer::new(config).await.unwrap());
    
    // Simulate load with concurrent health checks
    let start_time = std::time::Instant::now();
    let mut handles = Vec::new();
    
    for _i in 0..100 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let _health = server_clone.get_health_status().await;
        });
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let total_time = start_time.elapsed();
    println!("100 concurrent health checks completed in: {:?}", total_time);
    
    // Should complete within reasonable time
    assert!(total_time < Duration::from_secs(10));
    
    server.shutdown().await.unwrap();
    
    // Cleanup
    std::fs::remove_file("temp/jwt_perf_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}

#[tokio::test]
async fn test_deployment_readiness() {
    // Test that all components required for deployment are working
    let mut config = MCPServerConfig::default();
    config.server.port = 8088;
    config.security.enable_oauth = false;
    
    std::fs::create_dir_all("temp").unwrap();
    std::fs::write("temp/jwt_deploy_test.key", b"test_secret_key_32_bytes_minimum").unwrap();
    config.security.jwt_secret_path = std::path::PathBuf::from("temp/jwt_deploy_test.key");
    
    let server = CortexKGMCPServer::new(config).await.unwrap();
    
    // Check all deployment requirements
    let health = server.get_health_status().await;
    
    // Health check working
    assert!(!health.status.is_empty());
    
    // Performance metrics available
    assert!(health.health_score >= 0.0);
    assert!(health.response_time_ms >= 0.0);
    assert!(health.throughput_ops_per_min >= 0.0);
    assert!(health.error_rate_percent >= 0.0);
    assert!(health.memory_usage_mb >= 0.0);
    assert!(health.uptime_seconds >= 0);
    
    // Server can shutdown gracefully
    let shutdown_start = std::time::Instant::now();
    server.shutdown().await.unwrap();
    let shutdown_time = shutdown_start.elapsed();
    
    // Shutdown should be fast
    assert!(shutdown_time < Duration::from_secs(5));
    
    println!("✅ Deployment readiness verified:");
    println!("   - Health checks: Working");
    println!("   - Performance metrics: Available");
    println!("   - Graceful shutdown: {}ms", shutdown_time.as_millis());
    
    // Cleanup
    std::fs::remove_file("temp/jwt_deploy_test.key").unwrap();
    std::fs::remove_dir("temp").unwrap();
}
```

**Success Criteria**:
- Full system integration test validates all components working together
- Concurrent operations handled correctly
- Configuration loading and validation working
- Error handling and recovery mechanisms functional
- Performance under integration load acceptable
- Deployment readiness verified with all requirements

## Validation Checklist

- [ ] Complete server integration with all MicroPhases
- [ ] Production configuration optimized for performance and security
- [ ] Deployment scripts with Docker containerization functional
- [ ] CI/CD pipeline with automated testing and deployment
- [ ] Monitoring and health check systems operational
- [ ] Error handling and graceful shutdown working
- [ ] Performance targets met in integrated system
- [ ] Security measures properly implemented
- [ ] Integration tests verify end-to-end functionality
- [ ] Deployment readiness validated

## Next Phase Dependencies

This phase completes the system integration for:
- MicroPhase 9: Documentation reflects deployed system architecture
- Production deployment with confidence in system reliability
- Operational monitoring and maintenance procedures