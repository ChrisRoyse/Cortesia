# Directory Overview: Production System Components

## 1. High-Level Summary

The `src/production` directory contains comprehensive production-ready system components for the LLMKG (Large Language Model Knowledge Graph) system. This module provides enterprise-grade infrastructure components including error recovery, monitoring, health checks, graceful shutdown, and rate limiting. These components work together to create a robust, observable, and maintainable production system suitable for high-availability deployments.

The production system integrates all components through the `ProductionSystem` struct which orchestrates error recovery, monitoring, resource management, health monitoring, and graceful shutdown in a coordinated manner.

## 2. Tech Stack

- **Language:** Rust
- **Async Runtime:** Tokio
- **Serialization:** Serde (JSON)
- **Concurrency:** Arc, RwLock, AtomicU64, AtomicU32, AtomicBool
- **Collections:** DashMap for concurrent HashMaps
- **Time Management:** std::time (Duration, Instant, SystemTime)
- **Signal Handling:** tokio::signal (Unix/Windows)
- **Memory Management:** Arc, RwLock for shared state
- **Error Handling:** Custom GraphError types
- **Observability:** Structured logging, metrics collection, Prometheus format export

## 3. Directory Structure

The production directory contains 6 main files:
- `mod.rs` - Main integration module and public API
- `error_recovery.rs` - Circuit breakers and retry mechanisms
- `graceful_shutdown.rs` - System shutdown coordination
- `health_checks.rs` - Health monitoring and diagnostics
- `monitoring.rs` - Logging, metrics, and alerting
- `rate_limiting.rs` - Rate limiting and resource management

## 4. File Breakdown

### `mod.rs`

**Purpose:** Central integration module that coordinates all production components and provides the main `ProductionSystem` API.

**Key Structures:**
- `ProductionConfig`
  - **Description:** Complete production system configuration combining all component configs.
  - **Fields:**
    - `monitoring: MonitoringConfig` - Monitoring system configuration
    - `rate_limiting: RateLimitConfig` - Rate limiting configuration  
    - `resource_limits: ResourceLimits` - Resource usage limits
    - `health_checks: HealthCheckConfig` - Health check configuration
    - `shutdown: ShutdownConfig` - Graceful shutdown configuration
    - `error_recovery_enabled: bool` - Whether to enable error recovery

- `ProductionSystem`
  - **Description:** Main orchestrator that integrates all production components.
  - **Fields:**
    - `knowledge_engine: Arc<RwLock<KnowledgeEngine>>` - Core knowledge engine
    - `error_recovery: ErrorRecoveryManager` - Error recovery system
    - `monitor: Arc<ProductionMonitor>` - Monitoring system
    - `rate_limiter: RateLimitingManager` - Rate limiting system
    - `health_checker: HealthCheckSystem` - Health monitoring system
    - `shutdown_manager: Arc<GracefulShutdownManager>` - Shutdown coordinator
    - `config: ProductionConfig` - System configuration

**Key Functions:**
- `new(knowledge_engine, config) -> ProductionSystem` - Creates new production system with all components
- `execute_protected_operation<T, F, Fut>(operation_name, user_id, operation) -> Result<T>` - Executes operations with full production protection (rate limiting, error recovery, monitoring)
- `get_system_status() -> HashMap<String, Value>` - Returns comprehensive system status from all components
- `get_health_report() -> SystemHealthReport` - Gets detailed health report
- `get_prometheus_metrics() -> String` - Exports metrics in Prometheus format
- `shutdown() -> Result<ShutdownReport>` - Initiates graceful system shutdown
- `configure_operation(operation_name, config)` - Configures operation-specific settings

**Helper Functions:**
- `create_production_system(knowledge_engine) -> ProductionSystem` - Creates system with default configuration
- `create_production_system_with_config(knowledge_engine, config) -> ProductionSystem` - Creates system with custom configuration

### `error_recovery.rs`

**Purpose:** Implements comprehensive error recovery with circuit breakers, retry logic, and graceful degradation.

**Key Structures:**
- `RetryConfig`
  - **Description:** Configuration for retry behavior with exponential backoff.
  - **Fields:**
    - `max_retries: u32` - Maximum retry attempts
    - `base_delay_ms: u64` - Initial delay between retries
    - `max_delay_ms: u64` - Maximum delay cap
    - `backoff_multiplier: f64` - Exponential backoff multiplier
    - `jitter_factor: f64` - Random jitter to prevent thundering herd

- `CircuitBreaker`
  - **Description:** Circuit breaker implementation for fault tolerance.
  - **Fields:**
    - `state: RwLock<CircuitState>` - Current circuit state (Closed/Open/HalfOpen)
    - `failure_count: AtomicU32` - Count of consecutive failures
    - `success_count: AtomicU32` - Count of consecutive successes
    - `last_failure_time: AtomicU64` - Timestamp of last failure
    - `config: CircuitBreakerConfig` - Circuit breaker configuration

- `ErrorRecoveryManager`
  - **Description:** Central manager for all error recovery mechanisms.
  - **Fields:**
    - `circuit_breakers: DashMap<String, Arc<CircuitBreaker>>` - Per-operation circuit breakers
    - `retry_configs: DashMap<String, RetryConfig>` - Per-operation retry configurations
    - `operation_stats: DashMap<String, OperationStats>` - Performance statistics per operation

**Key Functions:**
- `execute_with_recovery<T, F, Fut>(operation_name, operation) -> Result<T>` - Executes operation with retry and circuit breaker protection
- `configure_retry(operation, config)` - Configures retry behavior for specific operation
- `configure_circuit_breaker(operation, config)` - Configures circuit breaker for specific operation
- `get_operation_stats(operation_name) -> Option<HashMap<String, u64>>` - Gets performance statistics for operation
- `get_circuit_breaker_status(operation_name) -> Option<CircuitState>` - Gets circuit breaker state
- `health_check() -> HashMap<String, Value>` - Health check for error recovery system

### `graceful_shutdown.rs`

**Purpose:** Provides comprehensive graceful shutdown with data integrity preservation and coordinated resource cleanup.

**Key Structures:**
- `ShutdownPhase` (Enum)
  - **Description:** Represents different phases of the shutdown process.
  - **Variants:** Running, InitiateShutdown, StopAcceptingRequests, FinishActiveRequests, SaveState, CleanupResources, Terminated

- `GracefulShutdownManager`
  - **Description:** Coordinates graceful shutdown across all system components.
  - **Fields:**
    - `config: ShutdownConfig` - Shutdown configuration
    - `progress: Arc<ShutdownProgress>` - Shutdown progress tracking
    - `shutdown_handlers: DashMap<String, Arc<dyn ShutdownHandler>>` - Component shutdown handlers
    - `shutdown_checkpoints: Arc<Mutex<Vec<ShutdownCheckpoint>>>` - State checkpoints during shutdown
    - `is_shutdown_initiated: Arc<AtomicBool>` - Shutdown initiation flag

- `ActiveRequestGuard`
  - **Description:** RAII guard for tracking active requests during shutdown.
  - **Fields:**
    - `progress: Arc<ShutdownProgress>` - Reference to shutdown progress tracker

**Key Functions:**
- `initiate_shutdown() -> Result<ShutdownReport>` - Starts graceful shutdown process through all phases
- `force_shutdown() -> Result<ShutdownReport>` - Forces immediate shutdown when graceful fails
- `track_active_request() -> Result<ActiveRequestGuard>` - Tracks active request to prevent shutdown during processing
- `is_shutting_down() -> bool` - Checks if shutdown has been initiated
- `get_shutdown_progress() -> HashMap<String, Value>` - Gets current shutdown progress
- `setup_signal_handlers()` - Sets up OS signal handlers for shutdown triggers

### `health_checks.rs`

**Purpose:** Provides comprehensive health monitoring, status reporting, and system diagnostics with automated component tracking.

**Key Structures:**
- `HealthStatus` (Enum)
  - **Description:** Health status levels for components.
  - **Variants:** Healthy, Warning, Critical, Unknown

- `ComponentHealth`
  - **Description:** Tracks health metrics for individual system components.
  - **Fields:**
    - `name: String` - Component name
    - `status: RwLock<HealthStatus>` - Current health status
    - `consecutive_failures: AtomicU32` - Count of consecutive failures
    - `total_checks: AtomicU64` - Total health checks performed
    - `avg_response_time_ms: AtomicU64` - Average response time
    - `config: HealthCheckConfig` - Health check configuration
    - `is_essential: AtomicBool` - Whether component is essential for system operation

- `HealthCheckSystem`
  - **Description:** Comprehensive health monitoring system.
  - **Fields:**
    - `components: DashMap<String, Arc<ComponentHealth>>` - All monitored components
    - `knowledge_engine: Arc<RwLock<KnowledgeEngine>>` - Reference to knowledge engine
    - `check_runners: DashMap<String, JoinHandle<()>>` - Background health check tasks
    - `is_shutting_down: Arc<AtomicBool>` - Shutdown flag

**Key Functions:**
- `register_component(name, config, is_essential)` - Registers component for health monitoring
- `perform_full_health_check() -> SystemHealthReport` - Performs comprehensive system health check
- `check_component_health(component_name) -> HealthCheckResult` - Checks health of specific component
- `get_health_status() -> HashMap<String, Value>` - Gets current health status for all components
- `shutdown()` - Gracefully shuts down health monitoring system

**Component-Specific Health Checks:**
- `check_knowledge_engine_health()` - Verifies knowledge engine accessibility and basic stats
- `check_memory_health()` - Monitors system memory usage and thresholds
- `check_storage_health()` - Checks storage space and accessibility
- `check_database_health()` - Verifies database connectivity and performance
- `check_cache_health()` - Monitors cache hit rates and performance

### `monitoring.rs`

**Purpose:** Provides comprehensive monitoring, logging, and observability with structured logging, metrics collection, and alerting.

**Key Structures:**
- `LogLevel` (Enum)
  - **Description:** Log severity levels.
  - **Variants:** Trace(0), Debug(1), Info(2), Warn(3), Error(4), Critical(5)

- `LogEntry`
  - **Description:** Structured log entry with metadata.
  - **Fields:**
    - `timestamp: u64` - Log timestamp
    - `level: LogLevel` - Log severity level
    - `component: String` - Component that generated the log
    - `operation: String` - Operation being performed
    - `message: String` - Log message
    - `metadata: HashMap<String, Value>` - Additional structured data
    - `correlation_id: Option<String>` - Request correlation ID
    - `execution_time_ms: Option<u64>` - Operation execution time

- `ProductionMonitor`
  - **Description:** Central monitoring system with logging, metrics, and alerting.
  - **Fields:**
    - `log_buffer: Arc<RwLock<Vec<LogEntry>>>` - In-memory log buffer
    - `metrics: DashMap<String, Vec<MetricPoint>>` - Time-series metrics storage
    - `counters: DashMap<String, Arc<AtomicU64>>` - Counter metrics
    - `gauges: DashMap<String, Arc<AtomicU64>>` - Gauge metrics
    - `alert_rules: DashMap<String, AlertRule>` - Alerting rules
    - `active_alerts: DashMap<String, Alert>` - Currently active alerts
    - `system_stats: Arc<RwLock<SystemStats>>` - System performance statistics

**Key Functions:**
- `log(level, component, operation, message)` - Logs structured message
- `record_metric(name, value, type, tags)` - Records metric value
- `increment_counter(name)` - Increments counter metric
- `set_gauge(name, value)` - Sets gauge metric value
- `start_timer(operation) -> TimerHandle` - Starts timing operation (RAII)
- `get_metrics() -> HashMap<String, Vec<MetricPoint>>` - Gets all recorded metrics
- `get_system_stats() -> SystemStats` - Gets system performance statistics
- `export_prometheus_metrics() -> String` - Exports metrics in Prometheus format
- `health_check() -> HashMap<String, Value>` - Health check endpoint for monitoring system

**Background Tasks:**
- Alert checking - Evaluates alert rules against metrics
- Metrics cleanup - Removes old metrics based on retention policy
- System stats updating - Collects system performance data

### `rate_limiting.rs`

**Purpose:** Provides comprehensive rate limiting and resource management to prevent system overload and ensure stability.

**Key Structures:**
- `TokenBucket`
  - **Description:** Token bucket implementation for rate limiting.
  - **Fields:**
    - `tokens: AtomicU32` - Current available tokens
    - `capacity: u32` - Maximum token capacity
    - `refill_rate: u32` - Tokens per second refill rate
    - `last_refill: AtomicU64` - Last refill timestamp

- `ResourceUsage`
  - **Description:** Tracks current system resource usage.
  - **Fields:**
    - `memory_bytes: AtomicU64` - Current memory usage
    - `cpu_percent: AtomicU32` - Current CPU usage percentage
    - `active_operations: AtomicU32` - Number of active operations
    - `database_connections: AtomicU32` - Active database connections
    - `peak_memory_bytes: AtomicU64` - Peak memory usage seen
    - `peak_operations: AtomicU32` - Peak concurrent operations

- `RateLimitingManager`
  - **Description:** Central manager for all rate limiting and resource management.
  - **Fields:**
    - `global_rate_limiter: TokenBucket` - Global system rate limiter
    - `operation_rate_limiters: DashMap<String, Arc<dyn RateLimiterTrait>>` - Per-operation rate limiters
    - `user_rate_limiters: DashMap<String, Arc<dyn RateLimiterTrait>>` - Per-user rate limiters
    - `resource_usage: Arc<ResourceUsage>` - Current resource usage tracker
    - `operation_semaphore: Arc<Semaphore>` - Limits concurrent operations
    - `connection_pool: Arc<Semaphore>` - Database connection pool

**Key Functions:**
- `check_rate_limit(operation, user_id) -> Result<()>` - Checks if request is within rate limits
- `check_resource_limits(estimated_memory) -> Result<()>` - Verifies resource limits before operation
- `acquire_operation_permit() -> Result<OperationPermit>` - Acquires permit for operation (blocking)
- `try_acquire_operation_permit() -> Result<OperationPermit>` - Tries to acquire permit (non-blocking)
- `acquire_db_connection() -> Result<DatabaseConnection>` - Acquires database connection from pool
- `configure_operation_rate_limit(operation, config)` - Configures rate limiting for specific operation
- `configure_user_rate_limit(user_id, config)` - Configures rate limiting for specific user
- `get_resource_usage() -> HashMap<String, Value>` - Gets current resource usage statistics
- `get_rate_limit_stats() -> HashMap<String, HashMap<String, u64>>` - Gets rate limiting statistics

**RAII Guards:**
- `OperationPermit` - Automatically releases operation permit when dropped
- `DatabaseConnection` - Automatically returns connection to pool when dropped

## 5. Key Variables and Logic

### Core Integration Logic
- **Production System Orchestration:** The `ProductionSystem::execute_protected_operation` method provides the main integration point, orchestrating all protection mechanisms:
  1. Shutdown check - Prevents new operations during shutdown
  2. Active request tracking - Ensures graceful shutdown waits for completion
  3. Rate limit checking - Enforces operation and user-specific limits
  4. Resource limit validation - Prevents resource exhaustion
  5. Operation permit acquisition - Controls concurrency
  6. Error recovery execution - Handles failures with retries and circuit breakers
  7. Monitoring and logging - Records metrics and logs throughout

### Circuit Breaker State Machine
- **States:** Closed (normal) → Open (failing) → HalfOpen (testing) → Closed (recovered)
- **Transition Logic:** 
  - Closed to Open: After consecutive failures exceed threshold
  - Open to HalfOpen: After recovery timeout elapses
  - HalfOpen to Closed: After consecutive successes exceed threshold
  - HalfOpen to Open: On any failure during testing

### Graceful Shutdown Phases
1. **InitiateShutdown:** Signal shutdown start
2. **StopAcceptingRequests:** Reject new requests
3. **FinishActiveRequests:** Wait for active requests to complete
4. **SaveState:** Create state checkpoints for all components
5. **CleanupResources:** Release resources and cleanup
6. **Terminated:** Shutdown complete

### Rate Limiting Algorithms
- **Token Bucket:** Allows bursts up to capacity, refills at steady rate
- **Sliding Window:** Tracks requests in rolling time window
- **Fixed Window:** Resets counter at fixed intervals
- **Leaky Bucket:** Processes requests at steady rate regardless of input

## 6. Dependencies

### Internal Dependencies
- `crate::core::knowledge_engine::KnowledgeEngine` - Core knowledge graph engine
- `crate::error::{GraphError, Result}` - Custom error types and result handling

### External Dependencies
- **tokio** - Async runtime, synchronization primitives (RwLock, Mutex, Semaphore), signal handling
- **serde** - Serialization/deserialization for configuration and data structures
- **serde_json** - JSON serialization, Value type for dynamic data
- **dashmap** - Concurrent HashMap implementation for thread-safe collections
- **rand** - Random number generation for jitter in retry logic
- **chrono** - Date/time handling for timestamps (in some functions)
- **async_trait** - Trait definitions for async methods

### Standard Library Dependencies
- **std::collections::HashMap** - Data structure for key-value storage
- **std::sync::{Arc, atomic::*}** - Memory management and atomic operations
- **std::time::{Duration, Instant, SystemTime}** - Time handling and measurements
- **std::future::Future** - Async trait definitions

## 7. System Architecture

### Component Interaction
The production system components are designed to work together:

1. **ProductionSystem** acts as the orchestrator, routing requests through all protection layers
2. **ErrorRecoveryManager** wraps operations with retry logic and circuit breakers
3. **RateLimitingManager** enforces limits before operations execute
4. **ProductionMonitor** logs and tracks metrics throughout operation lifecycle
5. **HealthCheckSystem** continuously monitors all components
6. **GracefulShutdownManager** coordinates orderly system shutdown

### Data Flow
```
Incoming Request
    ↓
ProductionSystem.execute_protected_operation()
    ↓
Shutdown Check → Rate Limit Check → Resource Check
    ↓
Acquire Operation Permit → Start Monitoring Timer
    ↓
ErrorRecoveryManager.execute_with_recovery()
    ↓
Circuit Breaker Check → Execute with Retries
    ↓
Record Metrics → Log Results → Return Response
```

### Configuration Management
- Centralized configuration through `ProductionConfig`
- Per-component configuration objects (MonitoringConfig, RateLimitConfig, etc.)
- Runtime configuration updates supported for most components
- Default configurations provided for quick setup

### Error Handling Strategy
- Graceful degradation through circuit breakers
- Automatic retries with exponential backoff and jitter
- Resource exhaustion protection through limits and permits
- Comprehensive error logging and metrics collection
- Health status reporting for proactive monitoring

This production system provides enterprise-grade reliability, observability, and operational features necessary for running the LLMKG system in production environments.