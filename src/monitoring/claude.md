# Directory Overview: LLMKG Monitoring System

## 1. High-Level Summary

The `src/monitoring` directory contains a comprehensive, production-ready monitoring and observability system for the LLMKG (Large Language Model Knowledge Graph) project. This system provides real-time performance monitoring, metrics collection, alerting, dashboard visualization, and various data export capabilities. It's designed to monitor both system-level metrics (CPU, memory, disk) and application-specific metrics (knowledge graph operations, brain-inspired computations, API endpoints, test executions).

## 2. Tech Stack

- **Languages:** Rust (primary), HTML/JavaScript/CSS (dashboard), JSON
- **Frameworks:** 
  - Tokio (async runtime)
  - Warp (HTTP server for dashboard)
  - Serde (serialization)
  - Tokio-tungstenite (WebSocket support)
- **Libraries:** 
  - `sysinfo` (system metrics)
  - `reqwest` (HTTP client for exporters)
  - `chrono` (time handling)
  - `uuid` (unique identifiers)
  - `notify` (file watching)
  - `walkdir` (directory traversal)
  - `syn` (Rust AST parsing)
- **External Integrations:**
  - Prometheus (metrics export)
  - InfluxDB (time-series data)
  - JSON export for custom analysis
- **Real-time Communication:** WebSockets for live dashboard updates

## 3. Directory Structure

```
monitoring/
├── mod.rs                     # Module declarations and public exports
├── alerts.rs                  # Alert management system
├── brain_metrics_collector.rs # Brain-specific metrics collection
├── collectors.rs              # Base collector traits and system metrics  
├── collectors/                # Specialized metric collectors
│   ├── api_endpoint_monitor.rs    # API endpoint monitoring
│   ├── codebase_analyzer.rs       # Static code analysis
│   ├── knowledge_engine_metrics.rs # Knowledge graph metrics
│   ├── runtime_profiler.rs        # Function execution profiling
│   └── test_execution_tracker.rs  # Test suite monitoring
├── dashboard.rs               # Web dashboard with 3D visualization
├── exporters.rs              # Metrics export to external systems
├── memory_info.rs            # Memory information structures
├── observability.rs          # Distributed tracing and logging
└── performance.rs            # Performance monitoring core
```

## 4. File Breakdown

### `mod.rs`
- **Purpose:** Module organization and public API definition
- **Key Exports:** 
  - Performance monitoring: `PerformanceMonitor`, `PerformanceMetrics`
  - Observability: `ObservabilityEngine`, `TraceExporter`, `MetricsCollector`
  - Alerts: `AlertManager`, `AlertRule`, `AlertCondition`
  - Dashboard: `PerformanceDashboard`, `DashboardServer`
  - Metrics: `MetricRegistry`, `Counter`, `Gauge`, `Histogram`, `Timer`

### `metrics.rs`
- **Purpose:** Core metrics collection and aggregation system
- **Classes:**
  - `MetricRegistry`: Central registry for all metrics with thread-safe operations
  - `Counter`: Monotonically increasing values (e.g., request counts)
  - `Gauge`: Instantaneous values (e.g., CPU usage)
  - `Histogram`: Distribution of values with configurable buckets
  - `Timer`: Execution time tracking with percentile calculations
- **Key Methods:**
  - `MetricRegistry::counter()`: Get or create counter metric
  - `MetricRegistry::gauge()`: Get or create gauge metric
  - `Timer::time()`: Time a function execution
  - `MetricRegistry::collect_all_samples()`: Export all current metrics

### `alerts.rs`
- **Purpose:** Comprehensive alerting system with multiple notification channels
- **Classes:**
  - `AlertManager`: Core alert processing and notification
  - `AlertRule`: Configurable alert conditions and thresholds
  - `AlertCondition`: Specific metric conditions (GreaterThan, LessThan, etc.)
- **Notification Channels:**
  - `EmailNotificationChannel`: SMTP-based email alerts
  - `SlackNotificationChannel`: Slack webhook integration
  - `ConsoleNotificationChannel`: Console output for testing
- **Key Methods:**
  - `AlertManager::trigger_alert()`: Manually trigger an alert
  - `AlertManager::evaluate_conditions()`: Check metrics against rules
  - `AlertManager::acknowledge_alert()`: Mark alert as acknowledged

### `brain_metrics_collector.rs`
- **Purpose:** Specialized collector for brain-enhanced knowledge graph metrics
- **Key Metrics Collected:**
  - Entity and relationship counts
  - Graph density and clustering coefficients
  - Activation levels and learning efficiency
  - Concept coherence measures
- **Integration:** Connects to `BrainEnhancedKnowledgeGraph` for real-time brain metrics

### `collectors.rs`
- **Purpose:** Base collector framework and system-level metrics
- **Classes:**
  - `SystemMetricsCollector`: CPU, memory, disk, network, load metrics
  - `ApplicationMetricsCollector`: Process-specific metrics
  - `CustomMetricsCollector`: User-defined metric collection
- **Configuration:** `MetricsCollectionConfig` with collection intervals and filters
- **Platform Support:** Linux-specific implementations with fallbacks

### `collectors/api_endpoint_monitor.rs`
- **Purpose:** Real-time API endpoint monitoring and performance analysis
- **Key Features:**
  - Automatic endpoint discovery from source code
  - Request/response tracking with detailed timing
  - Error pattern analysis and performance bottleneck detection
  - Real-time API testing capabilities
- **Classes:**
  - `ApiEndpointMonitor`: Main monitoring coordinator
  - `ApiEndpoint`: Endpoint definition with parameters and documentation
  - `ApiMetrics`: Comprehensive endpoint statistics
- **Real Endpoints Discovered:**
  - `/api/metrics`: System metrics from MetricRegistry
  - `/api/history`: Historical metrics data
  - `/`: Dashboard HTML interface

### `collectors/codebase_analyzer.rs`
- **Purpose:** Static code analysis and dependency tracking
- **Key Features:**
  - Rust and TypeScript file analysis using AST parsing
  - Dependency graph construction and visualization
  - Complexity metrics calculation
  - Real-time file watching for changes
- **Classes:**
  - `CodebaseAnalyzer`: Main analysis engine with depth limiting for safety
  - `FileStructure`: Hierarchical representation of codebase
  - `DependencyGraph`: Module relationships and import analysis
- **Safety Features:** Skip problematic directories (.git, target, node_modules) to prevent recursion

### `collectors/knowledge_engine_metrics.rs`
- **Purpose:** Knowledge engine specific metrics extraction
- **Integration:** Direct connection to `KnowledgeEngine` for real-time data
- **Metrics:**
  - Entity counts and types distribution
  - Memory usage statistics
  - Graph density calculations
  - Simulated activation and learning metrics

### `collectors/runtime_profiler.rs`
- **Purpose:** Function-level execution profiling and performance analysis
- **Key Features:**
  - Function call tracing with parameter capture
  - Memory allocation tracking
  - Performance bottleneck detection
  - Hot path analysis
- **Classes:**
  - `RuntimeProfiler`: Main profiling engine
  - `ExecutionTrace`: Detailed function execution record
  - `PerformanceBottleneck`: Automated bottleneck detection
- **Macro Support:** `trace_function!` macro for easy instrumentation

### `collectors/test_execution_tracker.rs`
- **Purpose:** Comprehensive test suite monitoring and execution tracking
- **Key Features:**
  - Automatic test discovery (Rust and TypeScript)
  - Test execution with real-time progress tracking
  - Coverage analysis integration
  - Test health scoring and recommendations
- **Classes:**
  - `TestExecutionTracker`: Main test coordination
  - `TestSuite`: Test suite definition with configuration
  - `TestExecution`: Detailed execution tracking
- **Framework Support:** Rust tests, Jest, Mocha with extensible architecture

### `dashboard.rs`
- **Purpose:** Real-time web dashboard with 3D knowledge graph visualization
- **Key Features:**
  - WebSocket-based real-time updates
  - 3D knowledge graph visualization using Three.js
  - Interactive API endpoint monitoring
  - Test execution streaming
  - Codebase structure visualization
- **Classes:**
  - `PerformanceDashboard`: Main dashboard server
  - `DashboardServer`: HTTP/WebSocket server wrapper
  - `RealTimeMetrics`: Comprehensive metrics snapshot
- **Endpoints:**
  - `GET /`: Dashboard HTML interface
  - `GET /api/metrics`: Real-time metrics JSON
  - `GET /api/history`: Historical data
  - `GET /api/endpoints`: API monitoring data
  - WebSocket: Real-time metric streaming

### `exporters.rs`
- **Purpose:** Export metrics to external monitoring systems
- **Exporters:**
  - `PrometheusExporter`: Export to Prometheus format
  - `InfluxDBExporter`: Time-series data to InfluxDB
  - `JsonExporter`: JSON format with file rotation
  - `MultiExporter`: Parallel export to multiple systems
- **Features:**
  - Configurable export intervals and batching
  - Retry logic and error handling
  - File rotation for JSON exports

### `observability.rs`
- **Purpose:** Distributed tracing and centralized logging
- **Classes:**
  - `ObservabilityEngine`: Unified telemetry collection
  - `TraceExporter`: Distributed tracing with span management
  - `LogAggregator`: Centralized log collection with filtering
- **Integration:** Seamless integration with performance monitoring

### `performance.rs`
- **Purpose:** Core performance monitoring and health assessment
- **Classes:**
  - `PerformanceMonitor`: Main monitoring coordinator
  - `Operation`: Trackable operation definition
  - `PerformanceMetrics`: Comprehensive performance snapshot
- **Features:**
  - Operation lifecycle tracking
  - Resource usage monitoring
  - System health assessment
  - Performance trend analysis

### `memory_info.rs`
- **Purpose:** Memory information structures and utilities
- **Structure:** `MemoryInfo` with total, used, available memory and swap information
- **Methods:** Factory methods for creating memory snapshots

## 5. Key Data Structures and Metrics

### Core Metric Types
- **Counter**: Monotonically increasing values (requests, errors)
- **Gauge**: Point-in-time values (CPU%, memory usage)
- **Histogram**: Value distributions with configurable buckets
- **Timer**: Duration measurements with percentile calculations

### Brain-Specific Metrics
- **Entity Management**: Total entities, relationships, types
- **Graph Analysis**: Density, clustering coefficient, connectivity
- **Cognitive Metrics**: Activation levels, learning efficiency, concept coherence
- **Memory Statistics**: Node count, memory usage, index efficiency

### System Metrics
- **CPU**: Usage percentage, per-core metrics, load averages
- **Memory**: Total/used/available, swap usage, process memory
- **Disk**: Read/write operations, throughput, utilization
- **Network**: Bytes transferred, packet counts, error rates

### API Metrics
- **Endpoint Discovery**: Automatic detection from source code
- **Performance**: Request/response times, throughput, error rates
- **Error Analysis**: Pattern detection, bottleneck identification
- **Real-time Testing**: Automated endpoint validation

## 6. API Endpoints

### Dashboard HTTP Endpoints
- **`GET /`**: Main dashboard HTML interface with 3D visualization
- **`GET /api/metrics`**: Current metrics in JSON format
- **`GET /api/history`**: Historical metrics data
- **`GET /api/endpoints`**: API endpoint monitoring data
- **`GET /api/tests/discover`**: Available test suites
- **`POST /api/tests/execute`**: Execute test suite
- **`GET /api/tests/status/{id}`**: Test execution status

### WebSocket Endpoints
- **`ws://localhost:8081`**: Real-time metrics streaming
- **Message Types:**
  - `MetricsUpdate`: Live metric updates
  - `TestStarted/Progress/Completed`: Test execution events
  - `AlertUpdate`: Alert notifications

## 7. Key Variables and Logic

### Configuration Variables
- **Collection Intervals**: Default 15-second metrics collection
- **History Retention**: 24-hour performance history
- **WebSocket Limits**: 1000-message broadcast channels
- **Dashboard Ports**: HTTP (8090), WebSocket (8081)

### Performance Thresholds
- **Memory Alert**: > 1GB usage triggers alert
- **CPU Alert**: > 80% usage triggers alert
- **Latency Alert**: > 30 seconds operation time
- **Error Rate**: Tracked per endpoint with pattern analysis

### Brain Metrics Calculations
```rust
// Graph density calculation
let graph_density = if entity_count > 1 {
    let max_possible_edges = entity_count * (entity_count - 1);
    memory_stats.total_triples as f64 / max_possible_edges as f64
} else {
    0.0
};

// Learning efficiency based on memory usage
let learning_efficiency = if memory_stats.total_nodes > 0 {
    1.0 - (memory_stats.total_bytes as f64 / (memory_stats.total_nodes as f64 * 1024.0)).min(1.0)
} else {
    0.0
};
```

## 8. Dependencies

### Internal Dependencies
- `crate::core::knowledge_engine`: Knowledge graph engine
- `crate::core::brain_enhanced_graph`: Brain-enhanced graph operations
- `crate::error`: Common error handling
- `crate::monitoring::*`: Cross-module metric sharing

### External Dependencies
- **Async Runtime**: `tokio` for async operations
- **HTTP Server**: `warp` for dashboard web server
- **WebSockets**: `tokio-tungstenite` for real-time communication
- **Serialization**: `serde` with JSON/TOML support
- **System Info**: `sysinfo` for system metrics
- **File Watching**: `notify` for real-time file changes
- **Code Analysis**: `syn` for Rust AST parsing
- **Web Client**: `reqwest` for external API calls

### Optional Dependencies
- **Prometheus**: For metrics export
- **InfluxDB**: For time-series storage
- **Email/Slack**: For alert notifications

## 9. Real-time Features

### WebSocket Streaming
- Live metrics updates every second
- Test execution progress streaming
- Alert notifications
- API request monitoring

### 3D Visualization
- Interactive knowledge graph using Three.js
- Entity relationship visualization
- Real-time graph updates
- Search and navigation capabilities

### Auto-discovery
- API endpoints from source code analysis
- Test suites from file system scanning
- Dependency relationships from import analysis
- Performance bottlenecks from execution patterns

## 10. Production Readiness

### Error Handling
- Comprehensive error propagation
- Graceful degradation for missing dependencies
- Resource cleanup and memory management
- Safe directory traversal with depth limits

### Performance Optimizations
- Efficient metric aggregation
- Batched export operations
- Configurable collection intervals
- Memory-bounded history retention

### Security Considerations
- Input validation for API endpoints
- Safe file system operations
- Resource usage limits
- Secure WebSocket connections

This monitoring system represents a sophisticated, production-ready observability solution specifically designed for knowledge graph and AI applications, providing both traditional system monitoring and specialized brain-inspired metrics collection.