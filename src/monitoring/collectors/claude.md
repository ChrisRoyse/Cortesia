# Directory Overview: LLMKG Monitoring Collectors

## 1. High-Level Summary

The `src/monitoring/collectors` directory contains the core monitoring and metrics collection components for the LLMKG (Large Language Model Knowledge Graph) system. This directory implements comprehensive real-time monitoring capabilities including API endpoint tracking, codebase analysis, knowledge engine metrics, runtime profiling, and test execution monitoring. These collectors provide observability into system performance, code quality, and operational health.

## 2. Tech Stack

*   **Languages:** Rust
*   **Key Libraries:** 
    - `serde` - Serialization/deserialization
    - `tokio` - Async runtime and concurrency
    - `syn` - Rust AST parsing
    - `walkdir` - Directory traversal
    - `reqwest` - HTTP client
    - `notify` - File system watching
    - `uuid` - Unique identifier generation
    - `parking_lot` - High-performance synchronization primitives
*   **Dependencies:** 
    - Internal: `crate::monitoring::metrics::MetricRegistry`
    - External crates for specific functionality

## 3. Directory Structure

The directory contains 5 main collector modules, each focused on a specific aspect of system monitoring:

*   **api_endpoint_monitor.rs** - HTTP API monitoring and analytics
*   **codebase_analyzer.rs** - Static code analysis and structure monitoring  
*   **knowledge_engine_metrics.rs** - Brain/knowledge graph specific metrics
*   **runtime_profiler.rs** - Function execution and performance profiling
*   **test_execution_tracker.rs** - Test suite monitoring and execution tracking

## 4. File Breakdown

### `api_endpoint_monitor.rs`

*   **Purpose:** Real-time monitoring of HTTP API endpoints with comprehensive analytics and performance tracking.

*   **Key Structures:**
    *   `ApiEndpoint` - Defines API endpoint metadata (path, method, parameters, authentication, rate limits)
    *   `ApiRequest` - Tracks individual HTTP requests with full context
    *   `ApiResponse` - Captures HTTP response data and metrics
    *   `ApiMetrics` - Aggregates all API monitoring data
    *   `EndpointStats` - Per-endpoint performance statistics
    *   `ErrorAnalysis` - Error pattern detection and analysis

*   **Core Class: `ApiEndpointMonitor`**
    *   **Description:** Main monitoring orchestrator for API endpoints
    *   **Key Methods:**
        *   `discover_endpoints()` - Auto-discovers API endpoints from source code (dashboard routes, MCP endpoints)
        *   `register_endpoint(endpoint: ApiEndpoint)` - Manually registers an endpoint for monitoring
        *   `start_request()` - Begins tracking an HTTP request with full context
        *   `end_request()` - Completes request tracking and updates metrics
        *   `test_endpoint(test_request: ApiTestRequest)` - Executes endpoint health checks
        *   `get_metrics()` - Returns comprehensive API performance data

*   **Functions:**
    *   `discover_dashboard_endpoints()` - Discovers Warp server routes from dashboard.rs
    *   `discover_mcp_endpoints()` - Identifies MCP (Model Context Protocol) endpoints
    *   `analyze_error_patterns()` - Detects recurring error patterns and suggests fixes
    *   `update_performance_metrics()` - Calculates aggregated performance statistics

### `codebase_analyzer.rs`

*   **Purpose:** Static analysis of the codebase structure, dependencies, and complexity metrics.

*   **Key Structures:**
    *   `CodebaseMetrics` - Complete codebase analysis results
    *   `FileStructure` - Hierarchical representation of project structure
    *   `FunctionInfo` - Detailed function metadata and complexity
    *   `DependencyGraph` - Module relationships and import/export mapping
    *   `ComplexityMetrics` - Cyclomatic and cognitive complexity measurements

*   **Core Class: `CodebaseAnalyzer`**
    *   **Description:** Analyzes project structure and generates code quality metrics
    *   **Key Methods:**
        *   `analyze_codebase()` - Performs full codebase analysis including file structure, dependencies, and complexity
        *   `analyze_directory_with_depth()` - Recursively analyzes directory structure with depth limiting
        *   `analyze_rust_files()` - Parses Rust files using syn AST to extract functions, structs, enums
        *   `build_dependency_graph()` - Constructs module dependency relationships
        *   `start_watching()` - Enables real-time file system monitoring for changes

*   **Functions:**
    *   `should_skip_path()` - Filters out irrelevant directories (target, .git, node_modules)
    *   `parse_crate_import()` - Extracts internal module dependencies from use statements
    *   `parse_external_import()` - Identifies external crate dependencies
    *   `path_to_module_name()` - Converts file paths to Rust module names

*   **Helper Class: `RustAnalysisVisitor`**
    *   **Description:** AST visitor for extracting Rust code elements
    *   **Methods:**
        *   `visit_item_fn()` - Extracts function declarations
        *   `visit_item_struct()` - Identifies struct definitions
        *   `visit_item_enum()` - Captures enum declarations

### `knowledge_engine_metrics.rs`

*   **Purpose:** Specialized metrics collection for the knowledge graph and brain functionality.

*   **Core Class: `KnowledgeEngineMetricsCollector`**
    *   **Description:** Interfaces with KnowledgeEngine to extract brain-specific performance metrics
    *   **Key Methods:**
        *   `collect()` - Gathers comprehensive brain metrics including entity counts, memory usage, graph topology

*   **Key Metrics Collected:**
    *   Entity and relationship counts
    *   Graph density and clustering coefficients  
    *   Memory usage (nodes, triples, total bytes)
    *   Learning efficiency and concept coherence
    *   Activation levels and performance indicators

### `runtime_profiler.rs`

*   **Purpose:** Real-time function execution tracing and performance bottleneck detection.

*   **Key Structures:**
    *   `ExecutionTrace` - Complete function call trace with timing and context
    *   `RuntimeMetrics` - Aggregated runtime performance data
    *   `ExecutionStats` - Statistical analysis of function performance (avg, p50, p95, p99)
    *   `PerformanceBottleneck` - Identified performance issues with suggested fixes
    *   `HotPath` - Frequently executed code paths

*   **Core Class: `RuntimeProfiler`**
    *   **Description:** Traces function execution and identifies performance bottlenecks
    *   **Key Methods:**
        *   `start_function_trace()` - Begins tracking function execution with parameters
        *   `end_function_trace()` - Completes trace and calculates execution metrics
        *   `record_memory_allocation()` - Tracks memory usage per function
        *   `analyze_performance_bottlenecks()` - Identifies slow functions and high memory usage
        *   `analyze_hot_paths()` - Detects frequently executed code paths

*   **Macro: `trace_function!`**
    *   **Purpose:** Convenient macro for automatic function tracing with RAII cleanup
    *   **Usage:** `trace_function!(profiler, "function_name", param1, param2)`

### `test_execution_tracker.rs`

*   **Purpose:** Comprehensive test suite monitoring, execution, and health analysis.

*   **Key Structures:**
    *   `TestSuite` - Complete test suite definition with configuration
    *   `TestCase` - Individual test metadata and requirements
    *   `TestExecution` - Test run results with performance and coverage data
    *   `TestMetrics` - Aggregated test health and performance data
    *   `CoverageData` - Code coverage analysis results
    *   `TestHealth` - Overall test suite health assessment with recommendations

*   **Core Class: `TestExecutionTracker`**
    *   **Description:** Discovers, executes, and monitors test suites across multiple frameworks
    *   **Key Methods:**
        *   `discover_test_suites()` - Auto-discovers Rust and TypeScript test suites
        *   `execute_test_suite()` - Runs tests with coverage and performance monitoring
        *   `execute_rust_tests()` - Executes Cargo tests with optional coverage via tarpaulin
        *   `execute_jest_tests()` - Runs Jest/TypeScript tests with coverage
        *   `update_test_health()` - Calculates test suite health scores and recommendations

*   **Functions:**
    *   `discover_rust_tests()` - Finds Rust test functions using regex parsing
    *   `discover_typescript_tests()` - Locates Jest/Mocha test cases
    *   `parse_rust_test_output()` - Processes Cargo test JSON output
    *   `parse_jest_test_output()` - Handles Jest JSON result parsing

## 5. Key Variables and Logic

### Performance Thresholds
*   **Slow Function Threshold:** 100ms average execution time triggers bottleneck detection
*   **Memory Usage Threshold:** 10MB allocation triggers high memory usage alerts
*   **Frequent Call Threshold:** 1000+ calls triggers frequency optimization recommendations

### Health Score Calculations
*   **Test Health:** Weighted average of coverage (25%), reliability (25%), performance (25%), maintainability (25%)
*   **API Health:** Based on error rates, response times, and endpoint availability
*   **Graph Density:** `total_triples / (entity_count * (entity_count - 1))`

### File System Traversal
*   **Max Depth Limit:** 10 levels to prevent stack overflow in deep directory structures
*   **Skip Patterns:** `.git`, `target`, `node_modules`, `.cache`, `.vscode` for performance

## 6. Dependencies

### Internal Dependencies
*   `crate::monitoring::metrics::MetricRegistry` - Core metrics registration and storage
*   `crate::core::knowledge_engine::KnowledgeEngine` - Knowledge graph access for brain metrics

### External Dependencies
*   **serde/serde_json** - Data serialization and JSON parsing
*   **tokio** - Async runtime for concurrent operations
*   **reqwest** - HTTP client for endpoint testing
*   **syn** - Rust AST parsing for code analysis
*   **walkdir** - Efficient directory traversal
*   **notify** - File system change monitoring
*   **uuid** - Unique identifier generation
*   **parking_lot** - High-performance RwLock implementation

## 7. Common Patterns

### MetricsCollector Trait Implementation
All collectors implement the `MetricsCollector` trait with:
*   `collect()` - Registers metrics with the MetricRegistry
*   `name()` - Returns collector identifier
*   `is_enabled()` - Checks configuration for collector enablement

### Event Broadcasting
Most collectors use `tokio::sync::broadcast` channels for real-time event streaming to subscribers.

### Resource Management
Collectors use configurable limits for memory usage:
*   Maximum history retention (10,000 items for API requests)
*   Maximum timeline events (10,000 for runtime profiler)
*   Maximum execution history (1,000 for test tracker)

## 8. Testing Infrastructure

Each collector includes comprehensive test coverage:
*   **Unit Tests:** Individual function testing with mocked dependencies
*   **Integration Tests:** Full workflow testing with real data
*   **Performance Tests:** Multi-request scenarios with realistic load patterns
*   **Discovery Tests:** Validation of auto-discovery mechanisms for endpoints and tests

## 9. Configuration and Extensibility

### Configurable Parameters
*   History retention limits
*   Performance thresholds for bottleneck detection
*   Test framework support (extensible enum patterns)
*   Output formats (JSON, XML, TAP for test results)

### Extension Points
*   New test frameworks can be added via `TestFramework` enum
*   Additional HTTP methods supported through `HttpMethod` enum
*   Custom bottleneck types via `BottleneckType` enum
*   Pluggable validation rules for API parameters

This monitoring system provides comprehensive observability for the LLMKG system, enabling real-time performance tracking, code quality analysis, and operational health monitoring across all system components.