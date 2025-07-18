# LLMKG Integration Testing Guide

## Overview

The LLMKG Integration Testing Framework validates the complete system functionality by testing the interaction between all major components. This comprehensive test suite ensures that the system performs correctly in real-world scenarios across different platforms and deployment configurations.

## Test Architecture

### Test Categories

1. **Graph-Storage Integration Tests** (`tests/integration/graph_storage_integration.rs`)
   - CSR storage format validation
   - Bloom filter integration
   - Attribute indexing functionality
   - Persistence and serialization
   - Concurrent access patterns

2. **Embedding-Graph Integration Tests** (`tests/integration/embedding_graph_integration.rs`)
   - Quantization with graph structures
   - SIMD performance integration
   - Embedding-graph consistency
   - Multi-modal embeddings
   - Incremental updates

3. **WebAssembly Integration Tests** (`tests/integration/wasm_integration.rs`)
   - WASM runtime functionality
   - Cross-platform compatibility
   - Memory management in WASM
   - Async operations
   - Performance in browser environment

4. **MCP Integration Tests** (`tests/integration/mcp_integration.rs`)
   - Server tool functionality
   - Federation capabilities
   - Streaming responses
   - Tool composition
   - Performance under load

5. **Performance Integration Tests** (`tests/integration/performance_integration.rs`)
   - End-to-end latency validation
   - Memory efficiency testing
   - Compression integration
   - Concurrent access performance
   - Stress testing

### Test Infrastructure

The test infrastructure is built around several key components:

- **`IntegrationTestEnvironment`**: Manages test lifecycle and data collection
- **`TestDataGenerator`**: Creates synthetic test data for various scenarios
- **`PerformanceMonitor`**: Tracks metrics and analyzes performance
- **`ValidationUtils`**: Provides result validation and accuracy checking

## Running Integration Tests

### Local Execution

```bash
# Run all integration tests
cargo test --test '*integration*' --release

# Run specific test category
cargo test --test graph_storage_integration --release
cargo test --test embedding_graph_integration --release
cargo test --test wasm_integration --target wasm32-unknown-unknown
cargo test --test mcp_integration --release
cargo test --test performance_integration --release

# Run with specific features
cargo test --test '*integration*' --features=full --release

# Run stress tests (may take hours)
cargo test --test performance_integration --release -- stress --ignored
```

### Environment Variables

```bash
# Test configuration
export RUST_TEST_TIME_UNIT=30000          # Unit test timeout (ms)
export RUST_TEST_TIME_INTEGRATION=300000  # Integration test timeout (ms)
export LLMKG_KEEP_TEST_DATA=true         # Keep test artifacts for analysis
export LLMKG_LOG_LEVEL=debug             # Detailed logging
export LLMKG_STRESS_TEST_SIZE=medium     # Stress test size (small/medium/large)

# Performance tuning
export RUST_BACKTRACE=full               # Full stack traces
export RAYON_NUM_THREADS=8               # Parallel processing threads
```

### Docker-based Testing

```bash
# Build integration test container
docker build -f tests/integration/Dockerfile -t llmkg-integration .

# Run single-node tests
docker run --rm -e SCENARIO=single-node llmkg-integration

# Run distributed tests
docker run --rm -e SCENARIO=distributed llmkg-integration

# Run federation tests
docker run --rm -e SCENARIO=federation llmkg-integration

# Run with custom configuration
docker run --rm \
  -e SCENARIO=performance \
  -e TEST_TIMEOUT=7200 \
  -e PARALLEL_TESTS=true \
  -v $(pwd)/results:/results \
  llmkg-integration
```

## CI/CD Integration

### GitHub Actions Workflow

The integration tests run automatically on:
- Push to main/develop branches
- Pull requests
- Nightly schedule (full test suite)
- Manual triggers with labels

Test matrix includes:
- **Platforms**: Ubuntu, Windows, macOS
- **Rust versions**: Stable, Beta
- **Feature sets**: Default, Full, Minimal

### Test Artifacts

Results are automatically collected and stored:
- Test result XML files (JUnit format)
- Performance metrics (JSON)
- Resource usage data (CSV)
- HTML reports with charts
- Log files for debugging

Access artifacts via GitHub Actions:
1. Go to Actions tab in repository
2. Select completed workflow run
3. Download artifacts from "Artifacts" section

## Test Scenarios

### Single-Node Testing

Validates core functionality on a single machine:
- Component integration
- Performance characteristics
- Memory usage patterns
- WASM compatibility

**Typical runtime**: 10-20 minutes

### Distributed Testing

Tests multi-node scenarios:
- Cross-node communication
- Data consistency
- Load balancing
- Fault tolerance

**Typical runtime**: 20-40 minutes

### Federation Testing

Validates federated database scenarios:
- Multi-database queries
- Cross-shard relationships
- Federation performance
- Data partitioning

**Typical runtime**: 30-60 minutes

### Stress Testing

Extended testing under high load:
- Large dataset processing (50k+ entities)
- Extended runtime (1+ hours)
- Resource exhaustion scenarios
- Performance degradation analysis

**Typical runtime**: 1-3 hours

## Performance Targets

### Latency Requirements

| Operation | Target | Maximum |
|-----------|---------|---------|
| Single-hop query | < 100μs | < 1ms |
| Multi-hop traversal (depth 3) | < 5ms | < 10ms |
| Similarity search (top-20) | < 2ms | < 5ms |
| RAG context assembly | < 20ms | < 50ms |
| MCP tool request | < 10ms | < 100ms |

### Memory Efficiency

| Component | Target | Maximum |
|-----------|---------|---------|
| Graph overhead | < 20 bytes/entity | < 30 bytes/entity |
| Relationship storage | < 15 bytes/relationship | < 20 bytes/relationship |
| Total per entity | < 60 bytes | < 70 bytes |
| Embedding compression | > 10x ratio | > 8x ratio |

### Throughput Targets

| Scenario | Target | Minimum |
|----------|---------|---------|
| Concurrent reads | > 5000 ops/sec | > 1000 ops/sec |
| Mixed read/write | > 1000 ops/sec | > 500 ops/sec |
| MCP federation | > 100 req/sec | > 50 req/sec |
| WASM operations | > 500 ops/sec | > 100 ops/sec |

## Troubleshooting

### Common Issues

**Test Timeouts**
```bash
# Increase timeout for slow machines
export RUST_TEST_TIME_INTEGRATION=600000

# Run tests sequentially to reduce resource contention
cargo test --test '*integration*' -- --test-threads=1
```

**Memory Issues**
```bash
# Reduce test data size
export LLMKG_TEST_SCALE=small

# Monitor memory usage during tests
docker run --rm --memory=4g llmkg-integration
```

**WASM Test Failures**
```bash
# Install required tools
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
npm install -g @web/test-runner

# Check browser availability
wasm-pack test --chrome --headless
```

**MCP Federation Issues**
```bash
# Check network connectivity
export LLMKG_FEDERATION_TIMEOUT=30000

# Enable detailed federation logging
export LLMKG_MCP_LOG_LEVEL=trace
```

### Performance Analysis

**Slow Tests**
1. Check system resources (CPU, memory, disk)
2. Review test logs for bottlenecks
3. Compare with baseline performance metrics
4. Use profiling tools if available

**Memory Leaks**
1. Run with Valgrind (Linux only):
   ```bash
   valgrind --tool=memcheck cargo test --test memory_integration
   ```
2. Monitor memory usage over time
3. Check for proper resource cleanup

**Inconsistent Results**
1. Verify test isolation
2. Check for race conditions
3. Ensure deterministic test data
4. Review timing-sensitive assertions

## Test Data Management

### Synthetic Data Generation

The test framework generates realistic synthetic data:

```rust
// Academic publication network
let scenario = data_generator.generate_academic_scenario(
    1000,  // papers
    300,   // authors
    50,    // venues
    128    // embedding dimension
);

// Scale-free graph topology
let graph_spec = GraphSpec {
    entity_count: 10000,
    relationship_count: 25000,
    topology: TopologyType::ScaleFree { exponent: 2.1 },
    clustering_coefficient: 0.3,
};

// Correlated embeddings and graph structure
let correlated_data = data_generator.generate_correlated_graph_embeddings(
    500,   // entities
    1000,  // relationships
    128,   // dimension
    0.8    // correlation strength
);
```

### Test Environment Cleanup

Tests automatically clean up temporary data unless explicitly preserved:

```bash
# Keep test data for analysis
export LLMKG_KEEP_TEST_DATA=true

# Manual cleanup
rm -rf /tmp/llmkg_integration_tests/*
```

## Extending the Test Suite

### Adding New Integration Tests

1. **Create test file**: `tests/integration/new_feature_integration.rs`

2. **Import test infrastructure**:
   ```rust
   use crate::test_infrastructure::*;
   ```

3. **Implement test cases**:
   ```rust
   #[test]
   fn test_new_feature_integration() {
       let mut test_env = IntegrationTestEnvironment::new("new_feature");
       
       // Test implementation
       
       test_env.record_performance("operation_time", duration);
       test_env.record_metric("accuracy", score);
   }
   ```

4. **Add to CI/CD pipeline** in `.github/workflows/integration-tests.yml`

### Custom Test Scenarios

Create scenario-specific data generators:

```rust
impl TestDataGenerator {
    pub fn generate_custom_scenario(&mut self, params: CustomParams) -> CustomData {
        // Custom data generation logic
    }
}
```

### Performance Benchmarks

Add performance-specific tests:

```rust
#[test]
fn test_feature_performance() {
    let mut test_env = IntegrationTestEnvironment::new("feature_perf");
    
    // Benchmark implementation
    let benchmark = test_env.performance_monitor.start_measurement("feature_op");
    
    // Execute operation
    
    let duration = benchmark.complete();
    assert!(duration < TARGET_DURATION);
}
```

## Reporting and Analysis

### HTML Reports

Comprehensive HTML reports are generated automatically:
- Test execution summary
- Performance charts
- Resource usage graphs
- Failure analysis
- Cross-platform comparison

### Metrics Collection

Key metrics tracked automatically:
- Execution times
- Memory usage
- Throughput rates
- Error rates
- Resource utilization

### Performance Regression Detection

The CI/CD pipeline detects performance regressions:
- Compares against baseline measurements
- Flags significant degradations
- Provides detailed analysis in PR comments

## Best Practices

### Test Design

1. **Isolation**: Each test should be independent and repeatable
2. **Realistic Data**: Use representative data sizes and distributions
3. **Error Handling**: Test both success and failure scenarios
4. **Resource Management**: Clean up properly after tests
5. **Timing**: Use appropriate timeouts for different operations

### Performance Testing

1. **Baseline Establishment**: Establish performance baselines early
2. **Consistent Environment**: Use consistent test environments
3. **Multiple Runs**: Average results across multiple runs
4. **Statistical Significance**: Ensure sufficient sample sizes
5. **Trend Analysis**: Monitor performance trends over time

### Debugging

1. **Detailed Logging**: Use structured logging for analysis
2. **Test Artifacts**: Preserve test data for investigation
3. **Incremental Testing**: Start with smaller test cases
4. **Profiling**: Use profiling tools for performance issues
5. **Cross-Platform**: Test on multiple platforms early

## Maintenance

### Regular Tasks

1. **Update Performance Baselines**: Quarterly review and update
2. **Test Data Refresh**: Regenerate test datasets as needed
3. **Dependency Updates**: Keep test dependencies current
4. **Platform Compatibility**: Test on new platform versions
5. **Documentation Updates**: Keep documentation synchronized

### Monitoring

1. **CI/CD Health**: Monitor test execution times and failure rates
2. **Resource Usage**: Track resource consumption trends
3. **Test Coverage**: Ensure adequate integration test coverage
4. **Performance Trends**: Monitor for gradual performance degradation

The integration testing framework provides comprehensive validation of LLMKG functionality while maintaining fast feedback cycles for development. Regular maintenance and updates ensure the test suite continues to provide value as the system evolves.