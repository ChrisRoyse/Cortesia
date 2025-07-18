# LLMKG Unit Testing Framework

This document provides comprehensive documentation for the LLMKG Unit Testing Framework, which ensures 100% code coverage with deterministic, predictable outcomes for all unit-level functionality.

## Overview

The Phase 3 Unit Testing Framework provides:

- **Complete Code Coverage**: Tests every function, method, and code path
- **Component Isolation**: Tests each module independently with mocked dependencies  
- **Edge Case Coverage**: Tests all boundary conditions and error scenarios
- **Performance Validation**: Verifies unit-level performance characteristics
- **Deterministic Results**: Ensures all tests produce identical, predictable outcomes
- **Regression Prevention**: Catches breaking changes at the unit level

## Quick Start

### Running All Unit Tests

```bash
cd tests
cargo run --bin unit-test-runner
```

### Running Specific Module Tests

```bash
# Core module tests
cargo run --bin unit-test-runner -- --module core

# Storage layer tests  
cargo run --bin unit-test-runner -- --module storage

# Embedding system tests
cargo run --bin unit-test-runner -- --module embedding

# Query engine tests
cargo run --bin unit-test-runner -- --module query
```

### With Coverage Analysis

```bash
cargo run --bin unit-test-runner -- --coverage --verbose
```

### Generate HTML Report

```bash
cargo run --bin unit-test-runner -- --coverage --output html --report-file report.html
```

## Test Organization

### Directory Structure

```
tests/
├── unit/                          # Unit testing framework
│   ├── mod.rs                     # Framework configuration and utilities
│   ├── lib.rs                     # Main unit test library
│   ├── core/                      # Core module tests
│   │   ├── entity_tests.rs        # Entity management tests
│   │   ├── graph_tests.rs         # Graph structure tests
│   │   ├── memory_tests.rs        # Memory management tests
│   │   └── types_tests.rs         # Core types tests
│   ├── storage/                   # Storage layer tests
│   │   ├── csr_tests.rs           # CSR format tests
│   │   ├── bloom_tests.rs         # Bloom filter tests
│   │   ├── index_tests.rs         # Index structure tests
│   │   └── mmap_tests.rs          # Memory-mapped storage tests
│   ├── embedding/                 # Embedding system tests
│   │   ├── quantization_tests.rs  # Vector quantization tests
│   │   ├── simd_tests.rs          # SIMD operations tests
│   │   ├── similarity_tests.rs    # Similarity computation tests
│   │   └── store_tests.rs         # Embedding store tests
│   ├── query/                     # Query engine tests
│   │   ├── rag_tests.rs           # Graph RAG tests
│   │   ├── optimizer_tests.rs     # Query optimizer tests
│   │   ├── clustering_tests.rs    # Clustering tests
│   │   └── summarization_tests.rs # Summarization tests
│   ├── federation/                # Federation system tests
│   ├── mcp/                       # MCP system tests
│   └── wasm/                      # WASM interface tests
├── benches/                       # Performance benchmarks
│   └── unit_benchmarks.rs         # Comprehensive benchmarks
└── src/bin/                       # Test utilities
    └── unit_test_runner.rs         # Main test runner
```

## Test Framework Features

### Deterministic Testing

All tests use deterministic random number generators to ensure reproducible results:

```rust
let mut rng = DeterministicRng::new(ENTITY_TEST_SEED);
let test_data = generate_test_data(&mut rng);
```

### Memory Leak Detection

The framework automatically detects memory leaks and excessive memory usage:

```rust
let config = UnitTestConfig {
    fail_on_memory_leak: true,
    max_memory_bytes: 100 * 1024 * 1024, // 100MB limit
    ..Default::default()
};
```

### Performance Monitoring

Each test tracks execution time and memory usage:

```rust
let result = runner.run_test("test_name", || async {
    // Test implementation
    Ok(())
}).await;

println!("Duration: {}ms", result.duration_ms);
println!("Memory: {} bytes", result.memory_usage_bytes);
```

### Isolated Execution

Tests run in isolated environments to prevent interference:

```rust
let environment = TestEnvironment::new(ResourceLimits {
    max_memory: config.max_memory_bytes,
    max_cpu_percent: 50.0,
    timeout: Duration::from_millis(config.timeout_ms),
})?;
```

## Core Module Tests

### Entity Management Tests (`core/entity_tests.rs`)

Tests for entity creation, manipulation, serialization, and memory management:

- `test_entity_creation_deterministic()` - Deterministic entity creation
- `test_entity_key_generation()` - Key generation and collision resistance
- `test_entity_attribute_edge_cases()` - Edge cases for attributes
- `test_entity_memory_management()` - Memory usage tracking
- `test_entity_serialization_formats()` - JSON/binary serialization
- `test_entity_concurrent_access()` - Thread safety

### Graph Structure Tests (`core/graph_tests.rs`)

Tests for knowledge graph operations and CSR storage:

- `test_graph_basic_operations()` - Entity/relationship CRUD
- `test_graph_csr_storage_format()` - CSR format properties
- `test_graph_memory_efficiency()` - Memory scaling characteristics
- `test_graph_concurrent_access()` - Multi-threaded access
- `test_graph_pathfinding()` - Shortest path algorithms
- `test_graph_serialization()` - Graph persistence

### Memory Management Tests (`core/memory_tests.rs`)

Tests for memory allocation and leak detection:

- `test_memory_manager_basic_operations()` - Allocation/deallocation
- `test_memory_leak_detection()` - Leak detection algorithms
- `test_memory_fragmentation()` - Fragmentation handling
- `test_memory_pool_allocation()` - Pool-based allocation
- `test_memory_performance()` - Allocation performance

## Storage Layer Tests

### CSR Format Tests (`storage/csr_tests.rs`)

Tests for Compressed Sparse Row format:

- `test_csr_construction_deterministic()` - Matrix conversion
- `test_csr_access_patterns()` - Cache-friendly access
- `test_csr_memory_layout()` - Memory efficiency
- `test_csr_operations()` - Matrix operations
- `test_csr_serialization()` - Persistence
- `test_csr_performance_characteristics()` - Scaling behavior

### Bloom Filter Tests (`storage/bloom_tests.rs`)

Tests for bloom filter implementation:

- `test_bloom_filter_basic_operations()` - Insert/lookup operations
- `test_bloom_filter_deterministic()` - Deterministic behavior
- `test_bloom_filter_false_positive_rates()` - FP rate validation
- `test_bloom_filter_hash_function_quality()` - Hash distribution
- `test_bloom_filter_performance()` - Throughput testing
- `test_bloom_filter_concurrent_access()` - Thread safety

## Embedding System Tests

### Vector Quantization Tests (`embedding/quantization_tests.rs`)

Tests for product quantization:

- `test_product_quantization_accuracy()` - Compression quality
- `test_quantizer_training_convergence()` - Training stability
- `test_quantization_edge_cases()` - Boundary conditions
- `test_quantization_memory_efficiency()` - Memory usage
- `test_quantizer_serialization()` - Model persistence
- `test_quantizer_performance()` - Speed benchmarks

### SIMD Operations Tests (`embedding/simd_tests.rs`)

Tests for SIMD-accelerated operations:

- `test_simd_distance_computation()` - Distance calculations
- `test_simd_batch_operations()` - Batch processing
- `test_simd_alignment_requirements()` - Memory alignment
- `test_simd_vector_operations()` - Vector arithmetic
- `test_simd_performance_scaling()` - Performance scaling
- `test_simd_numerical_stability()` - Numerical precision

## Query Engine Tests

### Graph RAG Tests (`query/rag_tests.rs`)

Tests for Graph RAG functionality:

- `test_graph_rag_context_assembly()` - Context generation
- `test_rag_similarity_search_integration()` - Embedding integration
- `test_rag_context_quality_metrics()` - Quality measurement
- `test_rag_multi_strategy_integration()` - Strategy combination
- `test_rag_performance_characteristics()` - Scaling behavior
- `test_rag_concurrent_access()` - Thread safety

## Running Tests

### Command Line Options

```bash
# Basic usage
cargo run --bin unit-test-runner

# With coverage
--coverage                 Enable coverage analysis

# Output formats
--output console          Console output (default)
--output json             JSON format
--output junit            JUnit XML format  
--output html             HTML report

# Filtering
--module core             Run specific module
--timeout 10000           Set timeout (ms)
--memory-limit 200000000  Set memory limit (bytes)

# Reporting
--report-file report.html Output file for reports
--verbose                 Verbose output
--fail-fast              Stop on first failure
```

### Environment Variables

```bash
# Logging level
export RUST_LOG=debug

# Test parallelism
export RUST_TEST_THREADS=4

# Coverage output
export LLMKG_COVERAGE_DIR=./coverage
```

## Performance Benchmarks

Run performance benchmarks with:

```bash
cargo bench --bench unit_benchmarks
```

Benchmark categories:

- **Entity Operations**: Creation, attribute access, serialization
- **Graph Operations**: CRUD, traversal, pathfinding
- **Storage Operations**: CSR construction, bloom filters
- **Embedding Operations**: Quantization, SIMD processing
- **Query Operations**: RAG context assembly, similarity search
- **Memory Operations**: Allocation, pool management
- **Concurrent Operations**: Multi-threaded performance

## Coverage Analysis

### Generating Coverage Reports

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cd tests
cargo tarpaulin --verbose --all-features --workspace --timeout 300 \
  --out xml --output-dir coverage/ \
  --exclude-files "src/bin/*" "examples/*" "tests/*" \
  --run-types Tests \
  --fail-under 95
```

### Coverage Targets

- **Line Coverage**: 100% target, 95% minimum
- **Branch Coverage**: 100% target, 90% minimum  
- **Function Coverage**: 100% target, 95% minimum

## CI Integration

### GitHub Actions

The framework integrates with GitHub Actions for automated testing:

```yaml
# .github/workflows/unit-tests.yml
- name: Run unit tests with coverage
  run: |
    cd tests
    cargo run --bin unit-test-runner -- \
      --coverage \
      --verbose \
      --output junit \
      --report-file unit-test-results.xml
```

### Test Matrix

- **Rust Versions**: stable, beta, nightly
- **Platforms**: Ubuntu, Windows, macOS
- **Features**: Various feature combinations
- **Memory Testing**: Valgrind integration
- **Security**: Cargo audit integration

## Best Practices

### Writing Unit Tests

1. **Test Isolation**: Each test should be completely independent
2. **Deterministic Data**: Use seeded RNGs for reproducible results
3. **Clear Assertions**: Use descriptive assertion messages
4. **Edge Cases**: Test boundary conditions and error paths
5. **Performance**: Include performance expectations
6. **Memory**: Monitor memory usage and detect leaks

### Test Naming

```rust
// Good: Descriptive, specific
fn test_entity_creation_with_unicode_attributes()

// Bad: Vague, generic  
fn test_entity()
```

### Error Testing

```rust
#[test]
fn test_entity_invalid_key_format() {
    let result = Entity::from_invalid_key("invalid\0key");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("invalid key format"));
}
```

### Performance Testing

```rust
#[test]
fn test_graph_lookup_performance() {
    let graph = create_test_graph(10000, 20000);
    let entity_key = EntityKey::from_hash("test_entity_0");
    
    let (_, duration) = measure_execution_time(|| {
        for _ in 0..1000 {
            let _ = graph.get_entity(entity_key);
        }
    });
    
    assert!(duration.as_micros() < 10000, "Lookup too slow: {:?}", duration);
}
```

## Troubleshooting

### Common Issues

1. **Test Timeouts**: Increase timeout with `--timeout` flag
2. **Memory Limits**: Adjust with `--memory-limit` flag  
3. **Flaky Tests**: Check for non-deterministic behavior
4. **Coverage Issues**: Ensure all code paths are tested

### Debug Mode

```bash
# Run with debug output
RUST_LOG=debug cargo run --bin unit-test-runner -- --verbose

# Run specific failing test
cargo test test_name -- --nocapture
```

### Memory Debugging

```bash
# Run with Valgrind
valgrind --tool=memcheck --leak-check=full \
  ./target/debug/unit-test-runner --module core
```

## Contributing

### Adding New Tests

1. Create test file in appropriate module directory
2. Follow naming conventions and best practices
3. Include comprehensive edge case coverage
4. Add performance expectations
5. Update documentation

### Test Infrastructure

The framework provides utilities for common testing patterns:

```rust
use llmkg_tests::*;

// Create deterministic test data
let mut rng = DeterministicRng::new(TEST_SEED);
let test_graph = create_test_graph(100, 200);

// Measure performance
let (result, duration) = measure_execution_time(|| {
    expensive_operation()
});

// Assert vectors are equal within tolerance
assert_vectors_equal(&expected, &actual, 1e-6);
```

## Examples

### Complete Test Example

```rust
#[test]
fn test_entity_comprehensive() {
    // Setup with deterministic data
    let mut rng = DeterministicRng::new(ENTITY_TEST_SEED);
    let entity_key = EntityKey::from_hash("test_entity");
    let mut entity = Entity::new(entity_key, "Test Entity".to_string());
    
    // Test basic operations
    assert_eq!(entity.name(), "Test Entity");
    assert_eq!(entity.attributes().len(), 0);
    
    // Test attribute operations
    entity.add_attribute("type", "test");
    entity.add_attribute("value", "42");
    
    assert_eq!(entity.attributes().len(), 2);
    assert_eq!(entity.get_attribute("type"), Some("test"));
    assert_eq!(entity.get_attribute("value"), Some("42"));
    
    // Test memory tracking
    let expected_memory = calculate_expected_entity_memory(&entity);
    assert_eq!(entity.memory_usage(), expected_memory);
    
    // Test serialization
    let serialized = entity.serialize();
    let deserialized = Entity::deserialize(&serialized).unwrap();
    assert_eq!(entity, deserialized);
    
    // Test performance
    let (_, duration) = measure_execution_time(|| {
        for i in 0..1000 {
            entity.add_attribute(&format!("perf_{}", i), &format!("value_{}", i));
        }
    });
    
    assert!(duration.as_millis() < 100, "Attribute addition too slow");
}
```

This comprehensive unit testing framework ensures that every component of LLMKG is thoroughly validated in isolation, providing a solid foundation for integration and end-to-end testing phases.