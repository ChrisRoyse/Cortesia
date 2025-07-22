# GPU Module Tests

This directory contains comprehensive tests for the GPU acceleration module in LLMKG.

## Test Coverage

### `gpu_tests.rs` - Main comprehensive test suite
Tests for the GPU module including:

1. **CUDA Processor Tests** (`cuda_processor_tests`)
   - Tests CUDA processor creation with and without CUDA feature
   - Verifies appropriate error handling (FeatureNotEnabled vs NotImplemented)

2. **CPU Processor Tests** (`cpu_processor_tests`)
   - Basic functionality tests for CPU fallback implementation
   - Parallel traversal operations
   - Batch similarity computations with cosine similarity
   - Parallel shortest path calculations
   - Edge cases: empty inputs, zero vectors, infinity values

3. **Integration Tests** (`integration_tests`)
   - Trait object usage
   - Error propagation
   - Large batch processing performance
   - Edge cases with extreme values

4. **Performance Tests** (`performance_tests`)
   - Batch similarity performance with 10,000 embeddings
   - Parallel traversal performance with 1 million nodes
   - Ensures operations complete within reasonable time limits

### `test_gpu_module.rs` - Simple integration tests
Basic tests to verify the GPU module can be imported and used correctly.

## Running the Tests

```bash
# Run all GPU tests
cargo test gpu::

# Run specific test module
cargo test gpu::gpu_tests::

# Run with verbose output
cargo test gpu:: -- --nocapture
```

## Implementation Notes

- The CPU processor (`CpuGraphProcessor`) provides a working fallback implementation
- The CUDA processor (`CudaGraphProcessor`) is a placeholder that returns appropriate errors
- All tests handle the NotImplemented errors gracefully
- Performance tests ensure the CPU implementation scales reasonably well

## Future Work

When CUDA support is implemented:
1. Add tests for actual CUDA operations
2. Add GPU memory management tests
3. Add tests for CPU-GPU data transfer
4. Add benchmarks comparing CPU vs GPU performance