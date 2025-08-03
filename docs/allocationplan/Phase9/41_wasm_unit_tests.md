# Micro-Phase 9.41: WASM Unit Tests

## Objective
Implement comprehensive unit testing infrastructure for WASM modules using wasm-bindgen-test to ensure code quality and reliability.

## Prerequisites
- Completed micro-phase 9.40 (Performance Monitoring)
- WASM bindings and core functionality implemented (phases 9.04-9.08)
- Build configuration and dependencies setup (phases 9.01-9.03)

## Task Description
Create robust unit testing framework with wasm-bindgen-test covering all WASM functionality including memory management, allocation algorithms, and SIMD operations. Implement automated test execution pipeline with coverage reporting and performance benchmarks.

## Specific Actions

1. **Configure WASM test environment**
```toml
# Add to Cargo.toml
[dependencies]
wasm-bindgen-test = "0.3"
console_error_panic_hook = "0.1"
web-sys = { version = "0.3", features = ["console", "Performance"] }

[dev-dependencies]
wasm-bindgen-test = "0.3"

[lib]
crate-type = ["cdylib"]

[[bin]]
name = "test-runner"
path = "tests/test_runner.rs"
```

2. **Create core allocation algorithm tests**
```rust
// tests/allocation_tests.rs
use wasm_bindgen_test::*;
use crate::cortex_kg::{CortexKG, AllocationConfig, Column};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_cortex_kg_initialization() {
    console_error_panic_hook::set_once();
    
    let config = AllocationConfig {
        columns: 100,
        minicolumns_per_column: 32,
        cells_per_minicolumn: 8,
        proximal_threshold: 0.5,
        distal_threshold: 0.3,
    };
    
    let cortex = CortexKG::new(config);
    assert_eq!(cortex.get_column_count(), 100);
    assert_eq!(cortex.get_total_cell_count(), 100 * 32 * 8);
}

#[wasm_bindgen_test]
fn test_memory_allocation_patterns() {
    let mut cortex = create_test_cortex();
    
    // Test initial memory state
    let initial_memory = cortex.get_memory_usage();
    assert!(initial_memory.allocated > 0);
    assert!(initial_memory.free > 0);
    
    // Allocate large concept
    let concept_id = cortex.allocate_concept("large_test_concept", 1000);
    let after_allocation = cortex.get_memory_usage();
    
    assert!(after_allocation.allocated > initial_memory.allocated);
    assert_eq!(after_allocation.free, initial_memory.free - 1000);
    
    // Deallocate and verify cleanup
    cortex.deallocate_concept(concept_id);
    let after_deallocation = cortex.get_memory_usage();
    
    assert_eq!(after_deallocation.allocated, initial_memory.allocated);
    assert_eq!(after_deallocation.free, initial_memory.free);
}

#[wasm_bindgen_test]
fn test_spatial_pooling_algorithm() {
    let mut cortex = create_test_cortex();
    
    // Create test input pattern
    let input_pattern = vec![1, 0, 1, 1, 0, 0, 1, 0];
    let expected_active_columns = 10; // 10% sparsity
    
    let result = cortex.spatial_pooling(&input_pattern);
    
    assert_eq!(result.active_columns.len(), expected_active_columns);
    assert!(result.overlap_scores.len() > 0);
    assert!(result.boost_factors.iter().all(|&f| f >= 1.0));
    
    // Verify column activation distribution
    let activation_counts = count_column_activations(&result.active_columns);
    assert!(activation_counts.max() <= 1); // No column activated multiple times
}

#[wasm_bindgen_test]
fn test_temporal_memory_learning() {
    let mut cortex = create_test_cortex();
    
    // Train on sequence pattern
    let sequence = vec![
        vec![1, 0, 1, 0],
        vec![0, 1, 0, 1],
        vec![1, 1, 0, 0],
        vec![0, 0, 1, 1],
    ];
    
    // Learn sequence
    for pattern in &sequence {
        cortex.compute(pattern, true); // learning enabled
    }
    
    // Test prediction
    cortex.compute(&sequence[0], false); // learning disabled
    let prediction = cortex.get_prediction();
    
    // Should predict next pattern in sequence
    assert!(prediction.confidence > 0.7);
    assert_eq!(prediction.predicted_cells.len(), sequence[1].len());
}

#[wasm_bindgen_test]
fn test_anomaly_detection() {
    let mut cortex = create_test_cortex();
    
    // Train on normal patterns
    let normal_patterns = vec![
        vec![1, 0, 1, 0, 1, 0],
        vec![0, 1, 0, 1, 0, 1],
        vec![1, 1, 0, 0, 1, 1],
    ];
    
    for _ in 0..10 {
        for pattern in &normal_patterns {
            cortex.compute(pattern, true);
        }
    }
    
    // Test with anomalous pattern
    let anomalous_pattern = vec![1, 1, 1, 1, 1, 1];
    let result = cortex.compute(&anomalous_pattern, false);
    
    assert!(result.anomaly_score > 0.8);
    assert!(result.raw_anomaly_score > result.anomaly_likelihood);
}
```

3. **Create SIMD operation tests**
```rust
// tests/simd_tests.rs
use wasm_bindgen_test::*;
use crate::simd_processor::{SIMDNeuralProcessor, VectorOperation};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_simd_vector_operations() {
    let processor = SIMDNeuralProcessor::new();
    
    let vec_a = vec![1.0, 2.0, 3.0, 4.0];
    let vec_b = vec![2.0, 3.0, 4.0, 5.0];
    
    // Test vector addition
    let result = processor.vector_add(&vec_a, &vec_b);
    assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);
    
    // Test dot product
    let dot_product = processor.dot_product(&vec_a, &vec_b);
    assert_eq!(dot_product, 40.0); // 1*2 + 2*3 + 3*4 + 4*5
    
    // Test vector normalization
    let normalized = processor.normalize(&vec_a);
    let magnitude = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((magnitude - 1.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_simd_matrix_operations() {
    let processor = SIMDNeuralProcessor::new();
    
    let matrix_a = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];
    let matrix_b = vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    
    let result = processor.matrix_multiply(&matrix_a, &matrix_b);
    
    // Expected: [[19, 22], [43, 50]]
    assert_eq!(result[0][0], 19.0);
    assert_eq!(result[0][1], 22.0);
    assert_eq!(result[1][0], 43.0);
    assert_eq!(result[1][1], 50.0);
}

#[wasm_bindgen_test]
fn test_simd_performance_benchmarks() {
    let processor = SIMDNeuralProcessor::new();
    let large_vector = (0..10000).map(|i| i as f32).collect::<Vec<_>>();
    
    let start_time = web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now();
    
    // Perform intensive SIMD operations
    for _ in 0..100 {
        let _ = processor.vector_add(&large_vector, &large_vector);
    }
    
    let end_time = web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now();
    
    let execution_time = end_time - start_time;
    
    // Should complete within reasonable time (< 100ms for 1M operations)
    assert!(execution_time < 100.0);
    
    web_sys::console::log_1(&format!("SIMD operations took: {}ms", execution_time).into());
}
```

4. **Create memory management tests**
```rust
// tests/memory_tests.rs
use wasm_bindgen_test::*;
use crate::memory::{WASMAllocator, MemoryPool, AllocationStrategy};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_memory_allocator_basic() {
    let mut allocator = WASMAllocator::new(1024 * 1024); // 1MB
    
    // Test basic allocation
    let ptr1 = allocator.allocate(100).unwrap();
    assert!(!ptr1.is_null());
    
    let ptr2 = allocator.allocate(200).unwrap();
    assert!(!ptr2.is_null());
    assert_ne!(ptr1, ptr2);
    
    // Test deallocation
    allocator.deallocate(ptr1, 100);
    allocator.deallocate(ptr2, 200);
    
    let stats = allocator.get_stats();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.total_deallocated, 300);
}

#[wasm_bindgen_test]
fn test_memory_pool_efficiency() {
    let mut pool = MemoryPool::new(AllocationStrategy::FixedSize(64));
    
    // Allocate many small blocks
    let mut pointers = Vec::new();
    for _ in 0..1000 {
        let ptr = pool.allocate().unwrap();
        pointers.push(ptr);
    }
    
    let stats_after_allocation = pool.get_stats();
    assert_eq!(stats_after_allocation.active_allocations, 1000);
    
    // Deallocate all blocks
    for ptr in pointers {
        pool.deallocate(ptr);
    }
    
    let stats_after_deallocation = pool.get_stats();
    assert_eq!(stats_after_deallocation.active_allocations, 0);
    assert!(stats_after_deallocation.fragmentation_ratio < 0.1);
}

#[wasm_bindgen_test]
fn test_memory_leak_detection() {
    let mut allocator = WASMAllocator::new(1024 * 1024);
    
    // Simulate potential memory leak
    let initial_stats = allocator.get_stats();
    
    {
        let _ptr1 = allocator.allocate(1000).unwrap();
        let _ptr2 = allocator.allocate(2000).unwrap();
        // Intentionally not deallocating to test leak detection
    }
    
    // Force garbage collection check
    allocator.check_for_leaks();
    
    let leak_report = allocator.get_leak_report();
    assert!(leak_report.potential_leaks.len() > 0);
    assert_eq!(leak_report.total_leaked_bytes, 3000);
}
```

5. **Create test utilities and helpers**
```rust
// tests/test_utils.rs
use crate::cortex_kg::{CortexKG, AllocationConfig};

pub fn create_test_cortex() -> CortexKG {
    let config = AllocationConfig {
        columns: 10,
        minicolumns_per_column: 4,
        cells_per_minicolumn: 4,
        proximal_threshold: 0.5,
        distal_threshold: 0.3,
    };
    
    CortexKG::new(config)
}

pub fn generate_random_pattern(size: usize, sparsity: f32) -> Vec<u8> {
    let mut pattern = vec![0; size];
    let active_bits = (size as f32 * sparsity) as usize;
    
    for i in 0..active_bits {
        pattern[i] = 1;
    }
    
    // Shuffle pattern
    use web_sys::js_sys::Math;
    for i in 0..size {
        let j = (Math::random() * size as f64) as usize;
        pattern.swap(i, j);
    }
    
    pattern
}

pub fn assert_vector_equals(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&val_a, &val_b)) in a.iter().zip(b.iter()).enumerate() {
        assert!((val_a - val_b).abs() < tolerance, 
               "Values differ at index {}: {} vs {}", i, val_a, val_b);
    }
}

#[macro_export]
macro_rules! assert_performance {
    ($operation:expr, $max_time_ms:expr) => {
        let start = web_sys::window().unwrap().performance().unwrap().now();
        $operation;
        let end = web_sys::window().unwrap().performance().unwrap().now();
        let duration = end - start;
        assert!(duration < $max_time_ms, 
               "Operation took {}ms, expected < {}ms", duration, $max_time_ms);
    };
}
```

## Expected Outputs
- Comprehensive WASM unit test suite with 95%+ code coverage
- Automated test execution pipeline with CI/CD integration
- Performance benchmarks for critical algorithms and memory operations
- Test utilities and macros for consistent testing patterns
- Detailed test reports with coverage analysis and performance metrics

## Validation
1. Verify all WASM functions have corresponding unit tests with edge case coverage
2. Confirm test execution completes in under 30 seconds for full suite
3. Test memory allocation and deallocation scenarios without leaks
4. Validate SIMD operations produce correct results across different data sizes
5. Ensure test utilities provide consistent and reliable helper functions

## Next Steps
- Proceed to micro-phase 9.42 (JavaScript Unit Tests)
- Integrate with CI/CD pipeline for automated testing
- Configure test coverage reporting and quality gates