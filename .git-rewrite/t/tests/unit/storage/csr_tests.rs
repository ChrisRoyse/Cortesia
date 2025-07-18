//! CSR Format Unit Tests
//!
//! Comprehensive tests for Compressed Sparse Row format implementation,
//! including construction, access patterns, operations, and memory layout.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::storage::csr::CompressedSparseRow;
use std::time::Duration;

#[cfg(test)]
mod csr_tests {
    use super::*;

    #[test]
    fn test_csr_construction_deterministic() {
        let mut rng = DeterministicRng::new(CSR_TEST_SEED);
        
        // Create test adjacency matrix
        let n = 100;
        let density = 0.1;
        let adjacency_matrix = generate_random_adjacency_matrix(&mut rng, n, density);
        
        // Convert to CSR format
        let csr = CompressedSparseRow::from_adjacency_matrix(&adjacency_matrix);
        
        // Test 1: Verify structure integrity
        assert_eq!(csr.num_rows(), n);
        assert_eq!(csr.row_offsets().len(), n + 1);
        
        // Test 2: Verify data consistency
        let expected_nnz = count_nonzeros(&adjacency_matrix);
        assert_eq!(csr.num_nonzeros(), expected_nnz);
        assert_eq!(csr.column_indices().len(), expected_nnz);
        assert_eq!(csr.values().len(), expected_nnz);
        
        // Test 3: Verify CSR properties
        let offsets = csr.row_offsets();
        for i in 1..offsets.len() {
            assert!(offsets[i] >= offsets[i-1], "Row offsets not monotonic");
        }
        
        // Test 4: Verify round-trip conversion
        let reconstructed_matrix = csr.to_adjacency_matrix();
        assert_matrices_equal(&adjacency_matrix, &reconstructed_matrix);
        
        // Test 5: Test deterministic construction
        let csr2 = CompressedSparseRow::from_adjacency_matrix(&adjacency_matrix);
        assert_eq!(csr, csr2);
    }
    
    #[test]
    fn test_csr_access_patterns() {
        let csr = create_test_csr_matrix(1000, 0.05);
        
        // Test 1: Sequential row access
        let start_time = std::time::Instant::now();
        for row in 0..csr.num_rows() {
            let _row_data = csr.get_row(row);
        }
        let sequential_time = start_time.elapsed();
        
        // Test 2: Random row access
        let mut rng = DeterministicRng::new(ACCESS_PATTERN_SEED);
        let random_rows: Vec<usize> = (0..csr.num_rows())
            .map(|_| rng.gen_range(0..csr.num_rows()))
            .collect();
        
        let start_time = std::time::Instant::now();
        for &row in &random_rows {
            let _row_data = csr.get_row(row);
        }
        let random_time = start_time.elapsed();
        
        // Sequential access should be faster (cache-friendly)
        let ratio = random_time.as_nanos() as f64 / sequential_time.as_nanos() as f64;
        assert!(ratio > 1.2, "CSR format not showing cache advantage: ratio {}", ratio);
        
        // Test 3: Column access performance
        let start_time = std::time::Instant::now();
        for col in 0..csr.num_rows() {
            let _col_elements = csr.get_column_elements(col);
        }
        let column_time = start_time.elapsed();
        
        // Row access should be much faster than column access
        let row_col_ratio = column_time.as_nanos() as f64 / sequential_time.as_nanos() as f64;
        assert!(row_col_ratio > 5.0, "Column access not showing expected slowdown: ratio {}", row_col_ratio);
    }
    
    #[test]
    fn test_csr_memory_layout() {
        let sizes = vec![10, 100, 1000, 10000];
        let density = 0.05;
        
        for &size in &sizes {
            let csr = create_test_csr_matrix(size, density);
            
            // Calculate expected memory usage
            let expected_offsets_size = (size + 1) * std::mem::size_of::<usize>();
            let nnz = csr.num_nonzeros();
            let expected_indices_size = nnz * std::mem::size_of::<usize>();
            let expected_values_size = nnz * std::mem::size_of::<f32>();
            let expected_total = expected_offsets_size + expected_indices_size + expected_values_size;
            
            let actual_memory = csr.memory_usage();
            
            // Allow small overhead for struct metadata
            let overhead_ratio = actual_memory as f64 / expected_total as f64;
            assert!(overhead_ratio < 1.1, 
                   "CSR memory overhead too high: {} vs {} (ratio: {})", 
                   actual_memory, expected_total, overhead_ratio);
            
            // Verify memory usage scales linearly with non-zeros
            let memory_per_nnz = actual_memory as f64 / nnz as f64;
            let expected_per_nnz = (std::mem::size_of::<usize>() + std::mem::size_of::<f32>()) as f64;
            
            assert!((memory_per_nnz - expected_per_nnz).abs() < expected_per_nnz * 0.2,
                   "Memory per non-zero element incorrect: {} vs {}", 
                   memory_per_nnz, expected_per_nnz);
        }
    }
    
    #[test]
    fn test_csr_operations() {
        let mut rng = DeterministicRng::new(CSR_OPERATIONS_SEED);
        let csr = create_test_csr_matrix(500, 0.08);
        
        // Test 1: Matrix-vector multiplication
        let vector: Vec<f32> = (0..csr.num_rows()).map(|i| (i as f32) * 0.1).collect();
        let result = csr.multiply_vector(&vector);
        
        // Verify result dimensions
        assert_eq!(result.len(), csr.num_rows());
        
        // Verify against naive implementation
        let expected_result = naive_matrix_vector_multiply(&csr, &vector);
        for (i, (&actual, &expected)) in result.iter().zip(expected_result.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6, 
                   "Matrix-vector multiplication mismatch at index {}: {} vs {}", 
                   i, actual, expected);
        }
        
        // Test 2: Sparse matrix addition
        let csr2 = create_test_csr_matrix_with_seed(500, 0.08, CSR_OPERATIONS_SEED + 1);
        let sum_result = csr.add(&csr2);
        
        // Verify dimensions
        assert_eq!(sum_result.num_rows(), csr.num_rows());
        
        // Verify addition correctness by spot-checking
        for _ in 0..100 {
            let row = rng.gen_range(0..csr.num_rows());
            let col = rng.gen_range(0..csr.num_rows());
            
            let val1 = csr.get_element(row, col);
            let val2 = csr2.get_element(row, col);
            let sum_val = sum_result.get_element(row, col);
            
            assert!((sum_val - (val1 + val2)).abs() < 1e-6,
                   "Matrix addition incorrect at ({}, {}): {} vs {}", 
                   row, col, sum_val, val1 + val2);
        }
    }

    #[test]
    fn test_csr_construction_edge_cases() {
        // Test empty matrix
        let empty_matrix = vec![vec![0.0f32; 0]; 0];
        let empty_csr = CompressedSparseRow::from_adjacency_matrix(&empty_matrix);
        assert_eq!(empty_csr.num_rows(), 0);
        assert_eq!(empty_csr.num_nonzeros(), 0);
        
        // Test single element matrix
        let single_matrix = vec![vec![1.0f32]];
        let single_csr = CompressedSparseRow::from_adjacency_matrix(&single_matrix);
        assert_eq!(single_csr.num_rows(), 1);
        assert_eq!(single_csr.num_nonzeros(), 1);
        assert_eq!(single_csr.get_element(0, 0), 1.0);
        
        // Test all-zeros matrix
        let zero_matrix = vec![vec![0.0f32; 5]; 5];
        let zero_csr = CompressedSparseRow::from_adjacency_matrix(&zero_matrix);
        assert_eq!(zero_csr.num_rows(), 5);
        assert_eq!(zero_csr.num_nonzeros(), 0);
        
        // Test identity matrix
        let mut identity_matrix = vec![vec![0.0f32; 4]; 4];
        for i in 0..4 {
            identity_matrix[i][i] = 1.0;
        }
        let identity_csr = CompressedSparseRow::from_adjacency_matrix(&identity_matrix);
        assert_eq!(identity_csr.num_rows(), 4);
        assert_eq!(identity_csr.num_nonzeros(), 4);
        
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(identity_csr.get_element(i, j), expected);
            }
        }
        
        // Test fully dense matrix
        let dense_matrix = vec![vec![1.0f32; 3]; 3];
        let dense_csr = CompressedSparseRow::from_adjacency_matrix(&dense_matrix);
        assert_eq!(dense_csr.num_rows(), 3);
        assert_eq!(dense_csr.num_nonzeros(), 9);
    }

    #[test]
    fn test_csr_serialization() {
        let csr = create_test_csr_matrix(50, 0.1);
        
        // Test binary serialization
        let binary_data = csr.to_binary().unwrap();
        let deserialized_csr = CompressedSparseRow::from_binary(&binary_data).unwrap();
        
        assert_eq!(csr.num_rows(), deserialized_csr.num_rows());
        assert_eq!(csr.num_nonzeros(), deserialized_csr.num_nonzeros());
        assert_eq!(csr.row_offsets(), deserialized_csr.row_offsets());
        assert_eq!(csr.column_indices(), deserialized_csr.column_indices());
        assert_vectors_equal(csr.values(), deserialized_csr.values(), 1e-6);
        
        // Test JSON serialization
        let json_data = csr.to_json().unwrap();
        let json_deserialized = CompressedSparseRow::from_json(&json_data).unwrap();
        
        assert_eq!(csr.num_rows(), json_deserialized.num_rows());
        assert_eq!(csr.num_nonzeros(), json_deserialized.num_nonzeros());
        
        // Binary should be more compact
        assert!(binary_data.len() < json_data.len());
    }

    #[test]
    fn test_csr_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let csr = Arc::new(create_test_csr_matrix(1000, 0.05));
        let thread_count = 4;
        let operations_per_thread = 1000;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let csr_clone = Arc::clone(&csr);
            
            let handle = thread::spawn(move || {
                let mut rng = DeterministicRng::new(CSR_TEST_SEED + thread_id as u64);
                
                for _ in 0..operations_per_thread {
                    let row = rng.gen_range(0..csr_clone.num_rows());
                    let col = rng.gen_range(0..csr_clone.num_rows());
                    
                    // Read operations should be safe
                    let _value = csr_clone.get_element(row, col);
                    let _row_data = csr_clone.get_row(row);
                }
                
                thread_id
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            let thread_id = handle.join().unwrap();
            println!("CSR concurrent access thread {} completed", thread_id);
        }
    }

    #[test]
    fn test_csr_performance_characteristics() {
        let sizes = vec![100, 500, 1000, 2000];
        let density = 0.05;
        
        for &size in &sizes {
            // Test construction performance
            let matrix = generate_random_adjacency_matrix(
                &mut DeterministicRng::new(CSR_TEST_SEED), size, density
            );
            
            let (csr, construction_time) = measure_execution_time(|| {
                CompressedSparseRow::from_adjacency_matrix(&matrix)
            });
            
            println!("CSR construction time for {} x {}: {:?}", size, size, construction_time);
            
            // Construction time should scale reasonably
            let time_per_element = construction_time.as_nanos() as f64 / (size * size) as f64;
            assert!(time_per_element < 1000.0, "CSR construction too slow: {} ns per element", time_per_element);
            
            // Test access performance
            let mut rng = DeterministicRng::new(CSR_TEST_SEED);
            let test_rows: Vec<usize> = (0..100).map(|_| rng.gen_range(0..size)).collect();
            
            let (_, access_time) = measure_execution_time(|| {
                for &row in &test_rows {
                    let _row_data = csr.get_row(row);
                }
            });
            
            println!("CSR row access time for 100 operations: {:?}", access_time);
            assert!(access_time.as_micros() < 10000, "CSR row access too slow");
            
            // Test matrix-vector multiplication performance
            let vector: Vec<f32> = (0..size).map(|i| i as f32).collect();
            
            let (_, multiply_time) = measure_execution_time(|| {
                let _result = csr.multiply_vector(&vector);
            });
            
            println!("CSR matrix-vector multiply time: {:?}", multiply_time);
            let operations = csr.num_nonzeros() as f64;
            let ops_per_second = operations / multiply_time.as_secs_f64();
            
            assert!(ops_per_second > 1_000_000.0, "CSR multiply too slow: {:.0} ops/sec", ops_per_second);
        }
    }

    #[test]
    fn test_csr_memory_efficiency() {
        let sizes = vec![100, 500, 1000];
        let densities = vec![0.01, 0.05, 0.1, 0.2];
        
        for &size in &sizes {
            for &density in &densities {
                let matrix = generate_random_adjacency_matrix(
                    &mut DeterministicRng::new(CSR_TEST_SEED), size, density
                );
                let csr = CompressedSparseRow::from_adjacency_matrix(&matrix);
                
                // Calculate compression ratio
                let dense_memory = size * size * std::mem::size_of::<f32>();
                let sparse_memory = csr.memory_usage();
                let compression_ratio = dense_memory as f64 / sparse_memory as f64;
                
                println!("Size: {}, Density: {:.2}, Compression: {:.2}x", 
                        size, density, compression_ratio);
                
                // For low densities, should achieve significant compression
                if density <= 0.1 {
                    assert!(compression_ratio > 5.0, 
                           "Insufficient compression: {:.2}x for density {}", 
                           compression_ratio, density);
                }
                
                // Verify memory usage calculation
                let calculated_memory = calculate_expected_csr_memory(&csr);
                let actual_memory = csr.memory_usage();
                let memory_ratio = actual_memory as f64 / calculated_memory as f64;
                
                assert!(memory_ratio > 0.9 && memory_ratio < 1.1,
                       "Memory calculation inaccurate: {} vs {} (ratio: {:.2})",
                       actual_memory, calculated_memory, memory_ratio);
            }
        }
    }
}

// Helper functions for CSR tests
fn generate_random_adjacency_matrix(rng: &mut DeterministicRng, size: usize, density: f64) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0f32; size]; size];
    
    for i in 0..size {
        for j in 0..size {
            if rng.gen::<f64>() < density {
                matrix[i][j] = rng.gen_range(0.1..1.0);
            }
        }
    }
    
    matrix
}

fn count_nonzeros(matrix: &[Vec<f32>]) -> usize {
    matrix.iter()
        .flat_map(|row| row.iter())
        .filter(|&&value| value != 0.0)
        .count()
}

fn assert_matrices_equal(m1: &[Vec<f32>], m2: &[Vec<f32>]) {
    assert_eq!(m1.len(), m2.len());
    
    for (i, (row1, row2)) in m1.iter().zip(m2.iter()).enumerate() {
        assert_eq!(row1.len(), row2.len());
        
        for (j, (&val1, &val2)) in row1.iter().zip(row2.iter()).enumerate() {
            assert!((val1 - val2).abs() < 1e-6,
                   "Matrix mismatch at ({}, {}): {} vs {}", i, j, val1, val2);
        }
    }
}

fn create_test_csr_matrix(size: usize, density: f64) -> CompressedSparseRow {
    let mut rng = DeterministicRng::new(CSR_TEST_SEED);
    let matrix = generate_random_adjacency_matrix(&mut rng, size, density);
    CompressedSparseRow::from_adjacency_matrix(&matrix)
}

fn create_test_csr_matrix_with_seed(size: usize, density: f64, seed: u64) -> CompressedSparseRow {
    let mut rng = DeterministicRng::new(seed);
    let matrix = generate_random_adjacency_matrix(&mut rng, size, density);
    CompressedSparseRow::from_adjacency_matrix(&matrix)
}

fn naive_matrix_vector_multiply(csr: &CompressedSparseRow, vector: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; csr.num_rows()];
    
    for row in 0..csr.num_rows() {
        for col in 0..csr.num_rows() {
            let value = csr.get_element(row, col);
            if value != 0.0 {
                result[row] += value * vector[col];
            }
        }
    }
    
    result
}

fn calculate_expected_csr_memory(csr: &CompressedSparseRow) -> usize {
    let offsets_size = (csr.num_rows() + 1) * std::mem::size_of::<usize>();
    let indices_size = csr.num_nonzeros() * std::mem::size_of::<usize>();
    let values_size = csr.num_nonzeros() * std::mem::size_of::<f32>();
    
    offsets_size + indices_size + values_size
}