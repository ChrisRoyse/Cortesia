//! SIMD Operations Unit Tests
//!
//! Comprehensive tests for SIMD-accelerated vector operations including
//! distance computation, batch operations, alignment requirements, and
//! performance validation.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::embedding::simd_search::*;

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_distance_computation() {
        let dimensions = vec![32, 64, 128, 256, 512, 1024];
        
        for &dim in &dimensions {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED + dim as u64);
            
            // Generate test vectors
            let vector1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vector2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            // Compute distance using SIMD and scalar implementations
            let simd_distance = simd_euclidean_distance(&vector1, &vector2);
            let scalar_distance = scalar_euclidean_distance(&vector1, &vector2);
            
            // Results should be nearly identical
            let difference = (simd_distance - scalar_distance).abs();
            assert!(difference < 1e-5, 
                   "SIMD vs scalar distance mismatch for dim {}: {} vs {} (diff: {})", 
                   dim, simd_distance, scalar_distance, difference);
            
            // Test performance improvement
            let simd_time = measure_distance_computation_time(&vector1, &vector2, true);
            let scalar_time = measure_distance_computation_time(&vector1, &vector2, false);
            
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("SIMD speedup for dimension {}: {:.2}x", dim, speedup);
            
            // Should see significant speedup for larger dimensions
            if dim >= 128 {
                assert!(speedup > 2.0, "Insufficient SIMD speedup for dimension {}: {:.2}x", dim, speedup);
            }
        }
    }
    
    #[test]
    fn test_simd_batch_operations() {
        let dimension = 128;
        let batch_size = 1000;
        let query_count = 100;
        
        let mut rng = DeterministicRng::new(SIMD_BATCH_SEED);
        
        // Generate database vectors
        let database: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Generate query vectors
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Test batch distance computation
        for query in &queries {
            let simd_distances = simd_batch_distances(query, &database);
            let scalar_distances: Vec<f32> = database.iter()
                .map(|db_vec| scalar_euclidean_distance(query, db_vec))
                .collect();
            
            assert_eq!(simd_distances.len(), scalar_distances.len());
            
            for (i, (&simd_dist, &scalar_dist)) in simd_distances.iter().zip(scalar_distances.iter()).enumerate() {
                let difference = (simd_dist - scalar_dist).abs();
                assert!(difference < 1e-4, 
                       "Batch distance mismatch at index {}: {} vs {}", 
                       i, simd_dist, scalar_dist);
            }
        }
        
        // Test performance
        let simd_time = measure_batch_distance_time(&queries[0], &database, true);
        let scalar_time = measure_batch_distance_time(&queries[0], &database, false);
        
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        assert!(speedup > 3.0, "Insufficient batch SIMD speedup: {:.2}x", speedup);
    }
    
    #[test]
    fn test_simd_alignment_requirements() {
        // Test various alignment scenarios
        let dimension = 256;
        let mut rng = DeterministicRng::new(SIMD_ALIGNMENT_SEED);
        
        // Test aligned vectors
        let aligned_vec1 = create_aligned_vector(dimension, &mut rng);
        let aligned_vec2 = create_aligned_vector(dimension, &mut rng);
        
        let aligned_distance = simd_euclidean_distance(&aligned_vec1, &aligned_vec2);
        
        // Test unaligned vectors (should still work correctly)
        let unaligned_vec1 = create_unaligned_vector(dimension, &mut rng);
        let unaligned_vec2 = create_unaligned_vector(dimension, &mut rng);
        
        let unaligned_distance = simd_euclidean_distance(&unaligned_vec1, &unaligned_vec2);
        
        // Results should be consistent regardless of alignment
        let scalar_distance1 = scalar_euclidean_distance(&aligned_vec1, &aligned_vec2);
        let scalar_distance2 = scalar_euclidean_distance(&unaligned_vec1, &unaligned_vec2);
        
        assert!((aligned_distance - scalar_distance1).abs() < 1e-5);
        assert!((unaligned_distance - scalar_distance2).abs() < 1e-5);
        
        // Test mixed alignment
        let mixed_distance = simd_euclidean_distance(&aligned_vec1, &unaligned_vec2);
        let scalar_mixed = scalar_euclidean_distance(&aligned_vec1, &unaligned_vec2);
        
        assert!((mixed_distance - scalar_mixed).abs() < 1e-5);
    }
    
    #[test]
    fn test_simd_vector_operations() {
        let dimension = 128;
        let mut rng = DeterministicRng::new(SIMD_VECTOR_OPS_SEED);
        
        let vec1: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let vec2: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let scalar = 2.5f32;
        
        // Test vector addition
        let simd_add = simd_vector_add(&vec1, &vec2);
        let scalar_add = scalar_vector_add(&vec1, &vec2);
        verify_vector_equality(&simd_add, &scalar_add, 1e-6);
        
        // Test vector subtraction
        let simd_sub = simd_vector_subtract(&vec1, &vec2);
        let scalar_sub = scalar_vector_subtract(&vec1, &vec2);
        verify_vector_equality(&simd_sub, &scalar_sub, 1e-6);
        
        // Test scalar multiplication
        let simd_mul = simd_scalar_multiply(&vec1, scalar);
        let scalar_mul = scalar_scalar_multiply(&vec1, scalar);
        verify_vector_equality(&simd_mul, &scalar_mul, 1e-6);
        
        // Test dot product
        let simd_dot = simd_dot_product(&vec1, &vec2);
        let scalar_dot = scalar_dot_product(&vec1, &vec2);
        assert!((simd_dot - scalar_dot).abs() < 1e-5, 
               "Dot product mismatch: {} vs {}", simd_dot, scalar_dot);
        
        // Test vector normalization
        let simd_norm = simd_normalize(&vec1);
        let scalar_norm = scalar_normalize(&vec1);
        verify_vector_equality(&simd_norm, &scalar_norm, 1e-6);
        
        // Verify normalized vector has unit length
        let norm_length = simd_dot_product(&simd_norm, &simd_norm).sqrt();
        assert!((norm_length - 1.0).abs() < 1e-6, "Normalized vector length: {}", norm_length);
    }

    #[test]
    fn test_simd_cosine_similarity() {
        let dimension = 256;
        let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
        
        // Test various vector pairs
        let test_cases = vec![
            // Identical vectors (similarity = 1.0)
            (vec![1.0; dimension], vec![1.0; dimension]),
            // Opposite vectors (similarity = -1.0)
            (vec![1.0; dimension], vec![-1.0; dimension]),
            // Orthogonal vectors (similarity ≈ 0.0)
            ({
                let mut v1 = vec![0.0; dimension];
                v1[0] = 1.0;
                v1
            }, {
                let mut v2 = vec![0.0; dimension];
                v2[1] = 1.0;
                v2
            }),
        ];
        
        for (vec1, vec2) in test_cases {
            let simd_similarity = simd_cosine_similarity(&vec1, &vec2);
            let scalar_similarity = scalar_cosine_similarity(&vec1, &vec2);
            
            assert!((simd_similarity - scalar_similarity).abs() < 1e-6,
                   "Cosine similarity mismatch: {} vs {}", simd_similarity, scalar_similarity);
        }
        
        // Test random vectors
        for _ in 0..100 {
            let vec1: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vec2: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            let simd_similarity = simd_cosine_similarity(&vec1, &vec2);
            let scalar_similarity = scalar_cosine_similarity(&vec1, &vec2);
            
            assert!((simd_similarity - scalar_similarity).abs() < 1e-5,
                   "Random vector cosine similarity mismatch: {} vs {}", 
                   simd_similarity, scalar_similarity);
            
            // Cosine similarity should be in [-1, 1]
            assert!(simd_similarity >= -1.0 && simd_similarity <= 1.0,
                   "Cosine similarity out of range: {}", simd_similarity);
        }
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let dimension = 128;
        let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
        
        for _ in 0..100 {
            let vec1: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
            let vec2: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
            
            let simd_distance = simd_manhattan_distance(&vec1, &vec2);
            let scalar_distance = scalar_manhattan_distance(&vec1, &vec2);
            
            assert!((simd_distance - scalar_distance).abs() < 1e-5,
                   "Manhattan distance mismatch: {} vs {}", simd_distance, scalar_distance);
            
            // Manhattan distance should be non-negative
            assert!(simd_distance >= 0.0, "Manhattan distance negative: {}", simd_distance);
        }
    }

    #[test]
    fn test_simd_performance_scaling() {
        let dimensions = vec![64, 128, 256, 512, 1024, 2048];
        
        for &dim in &dimensions {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED + dim as u64);
            
            // Generate test vectors
            let vec1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vec2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            // Test various operations
            let operations = vec![
                ("euclidean_distance", Box::new(|v1: &[f32], v2: &[f32]| {
                    simd_euclidean_distance(v1, v2)
                }) as Box<dyn Fn(&[f32], &[f32]) -> f32>),
                ("dot_product", Box::new(|v1: &[f32], v2: &[f32]| {
                    simd_dot_product(v1, v2)
                })),
                ("cosine_similarity", Box::new(|v1: &[f32], v2: &[f32]| {
                    simd_cosine_similarity(v1, v2)
                })),
            ];
            
            for (op_name, op_fn) in operations {
                let (_, operation_time) = measure_execution_time(|| {
                    for _ in 0..1000 {
                        let _ = op_fn(&vec1, &vec2);
                    }
                });
                
                let ops_per_second = 1000.0 / operation_time.as_secs_f64();
                println!("{} for dim {}: {:.0} ops/sec", op_name, dim, ops_per_second);
                
                // Should maintain reasonable performance even for large dimensions
                assert!(ops_per_second > 10000.0, 
                       "{} too slow for dimension {}: {:.0} ops/sec", 
                       op_name, dim, ops_per_second);
            }
        }
    }

    #[test]
    fn test_simd_edge_cases() {
        // Test zero vectors
        let zero_vec = vec![0.0; 128];
        let normal_vec: Vec<f32> = (0..128).map(|i| i as f32).collect();
        
        let distance = simd_euclidean_distance(&zero_vec, &normal_vec);
        let expected_distance = scalar_euclidean_distance(&zero_vec, &normal_vec);
        assert!((distance - expected_distance).abs() < 1e-5);
        
        // Test identical vectors
        let identical_distance = simd_euclidean_distance(&normal_vec, &normal_vec);
        assert!(identical_distance < 1e-6, "Identical vectors should have zero distance");
        
        // Test very small vectors
        let tiny_vec: Vec<f32> = (0..128).map(|_| 1e-10).collect();
        let tiny_distance = simd_euclidean_distance(&tiny_vec, &zero_vec);
        assert!(tiny_distance > 0.0 && tiny_distance < 1e-8);
        
        // Test very large vectors
        let large_vec: Vec<f32> = (0..128).map(|_| 1e6).collect();
        let large_distance = simd_euclidean_distance(&large_vec, &zero_vec);
        assert!(large_distance > 1e5);
        
        // Test mixed positive/negative
        let mixed_vec: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mixed_dot = simd_dot_product(&mixed_vec, &normal_vec);
        let expected_dot = scalar_dot_product(&mixed_vec, &normal_vec);
        assert!((mixed_dot - expected_dot).abs() < 1e-5);
    }

    #[test]
    fn test_simd_memory_access_patterns() {
        let dimension = 1024;
        let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
        
        // Test various memory layouts
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Test sequential access
        let (_, sequential_time) = measure_execution_time(|| {
            for i in 0..vectors.len()-1 {
                let _ = simd_euclidean_distance(&vectors[i], &vectors[i+1]);
            }
        });
        
        // Test random access
        let random_indices: Vec<usize> = (0..vectors.len()-1)
            .map(|_| rng.gen_range(0..vectors.len()-1))
            .collect();
        
        let (_, random_time) = measure_execution_time(|| {
            for &i in &random_indices {
                let j = (i + 1) % vectors.len();
                let _ = simd_euclidean_distance(&vectors[i], &vectors[j]);
            }
        });
        
        println!("Sequential access time: {:?}", sequential_time);
        println!("Random access time: {:?}", random_time);
        
        // Random access might be slower due to cache misses, but should still be reasonable
        let slowdown_ratio = random_time.as_nanos() as f64 / sequential_time.as_nanos() as f64;
        assert!(slowdown_ratio < 3.0, "Random access too much slower: {:.2}x", slowdown_ratio);
    }

    #[test]
    fn test_simd_numerical_stability() {
        let dimension = 256;
        
        // Test with values near machine epsilon
        let epsilon_vec: Vec<f32> = (0..dimension).map(|_| f32::EPSILON).collect();
        let zero_vec = vec![0.0; dimension];
        
        let epsilon_distance = simd_euclidean_distance(&epsilon_vec, &zero_vec);
        assert!(epsilon_distance > 0.0 && epsilon_distance.is_finite());
        
        // Test with very small differences
        let base_vec: Vec<f32> = (0..dimension).map(|i| i as f32).collect();
        let slightly_different: Vec<f32> = base_vec.iter()
            .map(|&x| x + f32::EPSILON * 10.0)
            .collect();
        
        let small_diff_distance = simd_euclidean_distance(&base_vec, &slightly_different);
        assert!(small_diff_distance > 0.0 && small_diff_distance < 1e-5);
        
        // Test normalization stability
        let very_small_vec: Vec<f32> = (0..dimension).map(|_| 1e-20).collect();
        let normalized = simd_normalize(&very_small_vec);
        
        // Should either normalize properly or handle gracefully
        let norm = simd_dot_product(&normalized, &normalized).sqrt();
        assert!(norm.is_finite() && (norm < 1e-10 || (norm - 1.0).abs() < 1e-6));
    }
}

// Helper functions for SIMD tests
fn measure_distance_computation_time(vec1: &[f32], vec2: &[f32], use_simd: bool) -> std::time::Duration {
    let iterations = 10000;
    
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        if use_simd {
            let _ = simd_euclidean_distance(vec1, vec2);
        } else {
            let _ = scalar_euclidean_distance(vec1, vec2);
        }
    }
    start.elapsed()
}

fn measure_batch_distance_time(query: &[f32], database: &[Vec<f32>], use_simd: bool) -> std::time::Duration {
    let start = std::time::Instant::now();
    
    if use_simd {
        let _ = simd_batch_distances(query, database);
    } else {
        let _: Vec<f32> = database.iter()
            .map(|db_vec| scalar_euclidean_distance(query, db_vec))
            .collect();
    }
    
    start.elapsed()
}

fn create_aligned_vector(size: usize, rng: &mut DeterministicRng) -> Vec<f32> {
    // Create vector that should be naturally aligned
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn create_unaligned_vector(size: usize, rng: &mut DeterministicRng) -> Vec<f32> {
    // Create vector with potential alignment issues
    let mut vec = Vec::with_capacity(size + 1);
    vec.push(0.0); // Offset by one element
    
    for _ in 0..size {
        vec.push(rng.gen_range(-1.0..1.0));
    }
    
    vec[1..].to_vec() // Return slice that's offset
}

fn verify_vector_equality(v1: &[f32], v2: &[f32], tolerance: f32) {
    assert_eq!(v1.len(), v2.len());
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!((a - b).abs() < tolerance,
               "Vector mismatch at index {}: {} vs {} (tolerance: {})", i, a, b, tolerance);
    }
}

// Scalar implementations for comparison
fn scalar_euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn scalar_vector_add(v1: &[f32], v2: &[f32]) -> Vec<f32> {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| a + b).collect()
}

fn scalar_vector_subtract(v1: &[f32], v2: &[f32]) -> Vec<f32> {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| a - b).collect()
}

fn scalar_scalar_multiply(v: &[f32], scalar: f32) -> Vec<f32> {
    v.iter().map(|&x| x * scalar).collect()
}

fn scalar_dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum()
}

fn scalar_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|&x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn scalar_cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot = scalar_dot_product(v1, v2);
    let norm1 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm2 = v2.iter().map(|&x| x * x).sum::<f32>().sqrt();
    
    if norm1 > 0.0 && norm2 > 0.0 {
        dot / (norm1 * norm2)
    } else {
        0.0
    }
}

fn scalar_manhattan_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b).abs()).sum()
}