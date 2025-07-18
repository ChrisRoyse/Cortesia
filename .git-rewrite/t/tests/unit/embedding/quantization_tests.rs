//! Vector Quantization Unit Tests
//!
//! Comprehensive tests for product quantization including accuracy,
//! training convergence, edge cases, and memory efficiency.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::embedding::quantizer::ProductQuantizer;
use crate::math::similarity::{euclidean_distance, euclidean_norm};

#[cfg(test)]
mod quantization_tests {
    use super::*;

    #[test]
    fn test_product_quantization_accuracy() {
        let mut rng = DeterministicRng::new(PQ_TEST_SEED);
        let dimension = 128;
        let codebook_size = 256;
        let vector_count = 1000;
        
        // Generate test vectors with known properties
        let test_vectors = generate_clustered_vectors(&mut rng, vector_count, dimension, 10);
        
        // Train product quantizer
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        quantizer.train(&test_vectors).unwrap();
        
        // Test 1: Quantization and reconstruction
        let mut total_error = 0.0;
        let mut max_error = 0.0;
        
        for vector in &test_vectors {
            let quantized = quantizer.quantize(vector);
            let reconstructed = quantizer.reconstruct(&quantized);
            
            assert_eq!(reconstructed.len(), vector.len());
            
            let error = euclidean_distance(vector, &reconstructed);
            total_error += error;
            max_error = max_error.max(error);
        }
        
        let average_error = total_error / vector_count as f32;
        
        // Verify compression quality
        assert!(average_error < EXPECTED_PQ_AVERAGE_ERROR,
               "Average quantization error too high: {} vs {}", 
               average_error, EXPECTED_PQ_AVERAGE_ERROR);
        
        assert!(max_error < EXPECTED_PQ_MAX_ERROR,
               "Maximum quantization error too high: {} vs {}", 
               max_error, EXPECTED_PQ_MAX_ERROR);
        
        // Test 2: Compression ratio
        let original_size = vector_count * dimension * std::mem::size_of::<f32>();
        let compressed_size = quantizer.compressed_size(vector_count);
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        assert!(compression_ratio >= EXPECTED_MIN_COMPRESSION_RATIO,
               "Compression ratio too low: {} vs {}", 
               compression_ratio, EXPECTED_MIN_COMPRESSION_RATIO);
        
        // Test 3: Deterministic behavior
        let quantized1 = quantizer.quantize(&test_vectors[0]);
        let quantized2 = quantizer.quantize(&test_vectors[0]);
        assert_eq!(quantized1, quantized2);
        
        let reconstructed1 = quantizer.reconstruct(&quantized1);
        let reconstructed2 = quantizer.reconstruct(&quantized1);
        assert_vectors_equal(&reconstructed1, &reconstructed2, 1e-10);
    }
    
    #[test]
    fn test_quantizer_training_convergence() {
        let mut rng = DeterministicRng::new(PQ_TRAINING_SEED);
        let dimension = 64;
        let codebook_size = 128;
        
        // Generate training data with clear cluster structure
        let training_vectors = generate_clear_clusters(&mut rng, 5000, dimension, 8);
        
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Train with convergence monitoring
        let training_result = quantizer.train_with_monitoring(&training_vectors);
        
        assert!(training_result.converged, "Quantizer training did not converge");
        assert!(training_result.final_loss < EXPECTED_FINAL_TRAINING_LOSS,
               "Training loss too high: {} vs {}", 
               training_result.final_loss, EXPECTED_FINAL_TRAINING_LOSS);
        
        // Verify training stability
        assert!(training_result.loss_variance < EXPECTED_LOSS_VARIANCE,
               "Training loss too unstable: variance {}", training_result.loss_variance);
        
        // Test codebook quality
        verify_codebook_quality(&quantizer, &training_vectors);
    }
    
    #[test]
    fn test_quantization_edge_cases() {
        let dimension = 32;
        let codebook_size = 64;
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Test 1: Zero vector
        let zero_vector = vec![0.0; dimension];
        let training_data = vec![zero_vector.clone(); 100];
        quantizer.train(&training_data).unwrap();
        
        let quantized = quantizer.quantize(&zero_vector);
        let reconstructed = quantizer.reconstruct(&quantized);
        
        for &value in &reconstructed {
            assert!(value.abs() < 1e-6, "Zero vector reconstruction error: {}", value);
        }
        
        // Test 2: Unit vectors
        for dim in 0..dimension {
            let mut unit_vector = vec![0.0; dimension];
            unit_vector[dim] = 1.0;
            
            let quantized = quantizer.quantize(&unit_vector);
            let reconstructed = quantizer.reconstruct(&quantized);
            
            let error = euclidean_distance(&unit_vector, &reconstructed);
            assert!(error < 1.0, "Unit vector quantization error too high: {}", error);
        }
        
        // Test 3: Very large values
        let large_vector: Vec<f32> = (0..dimension).map(|i| (i as f32) * 1000.0).collect();
        let quantized = quantizer.quantize(&large_vector);
        let reconstructed = quantizer.reconstruct(&quantized);
        
        // Should handle large values gracefully
        let relative_error = euclidean_distance(&large_vector, &reconstructed) / 
                           euclidean_norm(&large_vector);
        assert!(relative_error < 0.5, "Large value quantization relative error: {}", relative_error);
        
        // Test 4: NaN and infinity handling
        let mut nan_vector = vec![1.0; dimension];
        nan_vector[0] = f32::NAN;
        
        let result = quantizer.quantize(&nan_vector);
        // Should either handle gracefully or return error
        assert!(result.is_empty() || !result.iter().any(|&x| x > codebook_size as u8));
        
        let mut inf_vector = vec![1.0; dimension];
        inf_vector[0] = f32::INFINITY;
        
        let result = quantizer.quantize(&inf_vector);
        assert!(result.is_empty() || !result.iter().any(|&x| x > codebook_size as u8));
    }
    
    #[test]
    fn test_quantization_memory_efficiency() {
        let dimensions = vec![32, 64, 128, 256];
        let codebook_size = 256;
        let vector_count = 1000;
        
        for &dim in &dimensions {
            let mut quantizer = ProductQuantizer::new(dim, codebook_size);
            
            // Generate training data
            let mut rng = DeterministicRng::new(PQ_MEMORY_SEED + dim as u64);
            let training_data = generate_random_vectors(&mut rng, vector_count, dim);
            
            quantizer.train(&training_data).unwrap();
            
            // Calculate memory usage
            let quantizer_memory = quantizer.memory_usage();
            let vector_memory = vector_count * quantizer.compressed_vector_size();
            let total_compressed_memory = quantizer_memory + vector_memory;
            
            let original_memory = vector_count * dim * std::mem::size_of::<f32>();
            let compression_ratio = original_memory as f64 / total_compressed_memory as f64;
            
            println!("Dimension {}: Compression ratio {:.2}x", dim, compression_ratio);
            
            // Should achieve significant compression
            assert!(compression_ratio >= 10.0,
                   "Insufficient compression for dimension {}: {:.2}x", dim, compression_ratio);
            
            // Memory usage should scale predictably
            let expected_quantizer_memory = calculate_expected_quantizer_memory(dim, codebook_size);
            let memory_ratio = quantizer_memory as f64 / expected_quantizer_memory as f64;
            
            assert!(memory_ratio > 0.8 && memory_ratio < 1.2,
                   "Quantizer memory usage unexpected: {} vs {} (ratio: {:.2})", 
                   quantizer_memory, expected_quantizer_memory, memory_ratio);
        }
    }

    #[test]
    fn test_quantizer_subspace_division() {
        let dimension = 96; // Divisible by multiple subspace counts
        let codebook_size = 256;
        
        for num_subspaces in vec![2, 3, 4, 6, 8, 12] {
            if dimension % num_subspaces != 0 {
                continue; // Skip if dimension not divisible
            }
            
            let mut quantizer = ProductQuantizer::with_subspaces(dimension, codebook_size, num_subspaces);
            
            // Test subspace properties
            assert_eq!(quantizer.num_subspaces(), num_subspaces);
            assert_eq!(quantizer.subspace_dimension(), dimension / num_subspaces);
            
            // Generate test data
            let mut rng = DeterministicRng::new(PQ_TEST_SEED + num_subspaces as u64);
            let training_data = generate_random_vectors(&mut rng, 1000, dimension);
            
            // Train quantizer
            quantizer.train(&training_data).unwrap();
            
            // Test quantization consistency
            let test_vector = &training_data[0];
            let quantized = quantizer.quantize(test_vector);
            
            // Should have one code per subspace
            assert_eq!(quantized.len(), num_subspaces);
            
            // Each code should be within codebook range
            for &code in &quantized {
                assert!(code < codebook_size as u8, "Code {} exceeds codebook size", code);
            }
            
            // Test reconstruction
            let reconstructed = quantizer.reconstruct(&quantized);
            assert_eq!(reconstructed.len(), dimension);
            
            // Error should be reasonable
            let error = euclidean_distance(test_vector, &reconstructed);
            assert!(error < 5.0, "Reconstruction error too high for {} subspaces: {}", 
                   num_subspaces, error);
        }
    }

    #[test]
    fn test_quantizer_serialization() {
        let dimension = 64;
        let codebook_size = 128;
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Train quantizer
        let mut rng = DeterministicRng::new(PQ_TEST_SEED);
        let training_data = generate_random_vectors(&mut rng, 1000, dimension);
        quantizer.train(&training_data).unwrap();
        
        // Test binary serialization
        let binary_data = quantizer.to_binary().unwrap();
        let deserialized = ProductQuantizer::from_binary(&binary_data).unwrap();
        
        // Should produce identical results
        let test_vector = &training_data[0];
        let original_quantized = quantizer.quantize(test_vector);
        let deserialized_quantized = deserialized.quantize(test_vector);
        
        assert_eq!(original_quantized, deserialized_quantized);
        
        let original_reconstructed = quantizer.reconstruct(&original_quantized);
        let deserialized_reconstructed = deserialized.reconstruct(&deserialized_quantized);
        
        assert_vectors_equal(&original_reconstructed, &deserialized_reconstructed, 1e-6);
        
        // Test JSON serialization
        let json_data = quantizer.to_json().unwrap();
        let json_deserialized = ProductQuantizer::from_json(&json_data).unwrap();
        
        let json_quantized = json_deserialized.quantize(test_vector);
        assert_eq!(original_quantized, json_quantized);
        
        // Binary should be more compact
        assert!(binary_data.len() < json_data.len());
    }

    #[test]
    fn test_quantizer_performance() {
        let dimension = 128;
        let codebook_size = 256;
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Generate test data
        let mut rng = DeterministicRng::new(PQ_TEST_SEED);
        let training_data = generate_random_vectors(&mut rng, 5000, dimension);
        let test_data = generate_random_vectors(&mut rng, 10000, dimension);
        
        // Test training performance
        let (_, training_time) = measure_execution_time(|| {
            quantizer.train(&training_data).unwrap();
        });
        
        println!("PQ training time for {} vectors: {:?}", training_data.len(), training_time);
        assert!(training_time.as_secs() < 30, "Training too slow: {:?}", training_time);
        
        // Test quantization performance
        let (quantized_vectors, quantization_time) = measure_execution_time(|| {
            test_data.iter().map(|v| quantizer.quantize(v)).collect::<Vec<_>>()
        });
        
        println!("PQ quantization time for {} vectors: {:?}", test_data.len(), quantization_time);
        let quantizations_per_second = test_data.len() as f64 / quantization_time.as_secs_f64();
        assert!(quantizations_per_second > 1000.0, "Quantization too slow: {:.0} ops/sec", 
               quantizations_per_second);
        
        // Test reconstruction performance
        let (_, reconstruction_time) = measure_execution_time(|| {
            for quantized in &quantized_vectors {
                let _ = quantizer.reconstruct(quantized);
            }
        });
        
        println!("PQ reconstruction time for {} vectors: {:?}", quantized_vectors.len(), reconstruction_time);
        let reconstructions_per_second = quantized_vectors.len() as f64 / reconstruction_time.as_secs_f64();
        assert!(reconstructions_per_second > 5000.0, "Reconstruction too slow: {:.0} ops/sec", 
               reconstructions_per_second);
    }

    #[test]
    fn test_quantizer_batch_operations() {
        let dimension = 64;
        let codebook_size = 128;
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Generate test data
        let mut rng = DeterministicRng::new(PQ_TEST_SEED);
        let training_data = generate_random_vectors(&mut rng, 1000, dimension);
        let test_vectors = generate_random_vectors(&mut rng, 5000, dimension);
        
        quantizer.train(&training_data).unwrap();
        
        // Test batch quantization
        let (batch_quantized, batch_time) = measure_execution_time(|| {
            quantizer.quantize_batch(&test_vectors)
        });
        
        // Test individual quantization for comparison
        let (individual_quantized, individual_time) = measure_execution_time(|| {
            test_vectors.iter().map(|v| quantizer.quantize(v)).collect::<Vec<_>>()
        });
        
        // Batch should be faster
        let speedup = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
        assert!(speedup > 1.5, "Batch quantization not faster: {:.2}x speedup", speedup);
        
        // Results should be identical
        assert_eq!(batch_quantized.len(), individual_quantized.len());
        for (batch_result, individual_result) in batch_quantized.iter().zip(individual_quantized.iter()) {
            assert_eq!(batch_result, individual_result);
        }
        
        // Test batch reconstruction
        let (batch_reconstructed, batch_recon_time) = measure_execution_time(|| {
            quantizer.reconstruct_batch(&batch_quantized)
        });
        
        let (individual_reconstructed, individual_recon_time) = measure_execution_time(|| {
            batch_quantized.iter().map(|q| quantizer.reconstruct(q)).collect::<Vec<_>>()
        });
        
        // Batch reconstruction should also be faster
        let recon_speedup = individual_recon_time.as_nanos() as f64 / batch_recon_time.as_nanos() as f64;
        assert!(recon_speedup > 1.5, "Batch reconstruction not faster: {:.2}x speedup", recon_speedup);
        
        // Results should be very close (allowing for floating point differences)
        assert_eq!(batch_reconstructed.len(), individual_reconstructed.len());
        for (batch_vec, individual_vec) in batch_reconstructed.iter().zip(individual_reconstructed.iter()) {
            assert_vectors_equal(batch_vec, individual_vec, 1e-6);
        }
    }
}

// Helper functions for quantization tests
fn generate_clustered_vectors(rng: &mut DeterministicRng, count: usize, dimension: usize, clusters: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();
    
    // Generate cluster centers
    let cluster_centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    // Generate vectors around cluster centers
    for _ in 0..count {
        let cluster_idx = rng.gen_range(0..clusters);
        let center = &cluster_centers[cluster_idx];
        
        let vector: Vec<f32> = center.iter()
            .map(|&c| c + rng.gen_range(-0.1..0.1)) // Small noise around center
            .collect();
        
        vectors.push(vector);
    }
    
    vectors
}

fn generate_clear_clusters(rng: &mut DeterministicRng, count: usize, dimension: usize, clusters: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();
    
    // Generate well-separated cluster centers
    let cluster_centers: Vec<Vec<f32>> = (0..clusters)
        .map(|i| {
            let base_value = (i as f32) * 3.0; // Well separated
            (0..dimension).map(|j| {
                if j % clusters == i {
                    base_value + rng.gen_range(-0.5..0.5)
                } else {
                    rng.gen_range(-0.5..0.5)
                }
            }).collect()
        })
        .collect();
    
    // Generate vectors with stronger clustering
    for _ in 0..count {
        let cluster_idx = rng.gen_range(0..clusters);
        let center = &cluster_centers[cluster_idx];
        
        let vector: Vec<f32> = center.iter()
            .map(|&c| c + rng.gen_range(-0.3..0.3))
            .collect();
        
        vectors.push(vector);
    }
    
    vectors
}

fn generate_random_vectors(rng: &mut DeterministicRng, count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn verify_codebook_quality(quantizer: &ProductQuantizer, training_vectors: &[Vec<f32>]) {
    // Test that codebook entries are well-distributed
    for subspace in 0..quantizer.num_subspaces() {
        let codebook = quantizer.get_codebook(subspace);
        
        // Each codebook entry should be different
        for i in 0..codebook.len() {
            for j in (i+1)..codebook.len() {
                let distance = euclidean_distance(&codebook[i], &codebook[j]);
                assert!(distance > 1e-6, "Codebook entries too similar in subspace {}: {} vs {}", 
                       subspace, i, j);
            }
        }
        
        // Codebook should span reasonable range
        let min_vals: Vec<f32> = (0..codebook[0].len())
            .map(|dim| codebook.iter().map(|entry| entry[dim]).fold(f32::INFINITY, f32::min))
            .collect();
        
        let max_vals: Vec<f32> = (0..codebook[0].len())
            .map(|dim| codebook.iter().map(|entry| entry[dim]).fold(f32::NEG_INFINITY, f32::max))
            .collect();
        
        for (min_val, max_val) in min_vals.iter().zip(max_vals.iter()) {
            let range = max_val - min_val;
            assert!(range > 0.1, "Codebook range too small in subspace {}: {}", subspace, range);
        }
    }
}

fn calculate_expected_quantizer_memory(dimension: usize, codebook_size: usize) -> usize {
    let num_subspaces = 8; // Default subspace count
    let subspace_dim = dimension / num_subspaces;
    
    // Memory for codebooks
    let codebook_memory = num_subspaces * codebook_size * subspace_dim * std::mem::size_of::<f32>();
    
    // Memory for metadata
    let metadata_memory = 1024; // Estimated overhead
    
    codebook_memory + metadata_memory
}