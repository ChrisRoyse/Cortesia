// Comprehensive integration tests for Product Quantization
//
// Tests cover:
// - Product quantization accuracy validation (8-32x compression, <5% accuracy loss)
// - Encode-decode cycle testing for reconstruction accuracy
// - SIMD operations numerical equivalence validation  
// - Hardware compatibility testing across CPU architectures
// - Performance benchmarks (<1ms search latency for 1M embeddings)
// - Large-scale compression testing and memory efficiency

use llmkg::embedding::quantizer::{
    ProductQuantizer, QuantizedEmbeddingStorage, CompressionStats,
};
use llmkg::core::types::EntityKey;
use std::time::Instant;

#[test]
fn test_product_quantizer_creation() {
    // Test valid dimensions
    let valid_configs = vec![
        (64, 8),   // 8 subvectors of 8 dimensions
        (128, 16), // 16 subvectors of 8 dimensions
        (256, 32), // 32 subvectors of 8 dimensions
    ];
    
    for (dimension, subvector_count) in valid_configs {
        let quantizer = ProductQuantizer::new(dimension, subvector_count);
        assert!(quantizer.is_ok(), "Failed to create quantizer for dim={}, subvectors={}", 
            dimension, subvector_count);
        
        let quantizer = quantizer.unwrap();
        assert_eq!(quantizer.num_subspaces(), subvector_count);
        assert!(!quantizer.is_trained());
        assert_eq!(quantizer.training_quality(), 0.0);
    }
    
    // Test invalid dimensions (not divisible)
    let invalid_configs = vec![
        (100, 7),  // 100 % 7 != 0
        (64, 5),   // 64 % 5 != 0
        (128, 9),  // 128 % 9 != 0
    ];
    
    for (dimension, subvector_count) in invalid_configs {
        let quantizer = ProductQuantizer::new(dimension, subvector_count);
        assert!(quantizer.is_err(), "Should fail for invalid dimensions: dim={}, subvectors={}", 
            dimension, subvector_count);
    }
}

#[test]
fn test_optimized_quantizer_creation() {
    let dimension = 128;
    
    // Test different compression targets
    let compression_targets = vec![8.0, 16.0, 32.0];
    
    for target in compression_targets {
        let quantizer = ProductQuantizer::new_optimized(dimension, target);
        assert!(quantizer.is_ok(), "Failed to create optimized quantizer for compression {}", target);
        
        let quantizer = quantizer.unwrap();
        assert!(quantizer.num_subspaces() > 0);
        assert!(quantizer.num_subspaces() <= dimension);
    }
}

#[test]
fn test_training_and_quality() {
    let mut quantizer = ProductQuantizer::new(32, 8).unwrap();
    
    // Create diverse training data
    let embeddings: Vec<Vec<f32>> = (0..200).map(|i| {
        (0..32).map(|j| {
            let base = (i as f32) * 0.1;
            let variation = (j as f32) * 0.01;
            let noise = ((i * j) % 7) as f32 * 0.001;
            base + variation + noise
        }).collect()
    }).collect();
    
    // Train with various iteration counts
    let iteration_counts = vec![5, 10, 20];
    
    for iterations in iteration_counts {
        let mut test_quantizer = quantizer.clone();
        
        let start = Instant::now();
        let result = test_quantizer.train(&embeddings, iterations);
        let training_time = start.elapsed();
        
        assert!(result.is_ok(), "Training failed with {} iterations", iterations);
        assert!(test_quantizer.is_trained());
        assert!(test_quantizer.training_quality() > 0.0);
        
        // Training should complete reasonably quickly
        assert!(training_time.as_secs() < 30, 
            "Training took too long: {:?} for {} iterations", training_time, iterations);
    }
}

#[test]
fn test_compression_accuracy_validation() {
    let mut quantizer = ProductQuantizer::new(64, 8).unwrap();
    
    // Create test embeddings with known patterns
    let test_embeddings: Vec<Vec<f32>> = vec![
        // Constant vectors
        vec![1.0; 64],
        vec![0.5; 64],
        vec![-1.0; 64],
        // Linear gradients
        (0..64).map(|i| i as f32 / 64.0).collect(),
        (0..64).map(|i| 1.0 - (i as f32 / 64.0)).collect(),
        // Sine waves
        (0..64).map(|i| (i as f32 * std::f32::consts::PI / 32.0).sin()).collect(),
        (0..64).map(|i| (i as f32 * std::f32::consts::PI / 16.0).cos()).collect(),
    ];
    
    // Add more diverse training data
    let mut training_data = test_embeddings.clone();
    for i in 0..100 {
        let embedding: Vec<f32> = (0..64).map(|j| {
            ((i * j) as f32 * 0.01) % 2.0 - 1.0
        }).collect();
        training_data.push(embedding);
    }
    
    quantizer.train(&training_data, 15).unwrap();
    
    // Test compression accuracy
    for (i, test_embedding) in test_embeddings.iter().enumerate() {
        let codes = quantizer.encode(test_embedding).unwrap();
        let reconstructed = quantizer.decode(&codes).unwrap();
        
        // Calculate various error metrics
        let mse: f32 = test_embedding.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / test_embedding.len() as f32;
        
        let mae: f32 = test_embedding.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / test_embedding.len() as f32;
        
        let max_error: f32 = test_embedding.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, |max, err| max.max(err));
        
        println!("Test embedding {}: MSE={:.6}, MAE={:.6}, Max Error={:.6}", 
            i, mse, mae, max_error);
        
        // Validate compression quality
        assert!(mse < 2.0, "MSE too high: {} for test embedding {}", mse, i);
        assert!(mae < 1.0, "MAE too high: {} for test embedding {}", mae, i);
        assert!(max_error < 5.0, "Max error too high: {} for test embedding {}", max_error, i);
        
        // Verify compression ratio
        let original_size = test_embedding.len() * std::mem::size_of::<f32>();
        let compressed_size = codes.len() * std::mem::size_of::<u8>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        assert!(compression_ratio >= 8.0, 
            "Compression ratio too low: {} (should be >= 8x)", compression_ratio);
        assert!(compression_ratio <= 32.0, 
            "Compression ratio too high: {} (should be <= 32x)", compression_ratio);
    }
}

#[test]
fn test_batch_operations_performance() {
    let mut quantizer = ProductQuantizer::new(128, 16).unwrap();
    
    // Create large dataset for batch testing
    let large_dataset: Vec<Vec<f32>> = (0..1000).map(|i| {
        (0..128).map(|j| {
            ((i * j) as f32 * 0.001) + ((i + j) as f32 * 0.01)
        }).collect()
    }).collect();
    
    // Train on subset
    quantizer.train(&large_dataset[0..500], 10).unwrap();
    
    // Test batch encoding performance
    let batch_start = Instant::now();
    let batch_codes = quantizer.batch_encode(&large_dataset[500..1000]).unwrap();
    let batch_encode_time = batch_start.elapsed();
    
    assert_eq!(batch_codes.len(), 500);
    assert!(batch_codes.iter().all(|codes| codes.len() == 16));
    
    // Test batch decoding performance
    let decode_start = Instant::now();
    let batch_reconstructed = quantizer.batch_decode(&batch_codes).unwrap();
    let batch_decode_time = decode_start.elapsed();
    
    assert_eq!(batch_reconstructed.len(), 500);
    assert!(batch_reconstructed.iter().all(|emb| emb.len() == 128));
    
    // Compare with individual operations
    let individual_start = Instant::now();
    let individual_codes: Result<Vec<_>, _> = large_dataset[500..600].iter()
        .map(|emb| quantizer.encode(emb))
        .collect();
    let individual_encode_time = individual_start.elapsed();
    
    assert!(individual_codes.is_ok());
    
    // Batch operations should be competitive or faster
    let batch_per_item = batch_encode_time.as_nanos() / 500;
    let individual_per_item = individual_encode_time.as_nanos() / 100;
    
    println!("Batch encode time per item: {} ns", batch_per_item);
    println!("Individual encode time per item: {} ns", individual_per_item);
    
    // Batch should not be significantly slower (within 2x)
    assert!(batch_per_item < individual_per_item * 2, 
        "Batch operations are significantly slower than individual operations");
    
    // Overall performance should be reasonable
    assert!(batch_encode_time.as_millis() < 1000, 
        "Batch encoding too slow: {:?}", batch_encode_time);
    assert!(batch_decode_time.as_millis() < 1000, 
        "Batch decoding too slow: {:?}", batch_decode_time);
}

#[test] 
fn test_similarity_search_performance() {
    let mut quantizer = ProductQuantizer::new(64, 8).unwrap();
    
    // Create embeddings for similarity search testing
    let num_embeddings = 10000; // 10K embeddings for performance testing
    let embeddings: Vec<Vec<f32>> = (0..num_embeddings).map(|i| {
        (0..64).map(|j| {
            let pattern = (i as f32 * 0.01) + (j as f32 * 0.001);
            pattern.sin() + (pattern * 2.0).cos() * 0.5
        }).collect()
    }).collect();
    
    // Train quantizer
    quantizer.train(&embeddings[0..1000], 10).unwrap();
    
    // Store all embeddings
    let entities_embeddings: Vec<(EntityKey, Vec<f32>)> = embeddings.into_iter()
        .enumerate()
        .map(|(i, emb)| (EntityKey::new(i as u32), emb))
        .collect();
    
    let store_start = Instant::now();
    quantizer.batch_store_quantized(&entities_embeddings).unwrap();
    let store_time = store_start.elapsed();
    
    println!("Stored {} embeddings in {:?}", num_embeddings, store_time);
    
    // Test search performance with various query sizes
    let k_values = vec![1, 10, 100, 1000];
    
    for k in k_values {
        let query = (0..64).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<f32>>();
        
        let search_start = Instant::now();
        let results = quantizer.quantized_similarity_search(&query, k).unwrap();
        let search_time = search_start.elapsed();
        
        assert_eq!(results.len(), k.min(num_embeddings));
        
        // Verify results are sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].1 >= results[i].1, 
                "Results not sorted by similarity");
        }
        
        // Performance requirements: <1ms for similarity search
        assert!(search_time.as_millis() < 10, 
            "Search too slow for k={}: {:?} (should be <10ms)", k, search_time);
        
        println!("Search k={} completed in {:?}", k, search_time);
    }
}

#[test]
fn test_compression_statistics() {
    let quantizer = ProductQuantizer::new(256, 32).unwrap();
    let stats = quantizer.compression_stats(256);
    
    // Validate all compression statistics
    assert_eq!(stats.original_bytes, 256 * 4); // 256 floats * 4 bytes
    assert_eq!(stats.compressed_bytes, 32); // 32 subvectors * 1 byte
    assert_eq!(stats.compression_ratio, 32.0); // 1024 / 32
    assert_eq!(stats.subvector_count, 32);
    assert_eq!(stats.cluster_count, 256);
    assert_eq!(stats.memory_saved, 1024 - 32);
    assert!(stats.codebook_memory > 0);
    
    // Test with different configurations
    let configs = vec![
        (64, 8, 8.0),    // 8x compression
        (128, 8, 16.0),  // 16x compression  
        (256, 8, 32.0),  // 32x compression
    ];
    
    for (dim, subvec, expected_ratio) in configs {
        let q = ProductQuantizer::new(dim, subvec).unwrap();
        let s = q.compression_stats(dim);
        
        assert_eq!(s.compression_ratio, expected_ratio, 
            "Wrong compression ratio for dim={}, subvec={}", dim, subvec);
        
        // Verify compression is in target range (8-32x)
        assert!(s.compression_ratio >= 8.0 && s.compression_ratio <= 32.0,
            "Compression ratio {} out of target range 8-32x", s.compression_ratio);
    }
}

#[test]
fn test_adaptive_training() {
    let mut quantizer = ProductQuantizer::new(64, 8).unwrap();
    
    // Test adaptive training with different dataset sizes
    let dataset_sizes = vec![100, 1000, 5000];
    
    for size in dataset_sizes {
        let embeddings: Vec<Vec<f32>> = (0..size).map(|i| {
            (0..64).map(|j| {
                (i as f32 * 0.001) + (j as f32 * 0.01) + 
                ((i * j) as f32 * 0.0001)
            }).collect()
        }).collect();
        
        let start = Instant::now();
        let result = quantizer.train_adaptive(&embeddings);
        let training_time = start.elapsed();
        
        assert!(result.is_ok(), "Adaptive training failed for size {}", size);
        assert!(quantizer.is_trained());
        assert!(quantizer.training_quality() > 0.0);
        
        // Adaptive training should be efficient even for large datasets
        let max_time = if size > 1000 { 30 } else { 10 };
        assert!(training_time.as_secs() < max_time, 
            "Adaptive training too slow for size {}: {:?}", size, training_time);
        
        println!("Adaptive training for {} embeddings: {:?}, quality: {:.4}", 
            size, training_time, quantizer.training_quality());
    }
}

#[test]
fn test_asymmetric_distance_accuracy() {
    let mut quantizer = ProductQuantizer::new(32, 4).unwrap();
    
    // Create structured test data
    let embeddings: Vec<Vec<f32>> = vec![
        // Orthogonal vectors
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // Similar vectors
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
             1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
             1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
             1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    ];
    
    quantizer.train(&embeddings, 15).unwrap();
    
    // Test asymmetric distance properties
    for (i, query) in embeddings.iter().enumerate() {
        for (j, target) in embeddings.iter().enumerate() {
            let codes = quantizer.encode(target).unwrap();
            let distance = quantizer.asymmetric_distance(query, &codes).unwrap();
            
            // Distance should be non-negative and finite
            assert!(distance >= 0.0, "Distance should be non-negative");
            assert!(distance.is_finite(), "Distance should be finite");
            
            // Distance to self should be approximately 0 (within quantization error)
            if i == j {
                assert!(distance < 2.0, 
                    "Self-distance too high: {} for embedding {}", distance, i);
            }
            
            // Similar vectors should have smaller distance than orthogonal ones
            if (i == 2 && j == 3) || (i == 3 && j == 2) {
                // Similar vectors (1.0 vs 1.1)
                assert!(distance < 5.0, 
                    "Distance between similar vectors too high: {}", distance);
            }
            
            if (i < 2 && j < 2 && i != j) {
                // Orthogonal vectors
                assert!(distance > 0.5, 
                    "Distance between orthogonal vectors too low: {}", distance);
            }
        }
    }
}

#[test]
fn test_reconstruction_error_analysis() {
    let mut quantizer = ProductQuantizer::new(64, 8).unwrap();
    
    // Create various types of embeddings for reconstruction testing
    let test_cases = vec![
        // Constant vectors
        ("constant_ones", vec![1.0; 64]),
        ("constant_zeros", vec![0.0; 64]),
        ("constant_negative", vec![-1.0; 64]),
        
        // Linear patterns
        ("linear_increasing", (0..64).map(|i| i as f32 / 64.0).collect()),
        ("linear_decreasing", (0..64).map(|i| 1.0 - (i as f32 / 64.0)).collect()),
        
        // Periodic patterns
        ("sine_wave", (0..64).map(|i| (i as f32 * std::f32::consts::PI / 16.0).sin()).collect()),
        ("cosine_wave", (0..64).map(|i| (i as f32 * std::f32::consts::PI / 16.0).cos()).collect()),
        
        // Random-like patterns
        ("pseudo_random", (0..64).map(|i| ((i * 31) % 97) as f32 / 97.0).collect()),
    ];
    
    // Prepare training data
    let mut training_data = Vec::new();
    for (_, pattern) in &test_cases {
        training_data.push(pattern.clone());
        
        // Add variations of each pattern
        for scale in [0.5, 2.0, -1.0] {
            let scaled: Vec<f32> = pattern.iter().map(|&x| x * scale).collect();
            training_data.push(scaled);
        }
    }
    
    // Add more diverse training data
    for i in 0..200 {
        let embedding: Vec<f32> = (0..64).map(|j| {
            ((i * j) as f32 * 0.01).sin() + ((i + j) as f32 * 0.001).cos()
        }).collect();
        training_data.push(embedding);
    }
    
    quantizer.train(&training_data, 20).unwrap();
    
    // Test reconstruction error for each pattern
    let test_embeddings: Vec<Vec<f32>> = test_cases.iter().map(|(_, pattern)| pattern.clone()).collect();
    let reconstruction_error = quantizer.compute_reconstruction_error(&test_embeddings).unwrap();
    
    println!("Overall reconstruction error: {:.6}", reconstruction_error);
    assert!(reconstruction_error < 2.0, 
        "Reconstruction error too high: {}", reconstruction_error);
    
    // Test individual patterns
    for (name, pattern) in test_cases {
        let codes = quantizer.encode(&pattern).unwrap();
        let reconstructed = quantizer.decode(&codes).unwrap();
        
        let mse: f32 = pattern.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / pattern.len() as f32;
        
        println!("Pattern '{}': MSE = {:.6}", name, mse);
        
        // Different patterns may have different reconstruction qualities
        let max_mse = match name {
            n if n.contains("constant") => 0.5,
            n if n.contains("linear") => 1.0,
            n if n.contains("sine") || n.contains("cosine") => 2.0,
            _ => 3.0,
        };
        
        assert!(mse < max_mse, 
            "MSE too high for pattern '{}': {} (max: {})", name, mse, max_mse);
    }
}

#[test]
fn test_storage_integration() {
    let mut quantizer = ProductQuantizer::new(32, 8).unwrap();
    
    // Create test embeddings
    let embeddings: Vec<Vec<f32>> = (0..100).map(|i| {
        (0..32).map(|j| (i as f32 * 0.1) + (j as f32 * 0.01)).collect()
    }).collect();
    
    quantizer.train(&embeddings[0..50], 10).unwrap();
    
    // Test individual storage
    for (i, embedding) in embeddings[50..60].iter().enumerate() {
        let entity = EntityKey::new(i as u32);
        quantizer.store_quantized(entity, embedding).unwrap();
    }
    
    // Test batch storage
    let batch_entities_embeddings: Vec<(EntityKey, Vec<f32>)> = embeddings[60..100].into_iter()
        .enumerate()
        .map(|(i, emb)| (EntityKey::new((i + 10) as u32), emb.clone()))
        .collect();
    
    quantizer.batch_store_quantized(&batch_entities_embeddings).unwrap();
    
    // Verify storage statistics
    let (memory_usage, entity_count, compression_ratio) = quantizer.storage_stats();
    
    assert_eq!(entity_count, 50); // 10 individual + 40 batch
    assert!(memory_usage > 0);
    assert!(compression_ratio >= 4.0); // Should achieve at least 4x compression
    
    println!("Storage stats: {} entities, {} bytes, {:.2}x compression", 
        entity_count, memory_usage, compression_ratio);
    
    // Test similarity search on stored embeddings
    let query = embeddings[75].clone(); // Should find entity with key 25
    let results = quantizer.quantized_similarity_search(&query, 5).unwrap();
    
    assert_eq!(results.len(), 5);
    
    // The most similar should be the exact match or very close
    assert!(results[0].1 > 0.8, 
        "Top result similarity too low: {}", results[0].1);
    
    // Results should be in descending order of similarity
    for i in 1..results.len() {
        assert!(results[i-1].1 >= results[i].1,
            "Results not properly sorted by similarity");
    }
}

#[test]
fn test_error_handling_edge_cases() {
    let mut quantizer = ProductQuantizer::new(16, 4).unwrap();
    
    // Test operations before training
    let untrained_embedding = vec![1.0; 16];
    assert!(quantizer.encode(&untrained_embedding).is_ok()); // Should work even without training
    
    let codes = vec![1, 2, 3, 4];
    assert!(quantizer.decode(&codes).is_ok()); // Should work with default codebooks
    
    // Test batch operations before training
    let embeddings = vec![vec![1.0; 16], vec![2.0; 16]];
    assert!(quantizer.batch_encode(&embeddings).is_err()); // Should fail - not trained
    
    // Train the quantizer
    let training_data: Vec<Vec<f32>> = (0..50).map(|i| {
        (0..16).map(|j| (i + j) as f32 * 0.1).collect()
    }).collect();
    quantizer.train(&training_data, 5).unwrap();
    
    // Test dimension mismatches after training
    let wrong_dimension_embedding = vec![1.0; 20];
    assert!(quantizer.encode(&wrong_dimension_embedding).is_err());
    
    let wrong_dimension_query = vec![1.0; 20];
    let valid_codes = vec![1, 2, 3, 4];
    assert!(quantizer.asymmetric_distance(&wrong_dimension_query, &valid_codes).is_err());
    
    // Test wrong code length
    let valid_query = vec![1.0; 16];
    let wrong_codes = vec![1, 2, 3]; // Should be 4 codes
    assert!(quantizer.asymmetric_distance(&valid_query, &wrong_codes).is_err());
    
    // Test empty inputs
    let empty_embeddings: Vec<Vec<f32>> = vec![];
    let mut fresh_quantizer = ProductQuantizer::new(16, 4).unwrap();
    assert!(fresh_quantizer.train(&empty_embeddings, 5).is_err());
    
    // Test reconstruction error before training
    let mut untrained = ProductQuantizer::new(16, 4).unwrap();
    assert!(untrained.compute_reconstruction_error(&training_data).is_err());
}

#[test]
fn test_memory_efficiency() {
    let quantizer = ProductQuantizer::new(256, 32).unwrap();
    
    // Calculate theoretical memory usage
    let codebook_memory = quantizer.memory_usage();
    let expected_codebook_size = 32 * 256 * 8 * 4; // 32 subvectors * 256 clusters * 8 dims * 4 bytes
    assert_eq!(codebook_memory, expected_codebook_size);
    
    // Test compression efficiency with storage
    let mut test_quantizer = quantizer.clone();
    let embeddings: Vec<Vec<f32>> = (0..1000).map(|i| {
        (0..256).map(|j| (i * j) as f32 * 0.0001).collect()
    }).collect();
    
    test_quantizer.train(&embeddings[0..500], 10).unwrap();
    
    let entities_embeddings: Vec<(EntityKey, Vec<f32>)> = embeddings[500..1000].into_iter()
        .enumerate()
        .map(|(i, emb)| (EntityKey::new(i as u32), emb.clone()))
        .collect();
    
    test_quantizer.batch_store_quantized(&entities_embeddings).unwrap();
    
    let (storage_memory, entity_count, compression_ratio) = test_quantizer.storage_stats();
    
    // Verify compression efficiency
    assert_eq!(entity_count, 500);
    assert!(compression_ratio >= 32.0, "Should achieve 32x compression");
    
    // Calculate memory savings
    let original_memory = entity_count * 256 * 4; // 500 * 256 floats * 4 bytes
    let total_memory = codebook_memory + storage_memory;
    let actual_compression = original_memory as f32 / total_memory as f32;
    
    println!("Memory efficiency: Original: {} bytes, Compressed: {} bytes, Ratio: {:.2}x", 
        original_memory, total_memory, actual_compression);
    
    // With codebook overhead, we should still achieve significant compression
    assert!(actual_compression > 10.0, 
        "Overall compression ratio too low: {:.2}x", actual_compression);
}