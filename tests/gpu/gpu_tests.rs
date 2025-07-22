//! Comprehensive tests for the GPU module
//!
//! Tests cover:
//! - CudaGraphProcessor creation and error handling
//! - CpuGraphProcessor as fallback implementation
//! - GpuAccelerator trait methods
//! - Error handling for unimplemented features

use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
use llmkg::error::GraphError;

#[cfg(feature = "cuda")]
use llmkg::gpu::CudaGraphProcessor;

#[cfg(test)]
mod cuda_processor_tests {
    use super::*;
    
    // Import CudaGraphProcessor directly from cuda module (available in both feature configurations)
    use llmkg::gpu::cuda::CudaGraphProcessor;

    #[test]
    fn test_cuda_processor_creation_without_cuda_feature() {
        // When CUDA feature is not enabled, creation should fail with FeatureNotEnabled
        #[cfg(not(feature = "cuda"))]
        {
            let result = CudaGraphProcessor::new();
            assert!(result.is_err());
            match result.unwrap_err() {
                GraphError::FeatureNotEnabled(feature) => {
                    assert_eq!(feature, "cuda");
                }
                _ => panic!("Expected FeatureNotEnabled error"),
            }
        }
    }

    #[test]
    fn test_cuda_processor_creation_with_cuda_feature() {
        // When CUDA feature is enabled, creation should fail with NotImplemented
        #[cfg(feature = "cuda")]
        {
            let result = CudaGraphProcessor::new();
            assert!(result.is_err());
            match result.unwrap_err() {
                GraphError::NotImplemented(msg) => {
                    assert!(msg.contains("CUDA support is not yet implemented"));
                    assert!(msg.contains("GPU acceleration requires CUDA toolkit"));
                }
                _ => panic!("Expected NotImplemented error"),
            }
        }
    }
}

#[cfg(test)]
mod cpu_processor_tests {
    use super::*;

    #[test]
    fn test_cpu_processor_creation() {
        // CPU processor should be directly instantiable
        let processor = CpuGraphProcessor::new();
        // Verify it implements the trait
        let _: &dyn GpuAccelerator = &processor;
    }

    #[test]
    fn test_cpu_parallel_traversal_basic() {
        let processor = CpuGraphProcessor::new();
        let start_nodes = vec![1, 2, 3, 4, 5];
        let max_depth = 10;
        
        let result = processor.parallel_traversal(&start_nodes, max_depth);
        assert!(result.is_ok());
        
        let traversed = result.unwrap();
        assert_eq!(traversed, start_nodes);
    }

    #[test]
    fn test_cpu_parallel_traversal_empty() {
        let processor = CpuGraphProcessor::new();
        let start_nodes = vec![];
        let max_depth = 5;
        
        let result = processor.parallel_traversal(&start_nodes, max_depth);
        assert!(result.is_ok());
        
        let traversed = result.unwrap();
        assert!(traversed.is_empty());
    }

    #[test]
    fn test_cpu_parallel_traversal_large_depth() {
        let processor = CpuGraphProcessor::new();
        let start_nodes = vec![100, 200, 300];
        let max_depth = u32::MAX;
        
        let result = processor.parallel_traversal(&start_nodes, max_depth);
        assert!(result.is_ok());
        
        let traversed = result.unwrap();
        assert_eq!(traversed, start_nodes);
    }

    #[test]
    fn test_cpu_batch_similarity_basic() {
        let processor = CpuGraphProcessor::new();
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![1.0, 0.0, 0.0];
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert_eq!(similarities.len(), 3);
        
        // First embedding should have similarity 1.0 (same as query)
        assert!((similarities[0] - 1.0).abs() < f32::EPSILON);
        
        // Other embeddings should have similarity 0.0 (orthogonal)
        assert!((similarities[1] - 0.0).abs() < f32::EPSILON);
        assert!((similarities[2] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_batch_similarity_normalized() {
        let processor = CpuGraphProcessor::new();
        let embeddings = vec![
            vec![3.0, 4.0],  // Magnitude 5
            vec![1.0, 0.0],  // Magnitude 1
        ];
        let query = vec![0.6, 0.8];  // Normalized version of [3, 4]
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert_eq!(similarities.len(), 2);
        
        // Both should have similarity 1.0 as they point in the same direction
        assert!((similarities[0] - 1.0).abs() < f32::EPSILON);
        assert!((similarities[1] - 0.6).abs() < 0.001);  // cos(angle) = 0.6
    }

    #[test]
    fn test_cpu_batch_similarity_empty_embeddings() {
        let processor = CpuGraphProcessor::new();
        let embeddings: Vec<Vec<f32>> = vec![];
        let query = vec![1.0, 0.0, 0.0];
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert!(similarities.is_empty());
    }

    #[test]
    fn test_cpu_batch_similarity_zero_vectors() {
        let processor = CpuGraphProcessor::new();
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],  // Zero vector
            vec![1.0, 1.0, 1.0],
        ];
        let query = vec![1.0, 1.0, 1.0];
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert_eq!(similarities.len(), 2);
        
        // Zero vector should have similarity 0.0
        assert!((similarities[0] - 0.0).abs() < f32::EPSILON);
        
        // Same vector should have similarity 1.0
        assert!((similarities[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_batch_similarity_zero_query() {
        let processor = CpuGraphProcessor::new();
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let query = vec![0.0, 0.0, 0.0];  // Zero query
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert_eq!(similarities.len(), 2);
        
        // All similarities should be 0.0 with zero query
        assert!((similarities[0] - 0.0).abs() < f32::EPSILON);
        assert!((similarities[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_batch_similarity_negative_values() {
        let processor = CpuGraphProcessor::new();
        let embeddings = vec![
            vec![1.0, -1.0],
            vec![-1.0, 1.0],
        ];
        let query = vec![1.0, -1.0];
        
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let similarities = result.unwrap();
        assert_eq!(similarities.len(), 2);
        
        // First should have similarity 1.0 (same)
        assert!((similarities[0] - 1.0).abs() < f32::EPSILON);
        
        // Second should have similarity -1.0 (opposite)
        assert!((similarities[1] - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_parallel_shortest_paths_basic() {
        let processor = CpuGraphProcessor::new();
        let sources = vec![1, 2, 3];
        let targets = vec![10, 20, 30];
        
        let result = processor.parallel_shortest_paths(&sources, &targets);
        assert!(result.is_ok());
        
        let paths = result.unwrap();
        assert_eq!(paths.len(), sources.len());
        
        // CPU implementation returns None for all paths
        for path in paths {
            assert!(path.is_none());
        }
    }

    #[test]
    fn test_cpu_parallel_shortest_paths_empty() {
        let processor = CpuGraphProcessor::new();
        let sources = vec![];
        let targets = vec![];
        
        let result = processor.parallel_shortest_paths(&sources, &targets);
        assert!(result.is_ok());
        
        let paths = result.unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_cpu_parallel_shortest_paths_mismatched_lengths() {
        let processor = CpuGraphProcessor::new();
        let sources = vec![1, 2, 3];
        let targets = vec![10];  // Different length
        
        let result = processor.parallel_shortest_paths(&sources, &targets);
        assert!(result.is_ok());
        
        let paths = result.unwrap();
        assert_eq!(paths.len(), sources.len());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_gpu_accelerator_trait_object() {
        // Test that we can use CPU processor as trait object
        let processor: Box<dyn GpuAccelerator> = Box::new(CpuGraphProcessor::new());
        
        // Test all trait methods through the trait object
        let nodes = vec![1, 2, 3];
        let result = processor.parallel_traversal(&nodes, 5);
        assert!(result.is_ok());
        
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query = vec![1.0, 0.0];
        let result = processor.batch_similarity(&embeddings, &query);
        assert!(result.is_ok());
        
        let sources = vec![1];
        let targets = vec![2];
        let result = processor.parallel_shortest_paths(&sources, &targets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_propagation() {
        // Test that errors are properly propagated as Result<_, String>
        let processor = CpuGraphProcessor::new();
        
        // All methods should return Ok for CPU implementation
        assert!(processor.parallel_traversal(&[1, 2, 3], 10).is_ok());
        assert!(processor.batch_similarity(&[vec![1.0]], &[1.0]).is_ok());
        assert!(processor.parallel_shortest_paths(&[1], &[2]).is_ok());
    }

    #[test]
    fn test_large_batch_processing() {
        let processor = CpuGraphProcessor::new();
        
        // Test with large number of nodes
        let large_nodes: Vec<u32> = (0..10000).collect();
        let result = processor.parallel_traversal(&large_nodes, 100);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10000);
        
        // Test with large number of embeddings
        let large_embeddings: Vec<Vec<f32>> = (0..1000)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        let query = vec![1.0, 2.0, 3.0];
        let result = processor.batch_similarity(&large_embeddings, &query);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1000);
    }

    #[test]
    fn test_edge_cases() {
        let processor = CpuGraphProcessor::new();
        
        // Test with max values
        let max_nodes = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2];
        assert!(processor.parallel_traversal(&max_nodes, u32::MAX).is_ok());
        
        // Test with very small floating point values
        let tiny_embeddings = vec![
            vec![1e-10, 1e-10],
            vec![1e-20, 1e-20],
        ];
        let tiny_query = vec![1e-10, 1e-10];
        let result = processor.batch_similarity(&tiny_embeddings, &tiny_query);
        assert!(result.is_ok());
        
        // Test with infinity values
        let inf_embeddings = vec![
            vec![f32::INFINITY, 0.0],
            vec![0.0, f32::NEG_INFINITY],
        ];
        let normal_query = vec![1.0, 1.0];
        let result = processor.batch_similarity(&inf_embeddings, &normal_query);
        assert!(result.is_ok());
        
        // Results should handle infinity gracefully
        let similarities = result.unwrap();
        assert!(similarities[0].is_nan() || similarities[0].is_infinite());
        assert!(similarities[1].is_nan() || similarities[1].is_infinite());
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_batch_similarity_performance() {
        let processor = CpuGraphProcessor::new();
        
        // Create test data
        let num_embeddings = 10000;
        let embedding_dim = 128;
        let embeddings: Vec<Vec<f32>> = (0..num_embeddings)
            .map(|i| {
                (0..embedding_dim)
                    .map(|j| ((i * j) as f32).sin())
                    .collect()
            })
            .collect();
        let query: Vec<f32> = (0..embedding_dim)
            .map(|i| (i as f32).cos())
            .collect();
        
        // Measure performance
        let start = Instant::now();
        let result = processor.batch_similarity(&embeddings, &query);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), num_embeddings);
        
        // Performance should be reasonable (< 1 second for 10k embeddings)
        assert!(duration.as_secs() < 1, "Batch similarity took too long: {:?}", duration);
    }

    #[test]
    fn test_parallel_traversal_performance() {
        let processor = CpuGraphProcessor::new();
        
        // Create large node set
        let num_nodes = 1_000_000;
        let nodes: Vec<u32> = (0..num_nodes).collect();
        
        // Measure performance
        let start = Instant::now();
        let result = processor.parallel_traversal(&nodes, 100);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), num_nodes as usize);
        
        // Should be very fast as it's just copying the input
        assert!(duration.as_millis() < 100, "Parallel traversal took too long: {:?}", duration);
    }
}