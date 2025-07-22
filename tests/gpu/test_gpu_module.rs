//! Integration tests for GPU module functionality

use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_gpu_module_imports() {
    // Test that we can import the GPU module types
    use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
    
    // Verify CpuGraphProcessor can be instantiated
    let processor = CpuGraphProcessor::new();
    assert!(matches!(processor.graph, None));
}

#[test]
fn test_cpu_processor_basic_functionality() {
    use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
    
    let processor = CpuGraphProcessor::new();
    
    // Test parallel traversal without graph (should return input nodes)
    let nodes = vec![1, 2, 3];
    let result = processor.parallel_traversal(&nodes, 5);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), nodes);
    
    // Test batch similarity
    let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let query = vec![1.0, 0.0];
    let result = processor.batch_similarity(&embeddings, &query);
    assert!(result.is_ok());
    let similarities = result.unwrap();
    assert_eq!(similarities.len(), 2);
    assert!((similarities[0] - 1.0).abs() < 1e-6); // First embedding matches query
    assert!((similarities[1] - 0.0).abs() < 1e-6); // Second embedding orthogonal to query
    
    // Test parallel shortest paths without graph
    let sources = vec![1, 2];
    let targets = vec![3, 4];
    let result = processor.parallel_shortest_paths(&sources, &targets);
    assert!(result.is_ok());
    let paths = result.unwrap();
    assert_eq!(paths.len(), 2);
    assert!(paths.iter().all(|p| p.is_none()));
}

#[test]
fn test_cpu_processor_with_graph() {
    use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
    
    // Create a simple graph
    let mut graph = HashMap::new();
    graph.insert(0, vec![1, 2]);
    graph.insert(1, vec![3]);
    graph.insert(2, vec![3, 4]);
    graph.insert(3, vec![4]);
    graph.insert(4, vec![]);
    
    let graph_arc = Arc::new(graph);
    let processor = CpuGraphProcessor::new().with_graph(graph_arc);
    
    // Test parallel traversal with actual graph
    let start_nodes = vec![0];
    let result = processor.parallel_traversal(&start_nodes, 2);
    assert!(result.is_ok());
    let visited = result.unwrap();
    assert!(visited.contains(&0));
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3)); // depth 2 from node 0
    
    // Test shortest paths
    let sources = vec![0, 1];
    let targets = vec![4, 3];
    let result = processor.parallel_shortest_paths(&sources, &targets);
    assert!(result.is_ok());
    let paths = result.unwrap();
    assert_eq!(paths.len(), 2);
    
    // Path from 0 to 4
    assert!(paths[0].is_some());
    let path0 = paths[0].as_ref().unwrap();
    assert_eq!(path0[0], 0);
    assert_eq!(path0[path0.len() - 1], 4);
    
    // Path from 1 to 3
    assert!(paths[1].is_some());
    let path1 = paths[1].as_ref().unwrap();
    assert_eq!(path1, &vec![1, 3]);
}

#[test]
fn test_cosine_similarity_edge_cases() {
    use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
    
    let processor = CpuGraphProcessor::new();
    
    // Test with zero vectors
    let embeddings = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
    ];
    let query = vec![0.0, 0.0];
    let result = processor.batch_similarity(&embeddings, &query);
    assert!(result.is_ok());
    let similarities = result.unwrap();
    assert_eq!(similarities[0], 0.0); // zero vector similarity
    assert_eq!(similarities[1], 0.0); // zero query vector
}

#[test]
fn test_parallel_traversal_multiple_start_nodes() {
    use llmkg::gpu::{CpuGraphProcessor, GpuAccelerator};
    
    // Create a graph with separate components
    let mut graph = HashMap::new();
    // Component 1
    graph.insert(0, vec![1]);
    graph.insert(1, vec![]);
    // Component 2
    graph.insert(2, vec![3]);
    graph.insert(3, vec![]);
    
    let graph_arc = Arc::new(graph);
    let processor = CpuGraphProcessor::new().with_graph(graph_arc);
    
    // Start from both components
    let start_nodes = vec![0, 2];
    let result = processor.parallel_traversal(&start_nodes, 10);
    assert!(result.is_ok());
    let visited = result.unwrap();
    assert_eq!(visited.len(), 4);
    assert!(visited.contains(&0));
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_processor_not_implemented() {
    use llmkg::gpu::cuda::CudaGraphProcessor;
    
    let result = CudaGraphProcessor::new();
    assert!(result.is_err());
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_cuda_processor_feature_disabled() {
    use llmkg::gpu::cuda::CudaGraphProcessor;
    
    let result = CudaGraphProcessor::new();
    assert!(result.is_err());
}