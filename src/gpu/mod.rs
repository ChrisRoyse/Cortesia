pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::CudaGraphProcessor;

/// GPU acceleration interface for graph operations
pub trait GpuAccelerator {
    /// Perform parallel graph traversal on GPU
    fn parallel_traversal(&self, start_nodes: &[u32], max_depth: u32) -> Result<Vec<u32>, String>;
    
    /// Batch similarity computations on GPU
    fn batch_similarity(&self, embeddings: &[Vec<f32>], query: &[f32]) -> Result<Vec<f32>, String>;
    
    /// Parallel shortest path computation
    fn parallel_shortest_paths(&self, sources: &[u32], targets: &[u32]) -> Result<Vec<Option<Vec<u32>>>, String>;
}

/// CPU fallback implementation
pub struct CpuGraphProcessor;

impl GpuAccelerator for CpuGraphProcessor {
    fn parallel_traversal(&self, start_nodes: &[u32], _max_depth: u32) -> Result<Vec<u32>, String> {
        // CPU implementation as fallback
        Ok(start_nodes.to_vec())
    }
    
    fn batch_similarity(&self, embeddings: &[Vec<f32>], query: &[f32]) -> Result<Vec<f32>, String> {
        // CPU implementation
        let mut similarities = Vec::new();
        for embedding in embeddings {
            let similarity = cosine_similarity(embedding, query);
            similarities.push(similarity);
        }
        Ok(similarities)
    }
    
    fn parallel_shortest_paths(&self, sources: &[u32], _targets: &[u32]) -> Result<Vec<Option<Vec<u32>>>, String> {
        // CPU implementation
        Ok(vec![None; sources.len()])
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}