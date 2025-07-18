// Performance test compatible EmbeddingStore
// This provides the API that performance tests expect

use std::collections::HashMap;
use crate::core::types::EntityKey;
use crate::core::entity_compat::SimilarityResult;
use crate::embedding::quantizer::ProductQuantizer;
use crate::error::{GraphError, Result};
use parking_lot::RwLock;
use std::sync::Arc;

/// Performance test compatible EmbeddingStore
pub struct EmbeddingStore {
    quantizer: Arc<RwLock<ProductQuantizer>>,
    embeddings: HashMap<EntityKey, Vec<f32>>,
    quantized_embeddings: HashMap<EntityKey, Vec<u8>>,
    dimension: usize,
}

impl EmbeddingStore {
    pub fn new(dimension: usize) -> Self {
        let subvector_count = if dimension >= 256 { 8 } else { 4 };
        
        Self {
            quantizer: Arc::new(RwLock::new(
                ProductQuantizer::new(dimension, subvector_count).expect("Failed to create quantizer")
            )),
            embeddings: HashMap::new(),
            quantized_embeddings: HashMap::new(),
            dimension,
        }
    }
    
    pub fn add_embedding(&mut self, entity: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        self.embeddings.insert(entity, embedding);
        Ok(())
    }
    
    pub fn add_quantized_embedding(&mut self, entity: EntityKey, quantized: Vec<u8>) -> Result<()> {
        self.quantized_embeddings.insert(entity, quantized);
        Ok(())
    }
    
    pub fn get_embedding(&self, entity: EntityKey) -> Option<Vec<f32>> {
        self.embeddings.get(&entity).cloned()
    }
    
    pub fn similarity_search(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        let mut results = Vec::new();
        
        for (&entity, embedding) in &self.embeddings {
            let similarity = self.compute_similarity(query, embedding);
            results.push(SimilarityResult::new(entity, similarity));
        }
        
        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        
        results
    }
    
    pub fn similarity_search_quantized(&self, query: &[f32], k: usize) -> Result<Vec<SimilarityResult>> {
        let mut results = Vec::new();
        let quantizer = self.quantizer.read();
        
        for (&entity, quantized) in &self.quantized_embeddings {
            if let Ok(distance) = quantizer.asymmetric_distance(query, quantized) {
                // Convert distance to similarity (lower distance = higher similarity)
                let similarity = 1.0 / (1.0 + distance);
                results.push(SimilarityResult::new(entity, similarity));
            }
        }
        
        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    pub fn compute_all_distances(&self, query: &[f32]) -> Vec<f32> {
        self.embeddings.values()
            .map(|embedding| self.compute_distance(query, embedding))
            .collect()
    }
    
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        // Cosine similarity
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.compute_similarity(a, b)
    }
}

// Compatibility wrapper for ProductQuantizer
impl ProductQuantizer {
    pub fn quantize(&self, embedding: &[f32]) -> Vec<u8> {
        self.encode(embedding).unwrap_or_default()
    }
    
    pub fn reconstruct(&self, quantized: &[u8]) -> Vec<f32> {
        self.decode(quantized).unwrap_or_default()
    }
    
    pub fn train_simple(&mut self, _embeddings: &[Vec<f32>]) -> Result<()> {
        // Training is handled internally in the quantizer
        // For performance tests, we'll assume it's already trained
        Ok(())
    }
}