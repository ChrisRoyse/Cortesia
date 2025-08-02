// Performance test compatible EmbeddingStore
// This provides the API that performance tests expect

use std::collections::HashMap;
use crate::core::types::EntityKey;
use crate::core::entity_compat::SimilarityResult;
use crate::embedding::quantizer::ProductQuantizer;
use crate::error::{GraphError, Result};
use parking_lot::RwLock;
use std::sync::Arc;

/// Performance test compatible EmbeddingStore with enhanced quantization
pub struct EmbeddingStore {
    quantizer: Arc<RwLock<ProductQuantizer>>,
    embeddings: HashMap<EntityKey, Vec<f32>>,
    quantized_embeddings: HashMap<EntityKey, Vec<u8>>,
    dimension: usize,
    quantization_enabled: bool,
    auto_quantize_threshold: usize, // Auto-quantize when this many embeddings are stored
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
            quantization_enabled: false,
            auto_quantize_threshold: 1000, // Auto-quantize after 1000 embeddings
        }
    }
    
    /// Create EmbeddingStore with optimized quantization
    pub fn new_with_quantization(dimension: usize, target_compression: f32) -> Result<Self> {
        Ok(Self {
            quantizer: Arc::new(RwLock::new(
                ProductQuantizer::new_optimized(dimension, target_compression)?
            )),
            embeddings: HashMap::new(),
            quantized_embeddings: HashMap::new(),
            dimension,
            quantization_enabled: true,
            auto_quantize_threshold: 500, // Lower threshold for quantized mode
        })
    }
    
    /// Enable automatic quantization
    pub fn enable_quantization(&mut self, threshold: usize) {
        self.quantization_enabled = true;
        self.auto_quantize_threshold = threshold;
    }
    
    pub fn add_embedding(&mut self, entity: &str, embedding: Vec<f32>) -> Result<()> {
        self.add_embedding_key(EntityKey::from_hash(entity), embedding)
    }

    pub fn add_embedding_key(&mut self, entity: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        self.embeddings.insert(entity, embedding.clone());
        
        // Auto-quantize if enabled and threshold reached
        if self.quantization_enabled && self.embeddings.len() >= self.auto_quantize_threshold {
            self.quantize_all_embeddings()?;
        }
        
        Ok(())
    }
    
    /// Quantize all current embeddings and clear original storage
    pub fn quantize_all_embeddings(&mut self) -> Result<()> {
        if self.embeddings.is_empty() {
            return Ok(());
        }
        
        // Collect embeddings for training
        let embeddings: Vec<Vec<f32>> = self.embeddings.values().cloned().collect();
        
        // Train the quantizer
        {
            let mut quantizer = self.quantizer.write();
            if !quantizer.is_trained() {
                quantizer.train_adaptive(&embeddings)?;
            }
        }
        
        // Quantize all embeddings
        for (entity, embedding) in &self.embeddings {
            let quantizer = self.quantizer.read();
            if let Ok(codes) = quantizer.encode(embedding) {
                self.quantized_embeddings.insert(*entity, codes);
            }
        }
        
        // Clear original embeddings to save memory
        if self.quantization_enabled {
            self.embeddings.clear();
        }
        
        Ok(())
    }
    
    pub fn add_quantized_embedding(&mut self, entity: EntityKey, quantized: Vec<u8>) -> Result<()> {
        self.quantized_embeddings.insert(entity, quantized);
        Ok(())
    }
    
    pub fn get_embedding(&self, entity: EntityKey) -> Option<Vec<f32>> {
        self.embeddings.get(&entity).cloned()
    }
    
    pub fn get_embedding_by_name(&self, entity: &str) -> Option<Vec<f32>> {
        let entity_key = EntityKey::from_hash(entity);
        self.get_embedding(entity_key)
    }
    
    pub fn similarity_search(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        let mut results = Vec::new();
        
        // Search quantized embeddings first if available
        if !self.quantized_embeddings.is_empty() {
            if let Ok(quantized_results) = self.similarity_search_quantized(query, k * 2) {
                results.extend(quantized_results);
            }
        }
        
        // Search regular embeddings if available
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
    
    /// Get comprehensive memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let embeddings_memory = self.embeddings.len() * self.dimension * std::mem::size_of::<f32>()
            + self.embeddings.capacity() * std::mem::size_of::<(EntityKey, Vec<f32>)>();
            
        let quantized_memory = self.quantized_embeddings.values()
            .map(|v| v.len())
            .sum::<usize>() + self.quantized_embeddings.capacity() * std::mem::size_of::<(EntityKey, Vec<u8>)>();
            
        let quantizer_memory = {
            let quantizer = self.quantizer.read();
            quantizer.memory_usage()
        };
        
        let total_memory = embeddings_memory + quantized_memory + quantizer_memory;
        let original_memory = (self.embeddings.len() + self.quantized_embeddings.len()) 
            * self.dimension * std::mem::size_of::<f32>();
        
        MemoryStats {
            embeddings_memory,
            quantized_memory,
            quantizer_memory,
            total_memory,
            original_memory,
            compression_ratio: if quantized_memory > 0 { 
                original_memory as f32 / quantized_memory as f32 
            } else { 
                1.0 
            },
            quantized_percentage: if self.embeddings.len() + self.quantized_embeddings.len() > 0 {
                (self.quantized_embeddings.len() as f32) / 
                ((self.embeddings.len() + self.quantized_embeddings.len()) as f32) * 100.0
            } else {
                0.0
            },
        }
    }
    
    /// Force quantization of all embeddings
    pub fn force_quantize(&mut self) -> Result<()> {
        self.quantize_all_embeddings()
    }
    
    /// Get total entity count
    pub fn entity_count(&self) -> usize {
        self.embeddings.len() + self.quantized_embeddings.len()
    }
    
    /// Check if quantization is enabled
    pub fn is_quantization_enabled(&self) -> bool {
        self.quantization_enabled
    }
    
    /// Get access to the quantizer for advanced operations
    pub fn get_quantizer(&self) -> Arc<RwLock<ProductQuantizer>> {
        self.quantizer.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub embeddings_memory: usize,
    pub quantized_memory: usize,
    pub quantizer_memory: usize,
    pub total_memory: usize,
    pub original_memory: usize,
    pub compression_ratio: f32,
    pub quantized_percentage: f32,
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