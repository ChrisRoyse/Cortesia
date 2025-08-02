use crate::error::{GraphError, Result};
use crate::core::types::EntityKey;
use std::f32;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f32,
    pub codebook_memory: usize,
    pub subvector_count: usize,
    pub cluster_count: usize,
    pub training_quality: f32,
    pub memory_saved: usize,
    pub storage_efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct QuantizedEmbeddingStorage {
    pub codes: HashMap<EntityKey, Vec<u8>>,
    pub index: Option<Vec<(EntityKey, Vec<u8>)>>, // For faster searches
    pub entity_count: usize,
    pub memory_usage: usize,
}

impl Default for QuantizedEmbeddingStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedEmbeddingStorage {
    pub fn new() -> Self {
        Self {
            codes: HashMap::new(),
            index: None,
            entity_count: 0,
            memory_usage: 0,
        }
    }
    
    pub fn add_quantized(&mut self, entity: EntityKey, codes: Vec<u8>) {
        let code_size = codes.len();
        if let Some(old_codes) = self.codes.insert(entity, codes.clone()) {
            self.memory_usage = self.memory_usage.saturating_sub(old_codes.len());
        } else {
            self.entity_count += 1;
        }
        self.memory_usage += code_size;
        
        // Update index if it exists
        if let Some(ref mut index) = self.index {
            index.push((entity, codes));
        }
    }
    
    pub fn get_quantized(&self, entity: &EntityKey) -> Option<&Vec<u8>> {
        self.codes.get(entity)
    }
    
    pub fn build_search_index(&mut self) {
        self.index = Some(self.codes.iter().map(|(&k, v)| (k, v.clone())).collect());
    }
    
    pub fn memory_usage(&self) -> usize {
        self.memory_usage + (self.entity_count * std::mem::size_of::<EntityKey>())
    }
    
    pub fn compression_ratio(&self, original_dimension: usize) -> f32 {
        let original_size = self.entity_count * original_dimension * std::mem::size_of::<f32>();
        let compressed_size = self.memory_usage;
        original_size as f32 / compressed_size as f32
    }
}

pub struct ProductQuantizer {
    subvector_count: usize,      // M subvectors
    subvector_size: usize,       // D/M dimensions per subvector
    cluster_count: usize,        // K clusters per subvector (256 for u8 codes)
    codebooks: Vec<Vec<f32>>,    // M codebooks of K centroids
    is_trained: bool,            // Track if the quantizer has been trained
    training_quality: f32,       // Measure of quantization quality
    distance_cache: HashMap<(u8, u8), f32>, // Cache for frequent distance computations
    storage: Arc<RwLock<QuantizedEmbeddingStorage>>, // Quantized embeddings storage
}

impl ProductQuantizer {
    pub fn new(dimension: usize, subvector_count: usize) -> Result<Self> {
        if dimension % subvector_count != 0 {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: subvector_count * (dimension / subvector_count), 
                actual: dimension 
            });
        }
        
        let subvector_size = dimension / subvector_count;
        let cluster_count = 256; // For u8 codes
        
        Ok(Self {
            subvector_count,
            subvector_size,
            cluster_count,
            codebooks: vec![vec![0.0; subvector_size * cluster_count]; subvector_count],
            is_trained: false,
            training_quality: 0.0,
            distance_cache: HashMap::new(),
            storage: Arc::new(RwLock::new(QuantizedEmbeddingStorage::new())),
        })
    }
    
    /// Create optimized quantizer with automatic parameter selection
    pub fn new_optimized(dimension: usize, target_compression: f32) -> Result<Self> {
        // Calculate optimal subvector count based on target compression ratio
        let optimal_subvectors = match target_compression {
            r if r >= 32.0 => 8,   // High compression
            r if r >= 16.0 => 16,  // Medium compression
            r if r >= 8.0 => 32,   // Low compression
            _ => 4,                 // Minimal compression for high quality
        };
        
        let subvector_count = (dimension / optimal_subvectors).max(1).min(dimension);
        Self::new(dimension, subvector_count)
    }
    
    /// Train the Product Quantizer on a set of embeddings
    pub fn train(&mut self, embeddings: &[Vec<f32>], iterations: usize) -> Result<()> {
        if embeddings.is_empty() {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }
        
        let dimension = embeddings[0].len();
        if dimension != self.subvector_count * self.subvector_size {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.subvector_count * self.subvector_size, 
                actual: dimension 
            });
        }
        
        println!("🔧 Training Product Quantizer with {} embeddings, {} iterations", embeddings.len(), iterations);
        let mut total_distortion = 0.0;
        
        // Train each subquantizer independently
        for m in 0..self.subvector_count {
            let start = m * self.subvector_size;
            let end = start + self.subvector_size;
            
            // Extract subvectors for this quantizer
            let subvectors: Vec<Vec<f32>> = embeddings.iter()
                .map(|emb| emb[start..end].to_vec())
                .collect();
            
            // Run k-means clustering
            let distortion = self.train_subquantizer(m, &subvectors, iterations)?;
            total_distortion += distortion;
            
            if m % 2 == 0 {
                println!("  Subquantizer {}/{} trained (distortion: {:.4})", m + 1, self.subvector_count, distortion);
            }
        }
        
        self.training_quality = total_distortion / self.subvector_count as f32;
        self.is_trained = true;
        
        println!("✅ Product Quantizer training complete! Average distortion: {:.4}", self.training_quality);
        Ok(())
    }
    
    /// Train with automatic parameter selection based on data characteristics
    pub fn train_adaptive(&mut self, embeddings: &[Vec<f32>]) -> Result<()> {
        let sample_size = embeddings.len().min(10000); // Use subset for large datasets
        let max_iterations = if sample_size < 1000 { 50 } else { 20 }; // Fewer iterations for large datasets
        
        // Use a representative sample for training
        let training_data = if embeddings.len() > sample_size {
            let step = embeddings.len() / sample_size;
            embeddings.iter().step_by(step).cloned().collect()
        } else {
            embeddings.to_vec()
        };
        
        self.train(&training_data, max_iterations)
    }
    
    fn train_subquantizer(&mut self, quantizer_idx: usize, subvectors: &[Vec<f32>], iterations: usize) -> Result<f32> {
        let k = self.cluster_count;
        let dim = self.subvector_size;
        
        // Initialize centroids with random subvectors
        let mut centroids = vec![vec![0.0; dim]; k];
        let step = subvectors.len() / k;
        for i in 0..k {
            centroids[i] = subvectors[i * step].clone();
        }
        
        let mut assignments = vec![0usize; subvectors.len()];
        let mut prev_distortion = f32::INFINITY;
        
        for iter in 0..iterations {
            // Assignment step
            let mut changed = false;
            for (i, subvec) in subvectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = f32::INFINITY;
                
                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(subvec, centroid);
                    if dist < best_distance {
                        best_distance = dist;
                        best_cluster = j;
                    }
                }
                
                if assignments[i] != best_cluster {
                    changed = true;
                    assignments[i] = best_cluster;
                }
            }
            
            // Calculate current distortion for convergence check
            let mut current_distortion = 0.0;
            for (i, subvec) in subvectors.iter().enumerate() {
                let centroid = &centroids[assignments[i]];
                current_distortion += euclidean_distance(subvec, centroid).powi(2);
            }
            current_distortion /= subvectors.len() as f32;
            
            // Early stopping if converged
            if (!changed || (prev_distortion - current_distortion).abs() < 1e-6)
                && iter > 5 { // Ensure minimum iterations
                    break;
                }
            prev_distortion = current_distortion;
            
            // Update step
            for (j, centroid) in centroids.iter_mut().enumerate() {
                let mut count = 0;
                centroid.fill(0.0);
                
                for (i, &assignment) in assignments.iter().enumerate() {
                    if assignment == j {
                        for (k, &val) in subvectors[i].iter().enumerate() {
                            centroid[k] += val;
                        }
                        count += 1;
                    }
                }
                
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= count as f32;
                    }
                }
            }
        }
        
        // Store the trained codebook
        for (i, centroid) in centroids.iter().enumerate() {
            let offset = i * self.subvector_size;
            self.codebooks[quantizer_idx][offset..offset + self.subvector_size]
                .copy_from_slice(centroid);
        }
        
        // Calculate final distortion for this subquantizer
        let mut final_distortion = 0.0;
        for (i, subvec) in subvectors.iter().enumerate() {
            let centroid = &centroids[assignments[i]];
            final_distortion += euclidean_distance(subvec, centroid).powi(2);
        }
        final_distortion /= subvectors.len() as f32;
        
        Ok(final_distortion)
    }
    
    pub fn encode(&self, embedding: &[f32]) -> Result<Vec<u8>> {
        if embedding.len() != self.subvector_count * self.subvector_size {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.subvector_count * self.subvector_size, 
                actual: embedding.len() 
            });
        }
        
        let mut codes = Vec::with_capacity(self.subvector_count);
        
        for (m, chunk) in embedding.chunks(self.subvector_size).enumerate() {
            let mut best_cluster = 0u8;
            let mut best_distance = f32::INFINITY;
            
            for k in 0..self.cluster_count {
                let centroid_start = k * self.subvector_size;
                let centroid_end = centroid_start + self.subvector_size;
                let centroid = &self.codebooks[m][centroid_start..centroid_end];
                
                let distance = euclidean_distance(chunk, centroid);
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = k as u8;
                }
            }
            
            codes.push(best_cluster);
        }
        
        Ok(codes)
    }
    
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.subvector_count {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.subvector_count, 
                actual: codes.len() 
            });
        }
        
        let mut result = Vec::with_capacity(self.subvector_count * self.subvector_size);
        
        for (m, &code) in codes.iter().enumerate() {
            let centroid_start = (code as usize) * self.subvector_size;
            let centroid_end = centroid_start + self.subvector_size;
            result.extend_from_slice(&self.codebooks[m][centroid_start..centroid_end]);
        }
        
        Ok(result)
    }
    
    pub fn num_subspaces(&self) -> usize {
        self.subvector_count
    }
    
    pub fn get_codebook(&self, subspace: usize) -> &[f32] {
        &self.codebooks[subspace]
    }
    
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> Result<f32> {
        if query.len() != self.subvector_count * self.subvector_size {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.subvector_count * self.subvector_size, 
                actual: query.len() 
            });
        }
        
        if codes.len() != self.subvector_count {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.subvector_count, 
                actual: codes.len() 
            });
        }
        
        let mut distance = 0.0;
        
        for (m, (&code, chunk)) in codes.iter().zip(query.chunks(self.subvector_size)).enumerate() {
            let centroid_start = (code as usize) * self.subvector_size;
            let centroid_end = centroid_start + self.subvector_size;
            let centroid = &self.codebooks[m][centroid_start..centroid_end];
            
            for (&q, &c) in chunk.iter().zip(centroid.iter()) {
                let diff = q - c;
                distance += diff * diff;
            }
        }
        
        Ok(distance.sqrt())
    }
    
    pub fn memory_usage(&self) -> usize {
        self.codebooks.iter()
            .map(|cb| cb.capacity() * std::mem::size_of::<f32>())
            .sum()
    }
    
    /// Check if the quantizer has been trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    /// Get the training quality (lower is better)
    pub fn training_quality(&self) -> f32 {
        self.training_quality
    }
    
    /// Get compression statistics
    pub fn compression_stats(&self, original_dimension: usize) -> CompressionStats {
        let original_bytes = original_dimension * std::mem::size_of::<f32>();
        let compressed_bytes = self.subvector_count; // One u8 per subvector
        let compression_ratio = original_bytes as f32 / compressed_bytes as f32;
        let codebook_memory = self.memory_usage();
        let storage = self.storage.read();
        let memory_saved = original_bytes.saturating_sub(compressed_bytes);
        let storage_efficiency = storage.compression_ratio(original_dimension);
        
        CompressionStats {
            original_bytes,
            compressed_bytes,
            compression_ratio,
            codebook_memory,
            subvector_count: self.subvector_count,
            cluster_count: self.cluster_count,
            training_quality: self.training_quality,
            memory_saved,
            storage_efficiency,
        }
    }
    
    /// Batch encode multiple embeddings efficiently
    pub fn batch_encode(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        if !self.is_trained {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }
        
        embeddings.iter()
            .map(|emb| self.encode(emb))
            .collect()
    }
    
    /// Batch decode multiple quantized embeddings efficiently
    pub fn batch_decode(&self, codes: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        codes.iter()
            .map(|code| self.decode(code))
            .collect()
    }
    
    /// Store quantized embedding with entity key
    pub fn store_quantized(&self, entity: EntityKey, embedding: &[f32]) -> Result<()> {
        let codes = self.encode(embedding)?;
        let mut storage = self.storage.write();
        storage.add_quantized(entity, codes);
        Ok(())
    }
    
    /// Batch store multiple quantized embeddings
    pub fn batch_store_quantized(&self, entities_embeddings: &[(EntityKey, Vec<f32>)]) -> Result<()> {
        let mut storage = self.storage.write();
        
        for (entity, embedding) in entities_embeddings {
            let codes = self.encode(embedding)?;
            storage.add_quantized(*entity, codes);
        }
        
        // Build search index for better performance
        storage.build_search_index();
        Ok(())
    }
    
    /// Fast quantized similarity search
    pub fn quantized_similarity_search(&self, query: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let storage = self.storage.read();
        let mut results = Vec::new();
        
        for (entity, codes) in &storage.codes {
            if let Ok(distance) = self.asymmetric_distance(query, codes) {
                // Convert distance to similarity (lower distance = higher similarity)
                let similarity = 1.0 / (1.0 + distance);
                results.push((*entity, similarity));
            }
        }
        
        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Get storage statistics
    pub fn storage_stats(&self) -> (usize, usize, f32) {
        let storage = self.storage.read();
        let memory_usage = storage.memory_usage();
        let entity_count = storage.entity_count;
        let compression_ratio = storage.compression_ratio(self.subvector_count * self.subvector_size);
        (memory_usage, entity_count, compression_ratio)
    }
    
    /// Compute reconstruction error for a set of embeddings
    pub fn compute_reconstruction_error(&self, embeddings: &[Vec<f32>]) -> Result<f32> {
        if !self.is_trained {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }
        
        let mut total_error = 0.0;
        for embedding in embeddings {
            let codes = self.encode(embedding)?;
            let reconstructed = self.decode(&codes)?;
            
            let error: f32 = embedding.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total_error += error.sqrt();
        }
        
        Ok(total_error / embeddings.len() as f32)
    }
}

#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(target_feature = "avx2")]
mod simd {
    use std::arch::x86_64::*;
    
    #[inline]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            let squared = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, squared);
        }
        
        // Horizontal sum
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut total = result.iter().sum::<f32>();
        
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            let diff = a[i] - b[i];
            total += diff * diff;
        }
        
        total.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_encode_decode_cycle() {
        let mut quantizer = ProductQuantizer::new(64, 8).unwrap();
        
        // Create test embeddings
        let embeddings: Vec<Vec<f32>> = (0..100).map(|i| {
            (0..64).map(|j| (i * j) as f32 * 0.01).collect()
        }).collect();
        
        // Train the quantizer
        quantizer.train(&embeddings, 10).unwrap();
        
        // Test encode-decode cycle
        let test_embedding = embeddings[0].clone();
        let codes = quantizer.encode(&test_embedding).unwrap();
        let reconstructed = quantizer.decode(&codes).unwrap();
        
        assert_eq!(codes.len(), 8); // Should have 8 codes
        assert_eq!(reconstructed.len(), 64); // Should reconstruct to original dimension
        
        // Verify that reconstruction preserves approximate values
        let error: f32 = test_embedding.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        
        // Error should be reasonable (this is lossy compression)
        assert!(error < 100.0, "Reconstruction error too high: {error}");
    }

    #[test]
    fn test_asymmetric_distance() {
        let mut quantizer = ProductQuantizer::new(32, 4).unwrap();
        
        let embeddings: Vec<Vec<f32>> = (0..50).map(|i| {
            (0..32).map(|j| (i + j) as f32 * 0.1).collect()
        }).collect();
        
        quantizer.train(&embeddings, 5).unwrap();
        
        let query = embeddings[0].clone();
        let codes = quantizer.encode(&embeddings[10]).unwrap();
        
        let distance = quantizer.asymmetric_distance(&query, &codes).unwrap();
        assert!(distance >= 0.0, "Distance should be non-negative");
        assert!(distance.is_finite(), "Distance should be finite");
    }

    #[test]
    fn test_encode_decode_cycle_accuracy() {
        let mut quantizer = ProductQuantizer::new(16, 4).unwrap();
        
        // Use simple embeddings for predictable results
        let embeddings = vec![
            vec![1.0; 16],
            vec![2.0; 16],
            vec![3.0; 16],
        ];
        
        quantizer.train(&embeddings, 10).unwrap();
        
        for embedding in &embeddings {
            let codes = quantizer.encode(embedding).unwrap();
            let reconstructed = quantizer.decode(&codes).unwrap();
            
            // Calculate reconstruction error
            let mse: f32 = embedding.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>() / embedding.len() as f32;
            
            // For simple constant embeddings, reconstruction should be quite accurate
            assert!(mse < 1.0, "Mean squared error too high: {mse} for embedding {embedding:?}");
        }
    }

    #[test]
    fn test_compression_stats() {
        let quantizer = ProductQuantizer::new(128, 16).unwrap();
        let stats = quantizer.compression_stats(128);
        
        assert_eq!(stats.original_bytes, 128 * 4); // 128 floats * 4 bytes
        assert_eq!(stats.compressed_bytes, 16); // 16 subvectors * 1 byte
        assert_eq!(stats.compression_ratio, 32.0); // 512 / 16 = 32
        assert_eq!(stats.subvector_count, 16);
        assert_eq!(stats.cluster_count, 256);
    }

    #[test]
    fn test_quantized_embedding_storage() {
        let mut storage = QuantizedEmbeddingStorage::new();
        
        let entity1 = EntityKey::new(1.to_string());
        let entity2 = EntityKey::new(2.to_string());
        let codes1 = vec![1, 2, 3, 4];
        let codes2 = vec![5, 6, 7, 8];
        
        storage.add_quantized(entity1, codes1.clone());
        storage.add_quantized(entity2, codes2.clone());
        
        assert_eq!(storage.entity_count, 2);
        assert_eq!(storage.get_quantized(&entity1), Some(&codes1));
        assert_eq!(storage.get_quantized(&entity2), Some(&codes2));
        
        let compression_ratio = storage.compression_ratio(64); // Assume original dimension 64
        assert!(compression_ratio > 1.0, "Should have compression");
    }

    #[test]
    fn test_batch_operations() {
        let mut quantizer = ProductQuantizer::new(32, 8).unwrap();
        
        let embeddings: Vec<Vec<f32>> = (0..20).map(|i| {
            (0..32).map(|j| (i * j) as f32 * 0.05).collect()
        }).collect();
        
        quantizer.train(&embeddings[0..10], 5).unwrap();
        
        // Test batch encode
        let codes = quantizer.batch_encode(&embeddings[10..15]).unwrap();
        assert_eq!(codes.len(), 5);
        assert!(codes.iter().all(|c| c.len() == 8));
        
        // Test batch decode
        let reconstructed = quantizer.batch_decode(&codes).unwrap();
        assert_eq!(reconstructed.len(), 5);
        assert!(reconstructed.iter().all(|r| r.len() == 32));
    }

    #[test]
    fn test_batch_store_quantized() {
        let mut quantizer = ProductQuantizer::new(16, 4).unwrap();
        
        let embeddings: Vec<Vec<f32>> = (0..10).map(|i| {
            (0..16).map(|j| (i + j) as f32).collect()
        }).collect();
        
        quantizer.train(&embeddings, 5).unwrap();
        
        let entities_embeddings: Vec<(EntityKey, Vec<f32>)> = embeddings.into_iter()
            .enumerate()
            .map(|(i, emb)| (EntityKey::new((i as u32).to_string()), emb))
            .collect();
        
        quantizer.batch_store_quantized(&entities_embeddings).unwrap();
        
        let (memory_usage, entity_count, compression_ratio) = quantizer.storage_stats();
        assert_eq!(entity_count, 10);
        assert!(memory_usage > 0);
        assert!(compression_ratio > 1.0);
    }

    #[test]
    fn test_quantized_similarity_search() {
        let mut quantizer = ProductQuantizer::new(8, 2).unwrap();
        
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ];
        
        quantizer.train(&embeddings, 5).unwrap();
        
        // Store embeddings
        for (i, embedding) in embeddings.iter().enumerate() {
            quantizer.store_quantized(EntityKey::new((i as u32).to_string()), embedding).unwrap();
        }
        
        // Search for similar embeddings
        let query = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let results = quantizer.quantized_similarity_search(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        // Results should be sorted by similarity (highest first)
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_reconstruction_error() {
        let mut quantizer = ProductQuantizer::new(16, 4).unwrap();
        
        let embeddings: Vec<Vec<f32>> = (0..20).map(|i| {
            (0..16).map(|j| (i as f32 * 0.1) + (j as f32 * 0.01)).collect()
        }).collect();
        
        quantizer.train(&embeddings[0..15], 10).unwrap();
        
        let error = quantizer.compute_reconstruction_error(&embeddings[15..20]).unwrap();
        assert!(error >= 0.0, "Reconstruction error should be non-negative");
        assert!(error.is_finite(), "Reconstruction error should be finite");
    }

    #[test]
    fn test_new_optimized() {
        // Test different compression targets
        let optimized_8 = ProductQuantizer::new_optimized(128, 8.0).unwrap();
        let optimized_16 = ProductQuantizer::new_optimized(128, 16.0).unwrap();
        let optimized_32 = ProductQuantizer::new_optimized(128, 32.0).unwrap();
        
        // Higher compression targets should use fewer subvectors (more compression)
        assert!(optimized_32.subvector_count <= optimized_16.subvector_count);
        assert!(optimized_16.subvector_count <= optimized_8.subvector_count);
    }

    #[test]
    fn test_train_adaptive() {
        let mut quantizer = ProductQuantizer::new(32, 8).unwrap();
        
        // Create large dataset
        let large_embeddings: Vec<Vec<f32>> = (0..5000).map(|i| {
            (0..32).map(|j| (i + j) as f32 * 0.001).collect()
        }).collect();
        
        // Adaptive training should handle large datasets efficiently
        let start = std::time::Instant::now();
        quantizer.train_adaptive(&large_embeddings).unwrap();
        let elapsed = start.elapsed();
        
        assert!(quantizer.is_trained());
        assert!(elapsed.as_secs() < 10, "Adaptive training took too long: {elapsed:?}");
    }

    #[test]
    fn test_invalid_dimensions() {
        // Test dimension not divisible by subvector count
        let result = ProductQuantizer::new(100, 7);
        assert!(result.is_err());
        
        // Test encode with wrong dimension
        let quantizer = ProductQuantizer::new(16, 4).unwrap();
        let wrong_embedding = vec![1.0; 20]; // Wrong size
        let result = quantizer.encode(&wrong_embedding);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_quality() {
        let mut quantizer = ProductQuantizer::new(16, 4).unwrap();
        
        // Initially not trained
        assert!(!quantizer.is_trained());
        assert_eq!(quantizer.training_quality(), 0.0);
        
        let embeddings: Vec<Vec<f32>> = (0..50).map(|i| {
            (0..16).map(|j| (i + j) as f32 * 0.1).collect()
        }).collect();
        
        quantizer.train(&embeddings, 10).unwrap();
        
        assert!(quantizer.is_trained());
        assert!(quantizer.training_quality() > 0.0);
    }

    #[test]
    fn test_memory_usage() {
        let quantizer = ProductQuantizer::new(64, 8).unwrap();
        let memory_usage = quantizer.memory_usage();
        
        // Should have 8 codebooks * 256 clusters * 8 dimensions * 4 bytes
        let expected = 8 * 256 * 8 * 4;
        assert_eq!(memory_usage, expected);
    }

    #[test]
    fn test_euclidean_distance_function() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        
        let distance = euclidean_distance(&a, &b);
        // sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        let expected = (9.0 + 16.0 + 25.0_f32).sqrt();
        assert!((distance - expected).abs() < 1e-6);
        
        // Distance to self should be 0
        let self_distance = euclidean_distance(&a, &a);
        assert_eq!(self_distance, 0.0);
    }

    #[test]
    fn test_codebook_access() {
        let quantizer = ProductQuantizer::new(32, 4).unwrap();
        
        assert_eq!(quantizer.num_subspaces(), 4);
        
        for i in 0..4 {
            let codebook = quantizer.get_codebook(i);
            // Each codebook should have 256 clusters * 8 dimensions
            assert_eq!(codebook.len(), 256 * 8);
        }
    }

    #[test]
    fn test_quantized_embedding_storage_methods() {
        let mut storage = QuantizedEmbeddingStorage::new();
        
        // Test initial state
        assert_eq!(storage.entity_count, 0);
        assert_eq!(storage.memory_usage(), 0);
        
        // Add some quantized embeddings
        let entity1 = EntityKey::new(100.to_string());
        let entity2 = EntityKey::new(200.to_string());
        let codes1 = vec![1, 2, 3, 4, 5];
        let codes2 = vec![6, 7, 8, 9, 10];
        
        storage.add_quantized(entity1, codes1.clone());
        storage.add_quantized(entity2, codes2.clone());
        
        assert_eq!(storage.entity_count, 2);
        assert!(storage.memory_usage() > 0);
        
        // Test retrieval
        assert_eq!(storage.get_quantized(&entity1), Some(&codes1));
        assert_eq!(storage.get_quantized(&entity2), Some(&codes2));
        
        let nonexistent = EntityKey::new(999.to_string());
        assert_eq!(storage.get_quantized(&nonexistent), None);
        
        // Test building search index
        storage.build_search_index();
        assert!(storage.index.is_some());
        
        // Test compression ratio calculation
        let ratio = storage.compression_ratio(64); // Original dimension 64
        assert!(ratio > 1.0, "Should have compression ratio > 1.0");
    }

    #[test]
    fn test_compression_stats_accuracy() {
        let quantizer = ProductQuantizer::new(256, 32).unwrap();
        let stats = quantizer.compression_stats(256);
        
        // Verify all fields are reasonable
        assert_eq!(stats.original_bytes, 256 * 4); // 256 floats * 4 bytes each
        assert_eq!(stats.compressed_bytes, 32); // 32 subvectors * 1 byte each
        assert_eq!(stats.compression_ratio, 32.0); // 1024 / 32
        assert!(stats.codebook_memory > 0);
        assert_eq!(stats.subvector_count, 32);
        assert_eq!(stats.cluster_count, 256);
        assert_eq!(stats.memory_saved, 1024 - 32);
    }
}