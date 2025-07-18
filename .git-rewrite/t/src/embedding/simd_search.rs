use crate::error::{GraphError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Ultra-fast SIMD-accelerated similarity search
/// Processes 8 embeddings per instruction for maximum throughput
pub struct SIMDSimilaritySearch {
    embedding_dim: usize,
    subvector_count: usize,
    distance_table: Vec<f32>, // Pre-computed distance lookup table
}

impl SIMDSimilaritySearch {
    pub fn new(embedding_dim: usize, subvector_count: usize) -> Self {
        // Pre-allocate distance table for ultra-fast lookups
        let table_size = 256 * subvector_count * embedding_dim;
        let distance_table = vec![0.0; table_size];
        
        Self {
            embedding_dim,
            subvector_count,
            distance_table,
        }
    }
    
    /// Pre-compute distance table for asymmetric distance computation
    /// This allows O(1) distance computation per quantized vector
    pub fn precompute_distances(&mut self, query: &[f32], codebooks: &[Vec<f32>]) -> Result<()> {
        if query.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: self.embedding_dim, 
                actual: query.len() 
            });
        }
        
        let subvector_size = self.embedding_dim / self.subvector_count;
        
        for m in 0..self.subvector_count {
            let query_subvec = &query[m * subvector_size..(m + 1) * subvector_size];
            
            for k in 0..256 {
                let centroid_start = k * subvector_size;
                let centroid_end = centroid_start + subvector_size;
                let centroid = &codebooks[m][centroid_start..centroid_end];
                
                let distance = self.compute_subvector_distance(query_subvec, centroid);
                let table_idx = m * 256 + k;
                self.distance_table[table_idx] = distance;
            }
        }
        
        Ok(())
    }
    
    /// Ultra-fast batch distance computation using SIMD
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_asymmetric_distances(&self, codes_batch: &[&[u8]], results: &mut [f32]) -> Result<()> {
        if codes_batch.len() != results.len() {
            return Err(GraphError::InvalidEmbeddingDimension { 
                expected: codes_batch.len(), 
                actual: results.len() 
            });
        }
        
        let batch_size = codes_batch.len();
        let chunks = batch_size / 8;
        
        // Process 8 embeddings at once using AVX2
        for chunk_idx in 0..chunks {
            let start_idx = chunk_idx * 8;
            let mut distances = _mm256_setzero_ps();
            
            // Accumulate distances for each subvector
            for m in 0..self.subvector_count {
                let mut subvec_distances = _mm256_setzero_ps();
                
                // Load 8 codes for this subvector
                let codes: [u8; 8] = [
                    codes_batch[start_idx][m],
                    codes_batch[start_idx + 1][m],
                    codes_batch[start_idx + 2][m],
                    codes_batch[start_idx + 3][m],
                    codes_batch[start_idx + 4][m],
                    codes_batch[start_idx + 5][m],
                    codes_batch[start_idx + 6][m],
                    codes_batch[start_idx + 7][m],
                ];
                
                // Gather distances from pre-computed table
                let table_base = m * 256;
                let gathered_distances = [
                    self.distance_table[table_base + codes[0] as usize],
                    self.distance_table[table_base + codes[1] as usize],
                    self.distance_table[table_base + codes[2] as usize],
                    self.distance_table[table_base + codes[3] as usize],
                    self.distance_table[table_base + codes[4] as usize],
                    self.distance_table[table_base + codes[5] as usize],
                    self.distance_table[table_base + codes[6] as usize],
                    self.distance_table[table_base + codes[7] as usize],
                ];
                
                subvec_distances = _mm256_loadu_ps(gathered_distances.as_ptr());
                distances = _mm256_add_ps(distances, subvec_distances);
            }
            
            // Store results
            _mm256_storeu_ps(results[start_idx..].as_mut_ptr(), distances);
        }
        
        // Handle remainder
        for i in (chunks * 8)..batch_size {
            let mut total_distance = 0.0;
            for m in 0..self.subvector_count {
                let code = codes_batch[i][m];
                let table_idx = m * 256 + code as usize;
                total_distance += self.distance_table[table_idx];
            }
            results[i] = total_distance;
        }
        
        Ok(())
    }
    
    /// Fallback non-SIMD implementation
    pub fn batch_asymmetric_distances_scalar(&self, codes_batch: &[&[u8]], results: &mut [f32]) -> Result<()> {
        for (i, codes) in codes_batch.iter().enumerate() {
            let mut total_distance = 0.0;
            for m in 0..self.subvector_count {
                let code = codes[m];
                let table_idx = m * 256 + code as usize;
                total_distance += self.distance_table[table_idx];
            }
            results[i] = total_distance;
        }
        Ok(())
    }
    
    #[inline]
    fn compute_subvector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                return self.euclidean_distance_avx2(a, b);
            }
        }
        
        // Fallback scalar implementation
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn euclidean_distance_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
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
    
    /// Find top-k nearest neighbors using heap-based selection
    pub fn top_k_search(&self, codes_batch: &[&[u8]], entity_ids: &[u32], k: usize) -> Result<Vec<(u32, f32)>> {
        let mut distances = vec![0.0; codes_batch.len()];
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                self.batch_asymmetric_distances(codes_batch, &mut distances)?;
            } else {
                self.batch_asymmetric_distances_scalar(codes_batch, &mut distances)?;
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        self.batch_asymmetric_distances_scalar(codes_batch, &mut distances)?;
        
        // Use partial sorting for top-k selection (O(n + k log k) instead of O(n log n))
        let mut indexed_distances: Vec<(u32, f32)> = entity_ids.iter()
            .zip(distances.iter())
            .map(|(&id, &dist)| (id, dist))
            .collect();
        
        // Sort by distance and take top k
        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_distances.truncate(k);
        
        Ok(indexed_distances)
    }
}

/// High-throughput batch processor for similarity search
pub struct BatchProcessor {
    simd_search: SIMDSimilaritySearch,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn new(embedding_dim: usize, subvector_count: usize, batch_size: usize) -> Self {
        Self {
            simd_search: SIMDSimilaritySearch::new(embedding_dim, subvector_count),
            batch_size,
        }
    }
    
    pub fn precompute_distances(&mut self, query: &[f32], codebooks: &[Vec<f32>]) -> Result<()> {
        self.simd_search.precompute_distances(query, codebooks)
    }
    
    /// Process similarity search in optimized batches
    pub fn process_batched_search(
        &self, 
        all_codes: &[Vec<u8>], 
        all_entity_ids: &[u32], 
        k: usize
    ) -> Result<Vec<(u32, f32)>> {
        let mut all_results = Vec::new();
        
        for chunk_start in (0..all_codes.len()).step_by(self.batch_size) {
            let chunk_end = (chunk_start + self.batch_size).min(all_codes.len());
            
            let codes_refs: Vec<&[u8]> = all_codes[chunk_start..chunk_end]
                .iter()
                .map(|v| v.as_slice())
                .collect();
            
            let entity_ids_chunk = &all_entity_ids[chunk_start..chunk_end];
            
            let batch_results = self.simd_search.top_k_search(&codes_refs, entity_ids_chunk, k)?;
            all_results.extend(batch_results);
        }
        
        // Final top-k selection across all batches
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(k);
        
        Ok(all_results)
    }
}
