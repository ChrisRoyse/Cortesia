use crate::error::{GraphError, Result};
use std::f32;

pub struct ProductQuantizer {
    subvector_count: usize,      // M subvectors
    subvector_size: usize,       // D/M dimensions per subvector
    cluster_count: usize,        // K clusters per subvector (256 for u8 codes)
    codebooks: Vec<Vec<f32>>,    // M codebooks of K centroids
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
        })
    }
    
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
        
        // Train each subquantizer independently
        for m in 0..self.subvector_count {
            let start = m * self.subvector_size;
            let end = start + self.subvector_size;
            
            // Extract subvectors for this quantizer
            let subvectors: Vec<Vec<f32>> = embeddings.iter()
                .map(|emb| emb[start..end].to_vec())
                .collect();
            
            // Run k-means clustering
            self.train_subquantizer(m, &subvectors, iterations)?;
        }
        
        Ok(())
    }
    
    fn train_subquantizer(&mut self, quantizer_idx: usize, subvectors: &[Vec<f32>], iterations: usize) -> Result<()> {
        let k = self.cluster_count;
        let dim = self.subvector_size;
        
        // Initialize centroids with random subvectors
        let mut centroids = vec![vec![0.0; dim]; k];
        let step = subvectors.len() / k;
        for i in 0..k {
            centroids[i] = subvectors[i * step].clone();
        }
        
        let mut assignments = vec![0usize; subvectors.len()];
        
        for _ in 0..iterations {
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
            
            if !changed {
                break;
            }
            
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
        let codebook_start = quantizer_idx * self.subvector_size * self.cluster_count;
        for (i, centroid) in centroids.iter().enumerate() {
            let offset = i * self.subvector_size;
            self.codebooks[quantizer_idx][offset..offset + self.subvector_size]
                .copy_from_slice(centroid);
        }
        
        Ok(())
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
            
            for (i, (&q, &c)) in chunk.iter().zip(centroid.iter()).enumerate() {
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